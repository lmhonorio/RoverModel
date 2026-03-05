"""
Approximate Generalized Voronoi Diagram (GVD) built from obstacle boundaries.

Approach (Option A):
- Build shapely polygons for each obstacle using center `pos` and `size`.
- Sample points along each obstacle polygon boundary (configurable density).
- Compute `scipy.spatial.Voronoi` of the boundary samples (these are the "sites").
- For each Voronoi ridge (finite or infinite):
  - Convert ridge to a (possibly extended) LineString.
  - Intersect the ridge with the free-space polygon (workspace bbox minus union(obstacle_polygons)).
  - Keep the resulting segments (parts of the ridge that lie in free space) and add them to a NetworkX graph.
- Save resulting graph (JSON) and a visualization PNG in `Resultados_artigo`.

Notes:
- This produces an approximation of a GVD by using sampled boundary points as sites.
- Uses `shapely` to handle geometry operations (intersection, union, clipping).

Usage: run from repository root: `python Resultados_artigo/gvd_from_equipment.py`
"""

import os
import math
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, cKDTree
import networkx as nx
from shapely.geometry import Polygon, LineString, Point, box
from shapely.ops import unary_union

# Ensure repo root is importable
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in os.sys.path:
    os.sys.path.insert(0, REPO_ROOT)

from CriarPontosObservacao import build_graph
from roverclass import ObstacleLoader
from aabbutils import AABBUtils

RESULTS_DIR = os.path.dirname(__file__)

from shapely.geometry import Point, box
from shapely.geometry import JOIN_STYLE

def obstacle_polygons_from_obstacles(obstacles, margin=0.0):
    """Create shapely Polygons from obstacles entries.

    Assumes each `obs` has:
      - 'pos': (x, y) center
      - 'size': (w, h)
    Applies a uniform margin to each polygon with square corners.
    """
    polys = []
    for obs in obstacles:
        cx, cy = obs['pos']
        w, h = obs.get('size', (0.0, 0.0))

        # Obstáculos pontuais ou sem tamanho válido → vira círculo com margem
        if w is None or h is None or w <= 0 or h <= 0:
            polys.append(Point(cx, cy).buffer(0.5 + margin))
            continue

        # Monta retângulo centrado em (cx, cy)
        x0 = cx - w / 2.0
        y0 = cy - h / 2.0
        rect = box(x0, y0, x0 + w, y0 + h)

        # Aplica margem sem cantos arredondados
        if margin != 0:
            rect = rect.buffer(margin, join_style=JOIN_STYLE.mitre)

        polys.append(rect)

    return polys


def sample_boundary_points(polygons, points_per_meter=1.0, min_points_per_poly=8):
    """Sample points along polygon boundaries.

    - `points_per_meter` controls density; approximate number of points per unit length.
    - returns Nx2 numpy array of (x,y) coordinates.
    """
    samples = []
    for poly in polygons:
        perim = poly.length
        n = max(min_points_per_poly, int(math.ceil(perim * points_per_meter)))
        if n < 4:
            n = 4
        # Parametrize along boundary
        for frac in np.linspace(0.0, 1.0, n, endpoint=False):
            pt = poly.exterior.interpolate(frac, normalized=True)
            samples.append((pt.x, pt.y))
    return np.array(samples)


def workspace_bbox_from_polygons(polygons, padding=5.0):
    xs = []
    ys = []
    for p in polygons:
        minx, miny, maxx, maxy = p.bounds
        xs.extend([minx, maxx])
        ys.extend([miny, maxy])
    if not xs:
        return (-padding, padding, -padding, padding)
    xmin = min(xs) - padding
    xmax = max(xs) + padding
    ymin = min(ys) - padding
    ymax = max(ys) + padding
    return xmin, xmax, ymin, ymax


def extend_infinite_ridge(point, direction, bbox, max_dist=None):
    """Return a LineString starting from `point` extended along +/- `direction` and clipped to bbox."""
    if max_dist is None:
        xmin, xmax, ymin, ymax = bbox
        max_dist = max(xmax - xmin, ymax - ymin) * 3.0
    a = (point[0] - direction[0] * max_dist, point[1] - direction[1] * max_dist)
    b = (point[0] + direction[0] * max_dist, point[1] + direction[1] * max_dist)
    return LineString([a, b])


def connect_islands(G, obstacles_union):
    """Reduce graph fragmentation by connecting isolated components with obstacle-free edges.
    
    Optimized approach:
    - Uses KDTree for fast nearest-neighbor search.
    - Uses shapely.prepared for fast geometry checks.
    - Attempts to connect all islands in fewer passes.
    """
    try:
        from shapely.prepared import prep
        prepared_obstacles = prep(obstacles_union)
    except ImportError:
        prepared_obstacles = obstacles_union

    # Initial search parameter
    current_k = 5

    # Iterate until graph is connected or no more connections can be made
    while True:
        components = list(nx.connected_components(G))
        if len(components) <= 1:
            break

        # Sort components by size (largest is main)
        components.sort(key=len, reverse=True)
        main_comp = components[0]
        main_nodes_list = list(main_comp)
        
        # Build KDTree for main component
        main_positions = [G.nodes[n]['pos'] for n in main_nodes_list]
        tree = cKDTree(main_positions)

        edges_added = 0
        
        # Try to connect each island to the main component
        for comp in components[1:]:
            island_nodes = list(comp)
            island_positions = [G.nodes[n]['pos'] for n in island_nodes]

            # Query k nearest neighbors in main component
            k_neighbors = min(current_k, len(main_nodes_list))
            dists_arr, indices_arr = tree.query(island_positions, k=k_neighbors)
            
            # Normalize shapes
            if len(island_nodes) == 1:
                dists_arr = [[dists_arr]] if k_neighbors == 1 else [dists_arr]
                indices_arr = [[indices_arr]] if k_neighbors == 1 else [indices_arr]
            elif k_neighbors == 1:
                dists_arr = [[d] for d in dists_arr]
                indices_arr = [[i] for i in indices_arr]

            candidates = []
            for i, (ds, idxs) in enumerate(zip(dists_arr, indices_arr)):
                if np.isscalar(ds): ds, idxs = [ds], [idxs]
                for d, main_idx in zip(ds, idxs):
                    if d == float('inf') or main_idx >= len(main_nodes_list): continue
                    candidates.append((d, i, main_idx))
            
            candidates.sort(key=lambda x: x[0])

            for d, i_idx, m_idx in candidates:
                u = island_nodes[i_idx]
                v = main_nodes_list[m_idx]
                line = LineString([G.nodes[u]['pos'], G.nodes[v]['pos']])
                
                if prepared_obstacles.intersects(line):
                    inter = line.intersection(obstacles_union)
                    if not (inter.is_empty or inter.length < 1e-6):
                        continue
                
                G.add_edge(u, v, weight=d)
                edges_added += 1
                break
        
        if edges_added == 0:
            # If no edges added, progressively increase search depth
            if current_k < 70:
                current_k = 70
                continue
            break
        else:
            # Reset k to keep performance high for easy connections
            current_k = 5

    return G


def build_gvd_by_clipping(polygons, boundary_samples, bbox_padding=5.0):
    """Build approximate GVD graph by computing Voronoi of boundary samples and clipping ridges to free space."""
    if boundary_samples is None or len(boundary_samples) < 2:
        return nx.Graph(), None, None

    vor = Voronoi(boundary_samples)

    # Workspace bbox and free-space polygon
    xmin, xmax, ymin, ymax = workspace_bbox_from_polygons(polygons, padding=bbox_padding)
    workspace_poly = box(xmin, ymin, xmax, ymax)
    obstacles_union = unary_union(polygons) if polygons else Polygon()
    free_space = workspace_poly.difference(obstacles_union)

    G = nx.Graph()
    vert_map = {}

    def add_point(pt):
        key = (float(pt[0]), float(pt[1]))
        if key not in vert_map:
            vid = f"v{len(vert_map)}"
            vert_map[key] = vid
            G.add_node(vid, pos=key)
        return vert_map[key]

    # helper to add segment(s) from a LineString (possibly MultiLineString) after clipping with free_space
    def add_linestring_clipped(ls):
        inter = ls.intersection(free_space)
        if inter.is_empty:
            return
        # Handle MultiLineString or LineString
        geoms = [inter] if inter.geom_type == 'LineString' else list(inter.geoms)
        for g in geoms:
            if g.length == 0:
                continue
            coords = list(g.coords)
            # add small segments between consecutive coords
            for i in range(len(coords) - 1):
                a = coords[i]
                b = coords[i + 1]
                id_a = add_point(a)
                id_b = add_point(b)
                d = math.dist(a, b)
                if d > 0:
                    G.add_edge(id_a, id_b, weight=d)
                # Só adiciona se forem nós diferentes e distância > 0
                # if id_a != id_b and d > 0:
                #     G.add_edge(id_a, id_b, weight=d)

    # iterate ridges
    center_points = vor.points
    for (pidx0, pidx1), ridge_vertices in zip(vor.ridge_points, vor.ridge_vertices):
        if -1 in ridge_vertices:
            # infinite ridge: find finite vertex and direction
            finite_idx = next((rv for rv in ridge_vertices if rv != -1), None)
            if finite_idx is None:
                continue
            finite_v = vor.vertices[finite_idx]
            # direction perpendicular to the line between the two sites
            p0 = center_points[pidx0]
            p1 = center_points[pidx1]
            dir_vec = np.array([p0[1] - p1[1], p1[0] - p0[0]], dtype=float)
            norm = np.linalg.norm(dir_vec)
            if norm == 0:
                continue
            dir_vec /= norm
            ls = extend_infinite_ridge(finite_v, dir_vec, (xmin, xmax, ymin, ymax))
            add_linestring_clipped(ls)
        else:
            v0 = vor.vertices[ridge_vertices[0]]
            v1 = vor.vertices[ridge_vertices[1]]
            ls = LineString([tuple(v0), tuple(v1)])
            add_linestring_clipped(ls)

    # Remove nodes that landed inside obstacles (floating-point artifacts from clipping)
    nodes_inside = [nid for nid, data in G.nodes(data=True)
                    if obstacles_union.contains(Point(data['pos']))]
    G.remove_nodes_from(nodes_inside)

    # Add one node per polygon corner, offset by a tiny amount outward so the node
    # sits strictly in free space (not on the obstacle boundary) while remaining
    # visually indistinguishable from the actual corner.
    corner_offset = 0.05  # meters — imperceptible visually, but clears shapely boundary
    node_positions = [(nid, data['pos']) for nid, data in G.nodes(data=True)]
    for poly in polygons:
        centroid = poly.centroid
        corner_coords = list(poly.exterior.coords)[:-1]  # drop repeated closing vertex
        for cx, cy in corner_coords:
            dx, dy = cx - centroid.x, cy - centroid.y
            length = math.hypot(dx, dy)
            if length == 0:
                continue
            ocx = cx + dx / length * corner_offset
            ocy = cy + dy / length * corner_offset
            if not free_space.contains(Point(ocx, ocy)):
                continue  # squeezed between obstacles — skip
            key = (round(ocx, 6), round(ocy, 6))
            if key in vert_map:
                corner_id = vert_map[key]
            else:
                corner_id = f"v{len(vert_map)}"
                vert_map[key] = corner_id
                G.add_node(corner_id, pos=key)
                node_positions.append((corner_id, key))
            # Connect corner to the 2 nearest existing GVD nodes (one per side)
            candidates = sorted(
                [(math.dist(key, npos), nid) for nid, npos in node_positions if nid != corner_id]
            )
            for d, nid in candidates[:2]:
                G.add_edge(corner_id, nid, weight=d)

    # Connect remaining islands with obstacle-free edges
    G = connect_islands(G, obstacles_union)

    return G, free_space, obstacles_union


def plot_gvd_graph(G, free_space, obstacles_union, out_png, show=False, overlay_graph=None, obs_points=None):
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot free space and obstacles
    try:
        if free_space is not None and not free_space.is_empty:
            if free_space.geom_type == 'Polygon':
                xs, ys = free_space.exterior.xy
                ax.fill(xs, ys, facecolor='white', edgecolor='none')
            else:
                for geom in free_space.geoms:
                    xs, ys = geom.exterior.xy
                    ax.fill(xs, ys, facecolor='white', edgecolor='none')
    except Exception:
        pass

    if obstacles_union is not None and not obstacles_union.is_empty:
        if obstacles_union.geom_type == 'Polygon':
            xs, ys = obstacles_union.exterior.xy
            ax.fill(xs, ys, facecolor='lightgray', edgecolor='k', alpha=0.8)
        else:
            for geom in obstacles_union.geoms:
                xs, ys = geom.exterior.xy
                ax.fill(xs, ys, facecolor='lightgray', edgecolor='k', alpha=0.8)

    # Plot GVD edges and vertices
    for u, v in G.edges():
        x1, y1 = G.nodes[u]['pos']
        x2, y2 = G.nodes[v]['pos']
        ax.plot([x1, x2], [y1, y2], c='black', linewidth=0.8)

    vx = [p[0] for _, p in nx.get_node_attributes(G, 'pos').items()]
    vy = [p[1] for _, p in nx.get_node_attributes(G, 'pos').items()]
    ax.scatter(vx, vy, c='blue', s=6)

    # Optionally overlay the original graph (from CriarPontosObservacao) and observation points
    if overlay_graph is not None:
        try:
            for u, v in overlay_graph.edges():
                pu = overlay_graph.nodes[u].get('pos')
                pv = overlay_graph.nodes[v].get('pos')
                if pu is not None and pv is not None:
                    ax.plot([pu[0], pv[0]], [pu[1], pv[1]], c='red', linewidth=0.5, alpha=0.6)
        except Exception:
            pass

    if obs_points is not None:
        try:
            xs = [p[0] for p in obs_points]
            ys = [p[1] for p in obs_points]
            ax.scatter(xs, ys, c='green', s=8, label='obs points')
        except Exception:
            pass

    ax.set_aspect('equal', adjustable='box')
    ax.set_title('Approximate GVD (Voronoi of obstacle boundaries, clipped to free space)')
    ax.set_xlabel('X (meters)', fontsize=16)
    ax.set_ylabel('Y (meters)', fontsize=16)
    ax.tick_params(axis='both', which='major', labelsize=14)
    # Legend (if obs points are plotted)
    try:
        ax.legend(fontsize=14)
    except Exception:
        pass
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    try:
        plt.show()
    except Exception:
        # In non-interactive environments plt.show() may fail; ignore.
        pass
    plt.close(fig)


def save_graph_json(G, filename):
    data = {
        'nodes': {n: {'pos': list(data['pos'])} for n, data in G.nodes(data=True)},
        'edges': [(u, v, float(d.get('weight', 1.0))) for u, v, d in G.edges(data=True)]
    }
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--show', action='store_true', help='Show plot interactively after creating it')
    parser.add_argument('--overlay', action='store_true', help='Overlay original graph and observation points')
    parser.add_argument('--margin', type=float, default=1.5, help='Security margin around obstacles')
    parser.add_argument('--points-density', type=float, default=1.0, help='Boundary sampling density (points per meter)')
    args = parser.parse_args()

    file_path = os.path.abspath(os.path.join(RESULTS_DIR, '..', 'planilhas', 'equipment_processado.xlsx'))
    sheet_name = 'Parnaiba3_Transformado'

    print('Loading obstacles from sheet...')
    loader = ObstacleLoader(file_path, sheet_name)
    obstacles = loader.get_obstacles()
    print(f'Loaded {len(obstacles)} obstacles')

    # Build polygons from obstacles
    polygons = obstacle_polygons_from_obstacles(obstacles, margin=args.margin)

    # Sample boundary points (tune density via points_per_meter)
    boundary_samples = sample_boundary_points(polygons, points_per_meter=args.points_density, min_points_per_poly=12)
    print(f'Sampled {len(boundary_samples)} boundary points')

    print('Computing Voronoi and clipping ridges to free space...')
    G, free_space, obstacles_union = build_gvd_by_clipping(polygons, boundary_samples, bbox_padding=10.0)
    print(f'GVD graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges')

    out_png = os.path.join(RESULTS_DIR, 'gvd_equipment.png')
    out_json = os.path.join(RESULTS_DIR, 'gvd_equipment.json')

    # Optionally load original graph and observation points for overlay
    overlay_graph = None
    obs_points = None
    if args.overlay:
        try:
            print('Building original graph (for overlay) using CriarPontosObservacao.build_graph...')
            G_orig, aabbs, obs_pts_list, obstacles2 = build_graph(file_path, sheet_name, margin=args.margin, threshold=30, plotting=False)
            overlay_graph = G_orig
            # obs_pts_list is a list of (x,y,label) -> convert to (x,y)
            try:
                obs_points = [(x, y) for x, y, lab in obs_pts_list]
            except Exception:
                obs_points = None
        except Exception as e:
            print(f'Failed to build original graph for overlay: {e}')

    print(f'Saving plot to {out_png}')
    plot_gvd_graph(G, free_space, obstacles_union, out_png, show=args.show, overlay_graph=overlay_graph, obs_points=obs_points)

    print(f'Saving graph JSON to {out_json}')
    save_graph_json(G, out_json)

    print('Done.')


if __name__ == '__main__':
    main()
