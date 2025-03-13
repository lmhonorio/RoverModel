import math
import numpy as np
import networkx as nx


###############################################################################
# CLASSE: AABBUtils
###############################################################################
class AABBUtils:



    @staticmethod
    def merge_overlapping_aabbs(aabbs):
        merged = []
        while aabbs:
            base = aabbs.pop(0)
            bx, by = base[0]
            bw, bh = base[1], base[2]
            merged_flag = False

            for i, (other_pos, other_w, other_h) in enumerate(merged):
                ox, oy = other_pos
                # Se sobrepõem
                if not (bx + bw < ox or ox + other_w < bx or by + bh < oy or oy + other_h < by):
                    new_x = min(bx, ox)
                    new_y = min(by, oy)
                    new_w = max(bx + bw, ox + other_w) - new_x
                    new_h = max(by + bh, oy + other_h) - new_y
                    merged[i] = ((new_x, new_y), new_w, new_h)
                    merged_flag = True
                    break

            if not merged_flag:
                merged.append(base)
        return merged

    @staticmethod
    def get_aabbs(obstacles, margin):
        """
        Cria AABBs a partir de obstacles, adicionando 'margin'.
        Retorna lista [((ax, ay), w, h, label), ...].
        """
        aabbs = []
        for obs in obstacles:
            x, y = obs["pos"]
            w, h = obs["size"]

            aabb_x = x - (w / 2 + margin)
            aabb_y = y - (h / 2 + margin)
            aabb_w = w + 2 * margin
            aabb_h = h + 2 * margin

            # Agora cada AABB carrega o label do obstáculo original
            aabbs.append(((aabb_x, aabb_y), aabb_w, aabb_h))

        # Dupla fusão para garantir, mantendo os labels junto dos AABBs
        merged_aabbs = AABBUtils.merge_overlapping_aabbs(
            AABBUtils.merge_overlapping_aabbs(aabbs)
        )

        return merged_aabbs

    @staticmethod
    def distance_between_aabbs(aabb1, aabb2):
        """
        Distância min entre 2 AABBs (ax, ay, w, h). Se sobrepõem => 0.
        """
        ((ax1, ay1), w1, h1) = aabb1
        ((ax2, ay2), w2, h2) = aabb2

        x1_min, x1_max = ax1, ax1 + w1
        y1_min, y1_max = ay1, ay1 + h1
        x2_min, x2_max = ax2, ax2 + w2
        y2_min, y2_max = ay2, ay2 + h2

        # sobrepõe => 0
        overlap_x = not (x1_max < x2_min or x2_max < x1_min)
        overlap_y = not (y1_max < y2_min or y2_max < y1_min)
        if overlap_x and overlap_y:
            return 0.0

        # dist em X
        if x1_max < x2_min:
            dx = x2_min - x1_max
        elif x2_max < x1_min:
            dx = x1_min - x2_max
        else:
            dx = 0.0

        # dist em Y
        if y1_max < y2_min:
            dy = y2_min - y1_max
        elif y2_max < y1_min:
            dy = y1_min - y2_max
        else:
            dy = 0.0

        return math.hypot(dx, dy)

    @staticmethod
    def cluster_aabbs_scipy(aabbs, threshold, method='single'):
        """
        Clusteriza AABBs via scipy (hierárquico).
        Retorna array 'labels'.
        """
        n = len(aabbs)
        if n <= 1:
            return [1]*n

        dist_matrix = np.zeros((n,n))
        for i in range(n):
            for j in range(i+1, n):
                dist = AABBUtils.distance_between_aabbs(aabbs[i], aabbs[j])
                dist_matrix[i,j] = dist
                dist_matrix[j,i] = dist

        from scipy.spatial.distance import squareform
        from scipy.cluster.hierarchy import linkage, fcluster

        dist_cond = squareform(dist_matrix, checks=False)
        Z = linkage(dist_cond, method=method)
        labels = fcluster(Z, t=threshold, criterion='distance')
        return labels