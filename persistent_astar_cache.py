import json
import os

# RECUPERA DO ARQUIVO CACHE_ASTAR.JSON TODOS OS CAMINHOS JÁ MAPEADOS CALCULADOS COM A* SOBRE O GRAFO GERADO A PARTIR DAS AABBS

class PersistentAStarCache:
    _instance = None

    def __new__(cls, planner, file_path):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.__initialized = False
        return cls._instance

    def __init__(self, planner, file_path):
        if self.__initialized:
            return
        self.planner = planner
        self.file_path = file_path
        self.cache = self._load_cache()
        self.__initialized = True

    def _load_cache(self):
        if os.path.exists(self.file_path):
            with open(self.file_path, "r") as f:
                raw = json.load(f)
                return {tuple(eval(k)): v for k, v in raw.items()}
        return {}

    def _save_cache(self):
        print("salvando cache")
        raw = {str(k): v for k, v in self.cache.items()}
        with open(self.file_path, "w") as f:
            json.dump(raw, f, indent=2)

    def get_path(self, origem, destino):
        chave = (origem, destino)
        chave_invertida = (destino, origem)

        # Se já existe a chave direta
        if chave in self.cache and self.cache[chave]["path"]:
            return self.cache[chave]["path"], self.cache[chave]["cost"]

        # Se existe a chave inversa, inverte o caminho
        elif chave_invertida in self.cache:
            path_reversed = list(reversed(self.cache[chave_invertida]["path"]))
            cost = self.cache[chave_invertida]["cost"]
            # Opcional: salva também a versão direta para evitar repetir isso depois
            self.cache[chave] = {"path": path_reversed, "cost": cost}
            self._save_cache()
            return path_reversed, cost

        # Caso nenhum exista, calcula com A*
        print("Calculando com astar...")
        path, cost = self.planner.a_star(origem, destino)
        self.cache[chave] = {"path": path, "cost": cost}
        self._save_cache()
        return path, cost

