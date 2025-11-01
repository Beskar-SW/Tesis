# model_manager.py
import os
from embedding_manager import EmbeddingManager
from faiss_indexer import FaissIndexer
from dataset_loader import DatasetLoader
from modelos import MODELOS
import json

CONFIG_FILE = "config.json"

class ModelManager:
    def __init__(self, data_dir="data", save_dir="faiss_data"):
        self.dataset = DatasetLoader(data_dir)
        self.emb_manager = EmbeddingManager(MODELOS, save_dir)
        self.faiss_indexer = FaissIndexer(save_dir)
        self.save_dir = save_dir
        self.config = self.load_or_create_config()

    def load_or_create_config(self):
        if not os.path.exists(CONFIG_FILE):
            conf = {"trained_models": [], "indexed_models": []}
            with open(CONFIG_FILE, "w") as f:
                json.dump(conf, f, indent=4)
            return conf
        with open(CONFIG_FILE, "r") as f:
            return json.load(f)

    def save_config(self):
        with open(CONFIG_FILE, "w") as f:
            json.dump(self.config, f, indent=4)

    def prepare_models(self):
        """
        Carga documentos, embeddings e índices. Si algo no existe, lo genera.
        """
        docs = self.dataset.load_documents()
        RAM_DB = {}

        for model_id, model_name in MODELOS.items():
            print(f"\n🧠 Verificando modelo {model_id} ({model_name})")

            # 1️⃣ Intentar cargar embeddings
            data = self.emb_manager.load_embeddings(model_id)
            if data is None:
                print(f"⚙️ Generando embeddings para {model_id}...")
                RAM_DB[model_id] = self.emb_manager.build_embeddings(docs, self.dataset.chunk_text)[model_id]
                self.config["trained_models"].append(model_id)
                self.save_config()
            else:
                RAM_DB[model_id] = data
                print(f"✅ Embeddings cargados para {model_id}")

            # 2️⃣ Intentar cargar índice FAISS
            index, keys = self.faiss_indexer.load_index(model_id)
            if index is None:
                print(f"⚙️ Creando índice FAISS para {model_id}...")
                self.faiss_indexer.build_index(RAM_DB[model_id], model_id)
                self.config["indexed_models"].append(model_id)
                self.save_config()
            else:
                print(f"✅ Índice FAISS existente para {model_id}")

        return docs, RAM_DB
