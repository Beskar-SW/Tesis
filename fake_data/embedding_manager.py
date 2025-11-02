from transformers import AutoTokenizer, AutoModel
import numpy as np
import torch
import json
import os
import re

class EmbeddingManager:
    def __init__(self, modelos, save_dir="faiss_data"):
        self.modelos = modelos
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            print("🚀 Usando GPU CUDA")
        else:
            self.device = torch.device("cpu")
            print("🧠 Usando CPU")
        

    def get_embedding_model(self, model_name):
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        model.to(self.device)
        model.eval()
        return tokenizer, model

    def embed_texts(self, texts, tokenizer, model, batch_size=8):
        embeddings = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            batch_texts = [re.sub(r"\s+", " ", t.lower().strip()) for t in batch_texts]
            inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = model(**inputs)
            
            emb = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
            embeddings.extend(emb)
            
            embeddings = np.array(embeddings)
            embeddings = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)
            
        return embeddings

    def build_embeddings(self, docs, chunk_fn):
        """
        Genera embeddings de cada documento dividiéndolo en chunks.
        Devuelve un diccionario RAM_DB organizado por modelo.
        """
        RAM_DB = {m: {} for m in self.modelos}

        for model_id, model_name in self.modelos.items():
            print(f"\n⚙️ Generando embeddings para modelo {model_id}: {model_name}")
            tokenizer, model = self.get_embedding_model(model_name)

            for doc in docs:
                # 1️⃣ Fragmentar texto
                chunks = chunk_fn(doc["texto"])
                if not chunks:
                    continue

                # 2️⃣ Generar embeddings por chunk
                embeddings = self.embed_texts(chunks, tokenizer, model)

                # 3️⃣ Guardar cada chunk con clave única
                for i, emb in enumerate(embeddings):
                    key = f"{model_id}_{doc['archivo'].replace('.xml','')}_{doc['id_relato']}_{doc['tipo']}_{i}"

                    RAM_DB[model_id][key] = {
                        "archivo": doc["archivo"],
                        "etapa": doc["etapa"],
                        "id_relato": doc["id_relato"],
                        "titulo": doc["titulo"],
                        "tipo": doc["tipo"],
                        "texto": chunks[i],  # ✅ Cada fragmento, no todo el texto
                        "chunk": i,
                        "embedding": emb.tolist(),
                    }

            # 💾 Guardar embeddings del modelo actual
            self.save_embeddings(model_id, RAM_DB[model_id])
            print(f"✅ Embeddings guardados para {model_id}")

        return RAM_DB

    def build_embeddings_for_model(self, model_id, model_name, docs, chunk_fn):
        print(f"\n⚙️ Generando embeddings para modelo {model_id}: {model_name}")
        tokenizer, model = self.get_embedding_model(model_name)

        RAM_DB = {}
        for doc in docs:
            chunks = chunk_fn(doc["texto"])
            if not chunks:
                continue

            embeddings = self.embed_texts(chunks, tokenizer, model)
            for i, emb in enumerate(embeddings):
                key = f"{model_id}_{doc['archivo'].replace('.xml','')}_{doc['id_relato']}_{doc['tipo']}_{i}"
                RAM_DB[key] = {
                    "archivo": doc["archivo"],
                    "etapa": doc["etapa"],
                    "id_relato": doc["id_relato"],
                    "titulo": doc["titulo"],
                    "tipo": doc["tipo"],
                    "texto": chunks[i],
                    "chunk": i,
                    "embedding": emb.tolist(),
                }

        # 💾 guardar embeddings del modelo actual
        self.save_embeddings(model_id, RAM_DB)
        print(f"✅ Embeddings guardados para {model_id}")
        return RAM_DB

    def save_embeddings(self, model_id, data):
        path = os.path.join(self.save_dir, f"{model_id}_embeddings.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        print(f"💾 Guardado embeddings de {model_id}: {path}")

    def load_embeddings(self, model_id):
        path = os.path.join(self.save_dir, f"{model_id}_embeddings.json")
        if not os.path.exists(path):
            return None
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
