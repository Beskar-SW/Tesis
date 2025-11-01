import faiss
import numpy as np
import os

class FaissIndexer:
    def __init__(self, save_dir="faiss_data"):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

    def build_index(self, embeddings_dict, model_id):
        first_emb = np.array(next(iter(embeddings_dict.values()))["embedding"], dtype="float32")
        dim = first_emb.shape[0]

        index = faiss.IndexFlatIP(dim)  
        # index = faiss.IndexFlatL2(dim)
        keys, vectors = [], []
        for k, v in embeddings_dict.items():
            keys.append(k)
            # vectors.append(np.array(v["embedding"], dtype="float32"))
            vec = np.array(v["embedding"], dtype="float32")
            vec /= np.linalg.norm(vec) + 1e-8  # ✅ normalización individual
            vectors.append(vec)
        
        matrix = np.vstack(vectors).astype("float32")
        faiss.normalize_L2(matrix)  # seguridad extra
        index.add(matrix)
        # faiss.normalize_L2(np.vstack(vectors))
        # index.add(np.vstack(vectors))
        self.save_index(model_id, index, keys)
        return index, keys

    def save_index(self, model_id, index, keys):
        index_path = os.path.join(self.save_dir, f"{model_id}.index")
        keys_path = os.path.join(self.save_dir, f"{model_id}_keys.npy")
        faiss.write_index(index, index_path)
        np.save(keys_path, np.array(keys))
        print(f"💾 Guardado índice FAISS: {index_path}")

    def load_index(self, model_id):
        index_path = os.path.join(self.save_dir, f"{model_id}.index")
        keys_path = os.path.join(self.save_dir, f"{model_id}_keys.npy")
        if not os.path.exists(index_path):
            return None, None
        index = faiss.read_index(index_path)
        keys = np.load(keys_path, allow_pickle=True).tolist()
        return index, keys
