from bm25_search import BM25Search
import numpy as np
import faiss
import re

class SemanticSearch:
    def __init__(self, modelos, embedding_manager, faiss_indexer):
        self.modelos = modelos
        self.embedding_manager = embedding_manager
        self.faiss_indexer = faiss_indexer

    def search(self, query, model_id, RAM_DB, top_k=10):
        if model_id not in RAM_DB:
            print(f"No hay datos del modelo {model_id}")
            return

        query = re.sub(r"\s+", " ", query.lower().strip())
        tokenizer, model = self.embedding_manager.get_embedding_model(self.modelos[model_id])
        query_emb = self.embedding_manager.embed_texts([query], tokenizer, model)[0].astype("float32")
        query_emb = query_emb / np.linalg.norm(query_emb)  # ✅ normalización
        query_emb = query_emb.reshape(1, -1)  # Asegurar forma (1, dim)

        index, keys = self.faiss_indexer.load_index(model_id)
        if index is None:
            print("⚙️ Creando índice FAISS...")
            index, keys = self.faiss_indexer.build_index(RAM_DB[model_id], model_id)


        if isinstance(index, faiss.IndexFlatIP):
            faiss.normalize_L2(query_emb)
        
        distances, idxs = index.search(query_emb, k=top_k)

        results = []
        for i, (idx, dist) in enumerate(zip(idxs[0], distances[0])):
            entry = RAM_DB[model_id][keys[idx]]

            if isinstance(index, faiss.IndexFlatIP):
                score = float(dist)  # Para IP, la distancia es similitud coseno
            else:
                score = float(1 / (1 + dist))  # Para L2, convertir distancia a similitud

            results.append({
                "rank": i + 1,
                # "score": float(1 / (1 + dist)),
                "score": score,
                "distance": float(dist),
                **entry
            })
        return results

class HybridSearch(SemanticSearch):
    def __init__(self, modelos, embedding_manager, faiss_indexer, docs):
        super().__init__(modelos, embedding_manager, faiss_indexer)
        self.bm25 = BM25Search(docs)

    def hybrid_search(self, query, model_id, RAM_DB, top_k=10, alpha=0.6):
        """
        Combina BM25 y FAISS.
        alpha: peso entre 0 y 1 (0.6 = 60% embeddings + 40% BM25)
        """
        bm25_results = self.bm25.search(query, top_k * 3)
        semantic_results = self.search(query, model_id, RAM_DB, top_k * 3)

        # Convertir a diccionario por título para fusión
        bm25_dict = {r["titulo"]: r for r in bm25_results}
        semantic_dict = {r["titulo"]: r for r in semantic_results}

        combined = {}
        for title in set(bm25_dict.keys()) | set(semantic_dict.keys()):
            bm25_score = bm25_dict.get(title, {}).get("score", 0)
            sem_score = semantic_dict.get(title, {}).get("score", 0)
            hybrid_score = alpha * sem_score + (1 - alpha) * bm25_score
            combined[title] = {
                "titulo": title,
                "bm25_score": bm25_score,
                "semantic_score": sem_score,
                "score": hybrid_score,
                "etapa": bm25_dict.get(title, semantic_dict.get(title, {})).get("etapa", "desconocida")
            }

        # Ordenar y mostrar
        ranked = sorted(combined.values(), key=lambda x: x["score"], reverse=True)[:top_k]
        # print("\n🔍 Resultados HÍBRIDOS (BM25 + FAISS):")
        # print("=" * 70)
        # for r in ranked:
        #     print(f"#{ranked.index(r)+1} | {r['titulo']} ({r['etapa']}) → {r['score']:.3f}")
        return ranked
