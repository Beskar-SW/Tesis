# evaluation_manager.py
import numpy as np
from tqdm import tqdm
import json
import os

class EvaluationManager:
    def __init__(self, modelos, embedding_manager, faiss_indexer):
        self.modelos = modelos
        self.embedding_manager = embedding_manager
        self.faiss_indexer = faiss_indexer
        
    def evaluate_model(self, model_id, queries, documents, RAM_DB, k_values=[1, 3, 5, 10]):
        """
        Evalúa un modelo específico usando MRR y Recall@K
        """
        print(f"\n🔍 Evaluando modelo: {model_id} ({self.modelos[model_id]})")
        
        # Verificar que el modelo existe en RAM_DB
        if model_id not in RAM_DB or not RAM_DB[model_id]:
            print(f"❌ No hay embeddings para el modelo {model_id}")
            return None
        
        # Cargar o crear índice FAISS
        index, keys = self.faiss_indexer.load_index(model_id)
        if index is None:
            print(f"⚙️ Creando índice FAISS para {model_id}...")
            index, keys = self.faiss_indexer.build_index(RAM_DB[model_id], model_id)
        
        # Obtener tokenizer y modelo para embeddings de queries
        tokenizer, model = self.embedding_manager.get_embedding_model(self.modelos[model_id])
        
        recall_scores = {f"Recall@{k}": [] for k in k_values}
        mrr_scores = []
        
        print(f"📊 Evaluando {len(queries)} queries...")
        
        for i, (query, relevant_doc_text) in enumerate(tqdm(zip(queries, documents), total=len(queries))):
            try:
                # Generar embedding de la query
                query_emb = self.embedding_manager.embed_texts([query], tokenizer, model)[0].astype("float32")
                query_emb = query_emb / np.linalg.norm(query_emb)
                query_emb = query_emb.reshape(1, -1)
                
                # Buscar los top-K documentos
                max_k = max(k_values)
                if isinstance(index, faiss.IndexFlatIP):
                    faiss.normalize_L2(query_emb)
                
                distances, idxs = index.search(query_emb, k=max_k)
                
                # Obtener los documentos recuperados
                retrieved_docs = []
                for idx in idxs[0]:
                    if idx < len(keys):
                        doc_key = keys[idx]
                        retrieved_doc = RAM_DB[model_id][doc_key]
                        retrieved_docs.append(retrieved_doc["texto"])
                
                # Calcular Recall@K para cada valor de k
                for k in k_values:
                    top_k_docs = retrieved_docs[:k]
                    # Considerar relevante si el documento correcto está en los top-K
                    is_relevant = any(self._is_similar(relevant_doc_text, doc) for doc in top_k_docs)
                    recall_scores[f"Recall@{k}"].append(1.0 if is_relevant else 0.0)
                
                # Calcular MRR
                for rank, doc_text in enumerate(retrieved_docs, 1):
                    if self._is_similar(relevant_doc_text, doc_text):
                        mrr_scores.append(1.0 / rank)
                        break
                else:
                    mrr_scores.append(0.0)
                    
            except Exception as e:
                print(f"❌ Error evaluando query {i}: {str(e)}")
                # En caso de error, agregar scores de 0
                for k in k_values:
                    recall_scores[f"Recall@{k}"].append(0.0)
                mrr_scores.append(0.0)
        
        # Calcular métricas finales
        results = {
            "model": model_id,
            "model_name": self.modelos[model_id]
        }
        
        for k in k_values:
            recall_key = f"Recall@{k}"
            results[recall_key] = np.mean(recall_scores[recall_key])
        
        results["MRR"] = np.mean(mrr_scores)
        
        # Mostrar resultados
        print(f"\n📈 RESULTADOS para {model_id}:")
        print(f"MRR: {results['MRR']:.4f}")
        for k in k_values:
            print(f"Recall@{k}: {results[f'Recall@{k}']:.4f}")
        
        return results
    
    def _is_similar(self, doc1, doc2, similarity_threshold=0.8):
        """
        Determina si dos documentos son similares.
        Puedes ajustar el umbral o usar métodos más sofisticados.
        """
        # Método simple: comparar substrings (puedes mejorar esto)
        doc1_clean = doc1.lower().strip()
        doc2_clean = doc2.lower().strip()
        
        # Si los documentos son idénticos o muy similares
        if doc1_clean == doc2_clean:
            return True
        
        # Si uno contiene al otro (para chunks)
        if doc1_clean in doc2_clean or doc2_clean in doc1_clean:
            return True
            
        # Comparación por longitud de overlap (método simple)
        words1 = set(doc1_clean.split())
        words2 = set(doc2_clean.split())
        
        if len(words1) == 0 or len(words2) == 0:
            return False
            
        overlap = len(words1.intersection(words2))
        jaccard_similarity = overlap / len(words1.union(words2))
        
        return jaccard_similarity > similarity_threshold
    
    def evaluate_all_models(self, queries, documents, RAM_DB, k_values=[1, 3, 5, 10]):
        """
        Evalúa todos los modelos y compara resultados
        """
        all_results = {}
        
        for model_id in self.modelos.keys():
            results = self.evaluate_model(model_id, queries, documents, RAM_DB, k_values)
            if results:
                all_results[model_id] = results
        
        # Comparar modelos
        self._compare_models(all_results)
        
        return all_results
    
    def _compare_models(self, all_results):
        """
        Compara y muestra resultados de todos los modelos
        """
        print("\n" + "="*80)
        print("🏆 COMPARACIÓN DE MODELOS")
        print("="*80)
        
        # Crear tabla comparativa
        headers = ["Modelo", "MRR"] + [f"Recall@{k}" for k in [1, 3, 5, 10]]
        print(f"{'Modelo':<20} {'MRR':<8} {'R@1':<8} {'R@3':<8} {'R@5':<8} {'R@10':<8}")
        print("-" * 80)
        
        for model_id, results in all_results.items():
            mrr = results.get("MRR", 0)
            r1 = results.get("Recall@1", 0)
            r3 = results.get("Recall@3", 0)
            r5 = results.get("Recall@5", 0)
            r10 = results.get("Recall@10", 0)
            
            print(f"{model_id:<20} {mrr:<8.4f} {r1:<8.4f} {r3:<8.4f} {r5:<8.4f} {r10:<8.4f}")
        
        # Encontrar mejores modelos por métrica
        best_mrr = max(all_results.items(), key=lambda x: x[1].get("MRR", 0))
        best_r10 = max(all_results.items(), key=lambda x: x[1].get("Recall@10", 0))
        
        print(f"\n🎯 Mejor MRR: {best_mrr[0]} ({best_mrr[1]['MRR']:.4f})")
        print(f"🎯 Mejor Recall@10: {best_r10[0]} ({best_r10[1]['Recall@10']:.4f})")