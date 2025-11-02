# main_evaluation.py
from embedding_manager import EmbeddingManager
from faiss_indexer import FaissIndexer
from evaluation_manager import EvaluationManager
from datasets import load_dataset
from modelos import MODELOS
import numpy as np

def prepare_dataset_for_evaluation(dataset, sample_size=None):
    """
    Prepara el dataset para evaluación
    """
    queries = dataset["query"]
    documents = dataset["docid_text"]
    
    if sample_size and sample_size < len(queries):
        indices = np.random.choice(len(queries), sample_size, replace=False)
        queries = [queries[i] for i in indices]
        documents = [documents[i] for i in indices]
    
    return queries, documents

def build_embeddings_for_evaluation(documents, embedding_manager, chunk_fn):
    """
    Construye embeddings para todos los documentos del dataset
    """
    RAM_DB = {}
    
    # Crear documentos en formato compatible con tu sistema
    docs_formatted = []
    for i, text in enumerate(documents):
        docs_formatted.append({
            "archivo": f"dataset_doc_{i}",
            "etapa": "evaluation",
            "id_relato": f"doc_{i}",
            "titulo": f"Documento {i}",
            "tipo": "real",
            "texto": text
        })
    
    # Generar embeddings para cada modelo
    for model_id, model_name in MODELOS.items():
        print(f"⚙️ Generando embeddings para {model_id}...")
        try:
            RAM_DB[model_id] = embedding_manager.build_embeddings_for_model(
                model_id, model_name, docs_formatted, chunk_fn
            )
        except Exception as e:
            print(f"❌ Error con modelo {model_id}: {str(e)}")
            continue
    
    return RAM_DB

def main():
    # Cargar dataset
    print("📥 Cargando dataset spanish-ir/messirve...")
    dataset = load_dataset("spanish-ir/messirve", "mx")
    
    # Usar el split de prueba si existe, sino usar todo
    if 'test' in dataset:
        eval_dataset = dataset['test']
    else:
        eval_dataset = dataset['train']
    
    # Preparar datos (muestrear para hacerlo más rápido)
    queries, documents = prepare_dataset_for_evaluation(eval_dataset, sample_size=100)
    
    print(f"📊 Dataset preparado: {len(queries)} queries, {len(documents)} documentos")
    
    # Inicializar managers
    embedding_manager = EmbeddingManager(MODELOS)
    faiss_indexer = FaissIndexer()
    evaluation_manager = EvaluationManager(MODELOS, embedding_manager, faiss_indexer)
    
    # Función de chunking (usando la que ya tienes)
    def chunk_text(text, max_chars=800, min_chars=200):
        import re
        text = re.sub(r'\s+', ' ', text).strip()
        sentences = re.split(r'(?<=[.!?¿¡])\s+', text)
        
        chunks, current = [], ""
        for sent in sentences:
            if not sent.strip():
                continue
            if len(current) + len(sent) + 1 > max_chars and len(current) >= min_chars:
                chunks.append(current.strip())
                current = sent
            else:
                current += " " + sent
        
        if current.strip():
            chunks.append(current.strip())
        return chunks
    
    # Construir embeddings
    print("🔄 Construyendo embeddings...")
    RAM_DB = build_embeddings_for_evaluation(documents, embedding_manager, chunk_text)
    
    # Evaluar modelos
    print("🔍 Iniciando evaluación...")
    results = evaluation_manager.evaluate_all_models(
        queries, documents, RAM_DB, k_values=[1, 3, 5, 10]
    )
    
    # Guardar resultados
    with open("evaluation_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print("💾 Resultados guardados en evaluation_results.json")

if __name__ == "__main__":
    main()