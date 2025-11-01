# main.py
from model_manager import ModelManager
from search_engine import HybridSearch
from modelos import MODELOS
import pandas as pd

if __name__ == "__main__":
    print("🚀 Sistema de Búsqueda Semántica Híbrido (FAISS + BM25)")

    manager = ModelManager()
    docs, RAM_DB = manager.prepare_models()

    search_engine = HybridSearch(MODELOS, manager.emb_manager, manager.faiss_indexer, docs)

    queries = [
        "Sor Juana Inés de la Cruz",
        "El papel de las mujeres en la historia de México",
        "Batallas durante la Independencia de México",
        "Planes revolucionarios y movimientos campesinos",
        "Ciencia y religión en la época colonial",
        "El fin del gobierno de Madero",
        "Castillo Chapultepec",
    ]

    resultados = []

    for query in queries:
        print(f"\n===============================")
        print(f"🔎 Consulta: {query}")
        print(f"===============================")

        # BM25
        bm25_res = search_engine.bm25.search(query, top_k=5)
        resultados.extend([
            {"query": query, "modelo": "BM25", **r}
            for r in bm25_res
        ])

        for model in MODELOS:
            sem_res = search_engine.search(query, model, RAM_DB, top_k=5)
            resultados.extend([
                {"query": query, "modelo": f"FAISS ({model})", **r}
                for r in sem_res
            ])

            hybrid_res = search_engine.hybrid_search(query, model, RAM_DB, top_k=5, alpha=0.7)
            resultados.extend([
                {"query": query, "modelo": f"Híbrido (BM25+FAISS {model})", **r}
                for r in hybrid_res
            ])

    pd.DataFrame(resultados).to_excel("resultados_busqueda.xlsx", index=False)
    print("\n✅ Resultados guardados en 'resultados_busqueda.xlsx'")
