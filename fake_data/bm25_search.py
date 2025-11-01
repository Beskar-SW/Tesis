from rank_bm25 import BM25Okapi
import re

class BM25Search:
    def __init__(self, documents):
        """
        documents: lista de diccionarios con campos ['texto', 'titulo', 'etapa', etc.]
        """
        self.docs = documents
        self.tokenized_corpus = [self.tokenize(doc["texto"]) for doc in self.docs]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

    def tokenize(self, text):
        text = re.sub(r"[^a-záéíóúüñ\s]", " ", text.lower())
        return text.split()

    def search(self, query, top_k=10):
        """
        Retorna los top_k documentos más relevantes según BM25
        """
        tokenized_query = self.tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)
        ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

        results = []
        for rank, idx in enumerate(ranked_indices, 1):
            doc = self.docs[idx]
            results.append({
                "rank": rank,
                "score": float(scores[idx]),
                **doc
            })
        return results
