import numpy as np
from typing import List, Optional
from bpe import BPE
from w2v import Word2VecSubword

class WordEmbedder:
    def __init__(self, tokenizer: BPE, word2vec_model: Word2VecSubword):
        """
        Inicializa el embedder de palabras completas
        
        Args:
            tokenizer: Tokenizer BPE entrenado
            word2vec_model: Modelo Word2Vec entrenado con subwords
        """
        self.tokenizer = tokenizer
        self.model = word2vec_model
    
    # def embed_word(self, word: str, strategy: str = "mean") -> Optional[np.ndarray]:
    #     """
    #     Calcula el embedding de una palabra completa promediando sus subwords
        
    #     Args:
    #         word: Palabra a embeddar
    #         strategy: Estrategia para combinar subwords ("mean", "sum", "first")
            
    #     Returns:
    #         Vector de embedding o None si no se puede calcular
    #     """
    #     try:
    #         # Tokenizar la palabra en subwords
    #         encoded = self.tokenizer.encode(word)
    #         tokens = encoded.tokens
            
    #         if not tokens:
    #             return None
            
    #         # Obtener vectores de cada subword
    #         vectors = []
    #         for token in tokens:
    #             if token in self.model.model.wv:
    #                 vectors.append(self.model.model.wv[token])
    #             else:
    #                 print(f"⚠️  Subword '{token}' no encontrado en el vocabulario")
            
    #         if not vectors:
    #             return None
            
    #         # Combinar los vectores según la estrategia
    #         if strategy == "mean":
    #             return np.mean(vectors, axis=0)
    #         elif strategy == "sum":
    #             return np.sum(vectors, axis=0)
    #         elif strategy == "first":
    #             return vectors[0]
    #         elif strategy == "last":
    #             return vectors[-1]
    #         else:
    #             raise ValueError(f"Estrategia no válida: {strategy}")
                
    #     except Exception as e:
    #         print(f"❌ Error embeddando palabra '{word}': {e}")
    #         return None

    def embed_word(self, word: str, strategy: str = "mean") -> Optional[np.ndarray]:
        """
        Calcula el embedding de una palabra completa promediando sus subwords
        
        Args:
            word: Palabra a embeddar
            strategy: Estrategia para combinar subwords ("mean", "sum", "first", "last")
            
        Returns:
            Vector de embedding o None si no se puede calcular
        """
        try:
            # Tokenizar la palabra en subwords
            encoded = self.tokenizer.encode(word)
            tokens = encoded.tokens

            if not tokens:
                return None

            vectors = []
            seen_tokens = set()  # evitar duplicados

            for token in tokens:
                # Normalizar token: quitar prefijos Ġ y pasar a minúsculas
                token_clean = token.replace("Ġ", "").strip().lower()

                # Evitar repetir el mismo token
                if token_clean in seen_tokens:
                    continue
                seen_tokens.add(token_clean)

                # Buscar en el vocabulario tanto con prefijo como sin prefijo
                if token in self.model.model.wv:
                    vectors.append(self.model.model.wv[token])
                elif token_clean in self.model.model.wv:
                    vectors.append(self.model.model.wv[token_clean])
                elif "<unk>" in self.model.model.wv:
                    # Asignar vector <unk> para tokens desconocidos
                    vectors.append(self.model.model.wv["<unk>"])
                else:
                    print(f"⚠️  Subword '{token}' no encontrado en el vocabulario")

            if not vectors:
                return None

            # Combinar los vectores según la estrategia
            if strategy == "mean":
                return np.mean(vectors, axis=0)
            elif strategy == "sum":
                return np.sum(vectors, axis=0)
            elif strategy == "first":
                return vectors[0]
            elif strategy == "last":
                return vectors[-1]
            else:
                raise ValueError(f"Estrategia no válida: {strategy}")

        except Exception as e:
            print(f"❌ Error embeddando palabra '{word}': {e}")
            return None

    
    def embed_sentence(self, sentence: str, strategy: str = "mean") -> Optional[np.ndarray]:
        """
        Calcula el embedding de una oración completa
        
        Args:
            sentence: Oración a embeddar
            strategy: Estrategia para combinar subwords
            
        Returns:
            Vector de embedding de la oración
        """
        try:
            # Tokenizar la oración
            encoded = self.tokenizer.encode(sentence)
            tokens = encoded.tokens
            
            if not tokens:
                return None
            
            # Obtener vectores de cada token
            vectors = []
            for token in tokens:
                if token in self.model.model.wv:
                    vectors.append(self.model.model.wv[token])
            
            if not vectors:
                return None
            
            return np.mean(vectors, axis=0)
            
        except Exception as e:
            print(f"❌ Error embeddando oración: {e}")
            return None
    
    def batch_embed_words(self, words: List[str], strategy: str = "mean") -> List[Optional[np.ndarray]]:
        """
        Calcula embeddings para un lote de palabras
        
        Args:
            words: Lista de palabras
            strategy: Estrategia para combinar subwords
            
        Returns:
            Lista de vectores de embedding
        """
        return [self.embed_word(word, strategy) for word in words]
    
    def get_word_similarity(self, word1: str, word2: str, strategy: str = "mean") -> Optional[float]:
        """
        Calcula la similitud coseno entre dos palabras
        
        Args:
            word1: Primera palabra
            word2: Segunda palabra
            strategy: Estrategia para combinar subwords
            
        Returns:
            Score de similitud o None
        """
        vec1 = self.embed_word(word1, strategy)
        vec2 = self.embed_word(word2, strategy)
        
        if vec1 is None or vec2 is None:
            return None
        
        # Similitud coseno
        cosine_sim = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
        return cosine_sim
    
    def find_most_similar_words(self, word: str, topn: int = 10, strategy: str = "mean"):
        """
        Encuentra palabras similares basado en el embedding de la palabra completa
        
        Args:
            word: Palabra de consulta
            topn: Número de resultados
            strategy: Estrategia para combinar subwords
            
        Returns:
            Lista de palabras similares
        """
        target_vec = self.embed_word(word, strategy)
        
        if target_vec is None:
            return f"No se pudo calcular embedding para '{word}'"
        
        # Buscar en el vocabulario completo
        similarities = []
        for vocab_word in self.model.model.wv.key_to_index.keys():
            vocab_vec = self.model.model.wv[vocab_word]
            similarity = np.dot(target_vec, vocab_vec) / (
                np.linalg.norm(target_vec) * np.linalg.norm(vocab_vec)
            )
            similarities.append((vocab_word, similarity))
        
        # Ordenar por similitud
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:topn]

def main():
    """
    Ejemplo de uso de la clase WordEmbedder
    """
    # Cargar modelos
    tokenizer = BPE()
    tokenizer.load("bpe_tokenizer.json")
    
    word2vec_model = Word2VecSubword(tokenizer)
    word2vec_model.load_model("subword_word2vec.model")
    
    # Crear embedder
    embedder = WordEmbedder(tokenizer, word2vec_model)
    
    # Probar con diferentes palabras
    test_words = [
        "constitucion",
        "biblioteca", 
        "palafoxiana",
        "mexico",
        "historia",
        "obispo",
        "ciudad",
        "documento"
    ]
    
    print("🧪 PROBANDO EMBEDDINGS DE PALABRAS COMPLETAS")
    print("=" * 50)
    
    for word in test_words:
        vec = embedder.embed_word(word)
        if vec is not None:
            print(f"✅ '{word}': shape {vec.shape}, norma: {np.linalg.norm(vec):.4f}")
        else:
            print(f"❌ '{word}': No se pudo calcular embedding")
    
    # Probar similitudes
    print("\n🔍 SIMILITUDES ENTRE PALABRAS")
    print("=" * 50)
    
    pairs = [
        ("biblioteca", "palafoxiana"),
        ("mexico", "ciudad"),
        ("historia", "documento"),
        ("biblioteca", "mexico"),
        ("palafox", "pallafox")
    ]
    
    for word1, word2 in pairs:
        similarity = embedder.get_word_similarity(word1, word2, strategy="first")
        if similarity is not None:
            print(f"'{word1}' vs '{word2}': {similarity:.4f}")
        else:
            print(f"'{word1}' vs '{word2}': No se pudo calcular")

if __name__ == "__main__":
    main()

    tokenizer = BPE()
    tokenizer.load("bpe_tokenizer.json")

    model = Word2VecSubword(tokenizer)
    model.load_model("subword_word2vec.model")

    print(tokenizer.encode("palafoxiana").tokens)
    print(tokenizer.encode("mexico").tokens)
    print(tokenizer.encode("biblioteca palafoxiana").tokens)
    print(tokenizer.encode("pallafox").tokens)
    print(tokenizer.encode("palafox").tokens)


    encoded = tokenizer.encode("pallafox")
    tokens = encoded.tokens

    vectors = []
    for token in tokens:
        if token in model.model.wv:
            vectors.append(model.model.wv[token])

    print(np.average(vectors, axis=0))