import os
from gensim.models import Word2Vec
from typing import List, Generator
import logging

# Configurar logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

class Word2VecSubword:
    def __init__(self, tokenizer, vector_size=128, window=5, min_count=1, sg=1, workers=4):
        """
        Inicializa el modelo Word2Vec para subwords
        
        Args:
            tokenizer: Tokenizer BPE entrenado
            vector_size: Dimensión de los embeddings
            window: Tamaño de la ventana de contexto
            min_count: Frecuencia mínima de palabras
            sg: 1 para Skip-gram, 0 para CBOW
            workers: Número de workers para entrenamiento
        """
        self.tokenizer = tokenizer
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.sg = sg
        self.workers = workers
        self.model = None
        self.is_trained = False
    
    def corpus_bpe_iterator(self, corpus_files: List[str]) -> Generator[List[str], None, None]:
        """
        Generador que tokeniza los archivos del corpus usando BPE
        
        Args:
            corpus_files: Lista de rutas a archivos de texto
            
        Yields:
            Lista de tokens para cada documento
        """
        for file_path in corpus_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    
                if content:
                    # Tokenizar el contenido
                    encoded = self.tokenizer.encode(content)
                    yield encoded.tokens
                    
            except Exception as e:
                print(f"❌ Error procesando {file_path}: {e}")
                continue
    
    def prepare_dataset(self, corpus_files: List[str]) -> List[List[str]]:
        """
        Prepara el dataset tokenizado para entrenamiento
        
        Args:
            corpus_files: Lista de rutas a archivos de texto
            
        Returns:
            Lista de documentos tokenizados
        """
        print("🔄 Preparando dataset tokenizado...")
        dataset = list(self.corpus_bpe_iterator(corpus_files))
        print(f"✅ Dataset preparado: {len(dataset)} documentos")
        return dataset
    
    def train(self, corpus_files: List[str], epochs=10, **kwargs):
        """
        Entrena el modelo Word2Vec con subwords BPE
        
        Args:
            corpus_files: Lista de rutas a archivos de texto
            epochs: Número de épocas de entrenamiento
            **kwargs: Parámetros adicionales para Word2Vec
        """
        # Preparar dataset
        dataset = self.prepare_dataset(corpus_files)
        
        if not dataset:
            raise ValueError("No se pudo preparar el dataset. Verifica los archivos de entrada.")
        
        # Contar tokens totales
        total_tokens = sum(len(doc) for doc in dataset)
        print(f"📊 Total de tokens en el corpus: {total_tokens}")
        
        # Configurar parámetros del modelo
        model_params = {
            'vector_size': self.vector_size,
            'window': self.window,
            'min_count': self.min_count,
            'sg': self.sg,
            'workers': self.workers,
            'epochs': epochs,
            **kwargs
        }
        
        print(f"🔧 Parámetros del modelo: {model_params}")
        
        # Entrenar modelo
        print("🚀 Entrenando modelo Word2Vec...")
        self.model = Word2Vec(
            sentences=dataset,
            **model_params
        )
        
        self.is_trained = True
        print("✅ Modelo Word2Vec entrenado exitosamente")
    
    def get_embedding(self, word: str):
        """
        Obtiene el embedding de una palabra/subword
        
        Args:
            word: Palabra o subword
            
        Returns:
            Vector de embedding o None si no existe
        """
        if not self.is_trained:
            raise ValueError("El modelo no ha sido entrenado")
        
        try:
            return self.model.wv[word]
        except KeyError:
            return None
    
    def most_similar(self, word: str, topn=10):
        """
        Encuentra las palabras más similares
        
        Args:
            word: Palabra de consulta
            topn: Número de resultados a devolver
            
        Returns:
            Lista de tuplas (palabra, similitud)
        """
        if not self.is_trained:
            raise ValueError("El modelo no ha sido entrenado")
        
        try:
            return self.model.wv.most_similar(word, topn=topn)
        except KeyError:
            return f"La palabra '{word}' no está en el vocabulario"
    
    def similarity(self, word1: str, word2: str):
        """
        Calcula la similitud entre dos palabras
        
        Args:
            word1: Primera palabra
            word2: Segunda palabra
            
        Returns:
            Score de similitud o mensaje de error
        """
        if not self.is_trained:
            raise ValueError("El modelo no ha sido entrenado")
        
        try:
            return self.model.wv.similarity(word1, word2)
        except KeyError as e:
            return f"Error: {e}"
    
    def get_vocabulary_size(self):
        """
        Obtiene el tamaño del vocabulario aprendido
        
        Returns:
            Número de palabras en el vocabulario
        """
        if not self.is_trained:
            raise ValueError("El modelo no ha sido entrenado")
        
        return len(self.model.wv.key_to_index)
    
    def save_model(self, filename="subword_word2vec.model"):
        """
        Guarda el modelo entrenado
        
        Args:
            filename: Nombre del archivo para guardar
        """
        if not self.is_trained:
            raise ValueError("El modelo no ha sido entrenado")
        
        self.model.save(filename)
        print(f"💾 Modelo Word2Vec guardado como: {filename}")
    
    def load_model(self, filename="subword_word2vec.model"):
        """
        Carga un modelo previamente entrenado
        
        Args:
            filename: Nombre del archivo a cargar
        """
        self.model = Word2Vec.load(filename)
        self.is_trained = True
        print(f"📂 Modelo Word2Vec cargado desde: {filename}")
    
    def print_model_info(self):
        """
        Imprime información del modelo entrenado
        """
        if not self.is_trained:
            print("❌ Modelo no entrenado")
            return
        
        print("\n📈 INFORMACIÓN DEL MODELO WORD2VEC:")
        print(f"   Tamaño del vocabulario: {self.get_vocabulary_size()}")
        print(f"   Dimensión de embeddings: {self.vector_size}")
        print(f"   Ventana de contexto: {self.window}")
        print(f"   Arquitectura: {'Skip-gram' if self.sg else 'CBOW'}")
        
        # Mostrar algunas palabras del vocabulario
        vocab_words = list(self.model.wv.key_to_index.keys())[:10]
        print(f"   Ejemplo de palabras: {vocab_words}")