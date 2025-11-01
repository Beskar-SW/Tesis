import os
from bpe import BPE
from w2v import Word2VecSubword

def main():
    # Configuración
    CORPUS_DIR = "training_data"  # Carpeta con los archivos preparados
    BPE_MODEL_PATH = "bpe_tokenizer.json"
    
    print("🚀 Iniciando entrenamiento de Word2Vec con subwords...")
    
    # 1. Cargar tokenizer BPE
    print("📂 Cargando tokenizer BPE...")
    tokenizer = BPE()
    tokenizer.load(BPE_MODEL_PATH)
    print(f"✅ Tokenizer cargado - Vocabulario: {tokenizer.get_vocab_size()} tokens")
    
    # 2. Obtener archivos de entrenamiento
    corpus_files = [os.path.join(CORPUS_DIR, f) for f in os.listdir(CORPUS_DIR) 
                   if f.endswith('.txt')]
    
    if not corpus_files:
        print("❌ No se encontraron archivos de entrenamiento")
        return
    
    print(f"📁 Archivos de entrenamiento: {len(corpus_files)}")
    
    # 3. Configurar y entrenar Word2Vec
    word2vec_model = Word2VecSubword(
        tokenizer=tokenizer,
        vector_size=128,    # Dimensión de embeddings
        window=20,           # Contexto de n palabras
        min_count=1,        # Incluir todas las subwords
        sg=1,               # Skip-gram (mejor para datos pequeños)
        workers=4           # Paralelización
    )
    
    # 4. Entrenar el modelo
    word2vec_model.train(
        corpus_files=corpus_files,
        epochs=50           # Número de épocas
    )
    
    # 5. Guardar modelo
    word2vec_model.save_model("subword_word2vec.model")
    
    # 6. Mostrar información del modelo
    word2vec_model.print_model_info()
    
    # 7. Probar el modelo con ejemplos
    print("\n🧪 PROBANDO EL MODELO:")
    
    # Ejemplos de prueba
    test_words = ["biblioteca", "palafoxiana", "ciudad", "mexico", "historia"]
    
    for word in test_words:
        # Tokenizar la palabra para obtener sus subwords
        encoded = tokenizer.encode(word)
        subwords = encoded.tokens
        
        print(f"\n🔍 Palabra: '{word}'")
        print(f"   Subwords: {subwords}")
        
        # Buscar similitudes para cada subword
        for subword in subwords:
            similar = word2vec_model.most_similar(subword, topn=5)
            if isinstance(similar, list):
                print(f"   '{subword}': {similar}")
            else:
                print(f"   '{subword}': {similar}")

if __name__ == "__main__":
    main()