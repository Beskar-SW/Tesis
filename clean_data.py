import spacy
import os
import re
from pathlib import Path

# Cargar el modelo de spaCy para español
nlp = spacy.load("es_core_news_sm")

def clean_text_preserving_structure(text):
    """
    Limpia el texto eliminando palabras basadas en POS tagging pero preservando:
    - Encabezados con == ==
    - Saltos de línea
    - Estructura del documento
    """
    
    lines = text.split('\n')
    cleaned_lines = []
    
    pos_to_remove = {
        'ADP',  # Preposiciones (de, en, con, por, para, etc.)
        'DET',  # Determinantes (el, la, los, las, un, una, etc.)
        'CCONJ', # Conjunciones coordinantes (y, o, pero, etc.)
        'SCONJ', # Conjunciones subordinantes (que, porque, aunque, etc.)
        'AUX',   # Verbos auxiliares (ser, estar, haber)
        'PART',  # Partículas (no)
        'PRON',  # Pronombres (él, ella, nosotros, etc.)
        'INTJ',  # Interjecciones (ay, oh, eh, etc.)
        'SYM',   # Símbolos
        'PUNCT', # Puntuación
        'X'      # Otros
    }
    
    for line in lines:
        # Preservar líneas vacías y encabezados con ==
        if not line.strip() or line.strip().startswith('==') or line.strip() == 'CONTENIDO COMPLETO:' or '=' in line and line.count('=') >= 4:
            cleaned_lines.append(line)
            continue
        
        # Preservar metadatos (Título:, URL:, Resumen:)
        if any(prefix in line for prefix in ['Título:', 'URL:', 'Resumen:']):
            cleaned_lines.append(line)
            continue
        
        # Procesar línea de texto normal
        doc = nlp(line)
        filtered_tokens = []
        
        for token in doc:
            # Mantener tokens que NO están en las categorías a eliminar
            if (token.pos_ not in pos_to_remove and 
                not token.is_space and 
                len(token.text.strip()) > 0 and
                not token.like_num and
                len(token.text) > 1):
                
                # Mantener la capitalización original
                clean_token = re.sub(r'[^\w]', '', token.text)
                if clean_token:
                    filtered_tokens.append(clean_token)
        
        # Reconstruir la línea con las palabras filtradas
        if filtered_tokens:
            cleaned_line = ' '.join(filtered_tokens)
            cleaned_lines.append(cleaned_line)
        else:
            # Si la línea queda vacía después del filtrado, mantenerla vacía
            cleaned_lines.append('')
    
    return '\n'.join(cleaned_lines)

def process_corpus_files():
    """
    Procesa todos los archivos del corpus original y crea versiones limpias
    preservando la estructura
    """
    # Crear carpeta para los documentos limpios
    os.makedirs("corpus_cleaned", exist_ok=True)
    
    # Obtener lista de archivos en el corpus original
    corpus_path = Path("corpus")
    files = list(corpus_path.glob("*.txt"))
    
    if not files:
        print("❌ No se encontraron archivos en la carpeta 'corpus'")
        return
    
    print(f"📁 Procesando {len(files)} archivos...")
    
    for file_path in files:
        try:
            # Leer el archivo original
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Limpiar el contenido preservando estructura
            cleaned_content = clean_text_preserving_structure(content)
            
            # Crear archivo limpio
            cleaned_filename = f"cleaned_{file_path.name}"
            cleaned_filepath = Path("corpus_cleaned") / cleaned_filename
            
            # Guardar el contenido limpio
            with open(cleaned_filepath, 'w', encoding='utf-8') as f:
                f.write(cleaned_content)
            
            # Mostrar estadísticas
            original_words = len(re.findall(r'\b\w+\b', content))
            cleaned_words = len(re.findall(r'\b\w+\b', cleaned_content))
            
            print(f"✅ Procesado: {file_path.name}")
            print(f"   📊 Palabras: {original_words} → {cleaned_words} "
                  f"({((original_words - cleaned_words) / original_words * 100):.1f}% reducción)")
            print()
            
        except Exception as e:
            print(f"❌ Error procesando {file_path.name}: {e}")

def show_sample_comparison():
    """
    Muestra una comparación de muestra entre original y limpiado
    """
    corpus_path = Path("corpus")
    files = list(corpus_path.glob("*.txt"))
    
    if files:
        sample_file = files[0]
        with open(sample_file, 'r', encoding='utf-8') as f:
            original_content = f.read()
        
        cleaned_content = clean_text_preserving_structure(original_content)
        
        print("🔍 MUESTRA DE COMPARACIÓN:")
        print("=" * 50)
        print("ORIGINAL (primeros 300 caracteres):")
        print(original_content[:300])
        print("\n" + "=" * 50)
        print("LIMPIO (primeros 300 caracteres):")
        print(cleaned_content[:300])
        print("=" * 50)

# Ejecutar el procesamiento
if __name__ == "__main__":
    print("🔧 Iniciando limpieza de corpus preservando estructura...")
    
    # Mostrar muestra de comparación
    show_sample_comparison()
    
    # Procesar todos los archivos
    process_corpus_files()
    
    print("\n🎯 Proceso completado!")
    print("📁 Archivos originales: carpeta 'corpus'")
    print("🧹 Archivos limpios: carpeta 'corpus_cleaned'")