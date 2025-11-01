import os
import unicodedata
from bpe import BPE

class DataProcessor:
    def __init__(self, input_dir="corpus_cleaned"):
        self.input_dir = input_dir
    
    def normalize_text(self, text):
        """
        Normaliza el texto: minúsculas y sin acentos
        """
        # Convertir a minúsculas
        text = text.lower()
        
        # Eliminar acentos
        text = unicodedata.normalize('NFD', text)
        text = ''.join(c for c in text if unicodedata.category(c) != 'Mn')
        
        return text
    
    def extract_content_complete(self, file_path):
        """
        Extrae específicamente la sección después de 'CONTENIDO COMPLETO:'
        y normaliza el texto
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            in_content_section = False
            content_lines = []
            
            for line in lines:
                line = line.strip()
                
                # Buscar el inicio de CONTENIDO COMPLETO
                if "CONTENIDO COMPLETO:" in line:
                    in_content_section = True
                    continue
                
                # Si estamos en la sección de contenido
                if in_content_section:
                    # Detener cuando encontramos secciones de referencia o bibliografía
                    if line.startswith("== Referencias ==") or line.startswith("== Bibliografía =="):
                        break
                    
                    # Saltar líneas de separación con ====
                    if line.startswith("==") or "====" in line:
                        continue
                    
                    # Agregar líneas de contenido real (normalizadas)
                    if line and not line.startswith('='):
                        normalized_line = self.normalize_text(line)
                        content_lines.append(normalized_line)
            
            return ' '.join(content_lines)
            
        except Exception as e:
            print(f"❌ Error procesando {file_path}: {e}")
            return ""

    def prepare_training_data(self, output_dir="training_data"):
        """
        Prepara los datos de entrenamiento creando archivos temporales
        solo con el contenido que nos interesa (normalizado)
        """
        os.makedirs(output_dir, exist_ok=True)
        
        files = [f for f in os.listdir(self.input_dir) if f.endswith('.txt')]
        training_files = []
        
        print(f"🔍 Procesando {len(files)} archivos de {self.input_dir}...")
        
        for file_name in files:
            input_path = os.path.join(self.input_dir, file_name)
            output_path = os.path.join(output_dir, f"content_{file_name}")
            
            content = self.extract_content_complete(input_path)
            
            if content and len(content.split()) > 10:  # Solo usar contenido significativo
                # Asegurarnos de que el contenido es un string válido
                if isinstance(content, str) and content.strip():
                    with open(output_path, 'w', encoding='utf-8') as f:
                        f.write(content)
                    training_files.append(output_path)
                    print(f"✅ {file_name} -> contenido extraído ({len(content.split())} palabras)")
                else:
                    print(f"⚠️  {file_name} -> contenido no es string válido")
            else:
                print(f"⚠️  {file_name} -> contenido insuficiente")
        
        return training_files

    def get_training_statistics(self, training_files):
        """
        Obtiene estadísticas de los datos de entrenamiento
        """
        total_words = 0
        total_chars = 0
        
        for file_path in training_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if isinstance(content, str):
                        total_words += len(content.split())
                        total_chars += len(content)
            except Exception as e:
                print(f"❌ Error leyendo {file_path}: {e}")
        
        return {
            'num_files': len(training_files),
            'total_words': total_words,
            'total_chars': total_chars,
            'avg_words_per_file': total_words / len(training_files) if training_files else 0
        }

def main():
    # Configuración
    VOCAB_SIZE = 3000 #5000 mejor resultados #3000 estan bien
    MIN_FREQUENCY = 2
    
    print("🚀 Iniciando procesamiento de datos para BPE...")
    
    # 1. Procesar datos
    processor = DataProcessor("corpus_cleaned")
    training_files = processor.prepare_training_data("training_data")
    
    if not training_files:
        print("❌ No se encontraron datos válidos para entrenar")
        return
    
    # 2. Mostrar estadísticas
    stats = processor.get_training_statistics(training_files)
    print("\n📊 ESTADÍSTICAS DE ENTRENAMIENTO:")
    print(f"   Archivos procesados: {stats['num_files']}")
    print(f"   Total de palabras: {stats['total_words']}")
    print(f"   Total de caracteres: {stats['total_chars']}")
    print(f"   Promedio de palabras por archivo: {stats['avg_words_per_file']:.1f}")
    
    # 3. Entrenar tokenizer BPE
    print("\n🔧 Entrenando tokenizer BPE...")
    tokenizer = BPE(
        vocab_size=VOCAB_SIZE,
        min_frequency=MIN_FREQUENCY,
        special_tokens=["<pad>", "<unk>"]
    )
    
    tokenizer.train(training_files)
    
    # 4. Guardar tokenizer
    tokenizer.save("bpe_tokenizer.json")
    
    # 5. Mostrar información del tokenizer entrenado
    print(f"\n🎯 TOKENIZER ENTRENADO:")
    print(f"   Tamaño del vocabulario: {tokenizer.get_vocab_size()}")
    
    # 6. Probar el tokenizer con un ejemplo
    print("\n🧪 Probando tokenizer...")
    if training_files:
        try:
            with open(training_files[0], 'r', encoding='utf-8') as f:
                sample_text = f.read()
            
            # Asegurarnos de que sample_text es un string
            if isinstance(sample_text, str) and sample_text.strip():
                # Tomar solo una porción para la prueba
                test_text = sample_text[:200].strip()
                
                print(f"   Texto de prueba: '{test_text}'")
                
                # Codificar
                encoded = tokenizer.encode(test_text)
                print(f"   Tokens generados: {len(encoded.tokens)} tokens")
                print(f"   Primeros 10 IDs: {encoded.ids[:10]}")
                print(f"   Primeros 10 tokens: {encoded.tokens[:10]}")
                
                # Decodificar de vuelta
                decoded = tokenizer.decode(encoded.ids)
                print(f"   Texto decodificado: '{decoded}'")
                
                # Verificar que la decodificación funciona
                if test_text.lower() in decoded.lower() or decoded.lower() in test_text.lower():
                    print("   ✅ Codificación/decodificación exitosa")
                else:
                    print("   ⚠️  Diferencias en codificación/decodificación")
            else:
                print("   ❌ Texto de prueba no válido")
                
        except Exception as e:
            print(f"   ❌ Error en prueba: {e}")

if __name__ == "__main__":
    main()