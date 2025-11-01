import os
import re
import numpy as np
import pandas as pd
from lxml import etree
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
import torch
import faiss
import matplotlib.pyplot as plt
from modelos import MODELOS

def save_faiss_index(model_id, index, keys, save_dir="faiss_data"):
    """
    Guarda el índice FAISS y las claves asociadas en disco.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    index_path = os.path.join(save_dir, f"{model_id}.index")
    keys_path = os.path.join(save_dir, f"{model_id}_keys.npy")
    
    faiss.write_index(index, index_path)
    np.save(keys_path, np.array(keys))
    
    print(f"💾 Índice FAISS guardado para {model_id}: {index_path}")
    print(f"💾 Claves guardadas: {keys_path}")


def load_faiss_index(model_id, save_dir="faiss_data"):
    """
    Carga el índice FAISS y las claves asociadas desde disco.
    Retorna (index, keys) o None si no existen.
    """
    index_path = os.path.join(save_dir, f"{model_id}.index")
    keys_path = os.path.join(save_dir, f"{model_id}_keys.npy")
    
    if not os.path.exists(index_path) or not os.path.exists(keys_path):
        print(f"⚠️ No hay índice guardado para {model_id}")
        return None, None
    
    index = faiss.read_index(index_path)
    keys = np.load(keys_path, allow_pickle=True).tolist()
    
    print(f"📂 Índice FAISS cargado para {model_id} ({len(keys)} vectores)")
    return index, keys


# =======================
# Función: chunk_text
# =======================
def chunk_text(text, max_chars=800, min_chars=200):
    # Limpieza básica
    text = re.sub(r'\s+', ' ', text).strip()

    # Separar por puntos, signos, etc.
    sentences = re.split(r'(?<=[.!?¿¡])\s+', text)

    chunks, current = [], ""
    for sent in sentences:
        # Evitar agregar oraciones vacías
        if not sent.strip():
            continue

        # Si agregar esta oración pasa el límite -> nuevo chunk
        if len(current) + len(sent) + 1 > max_chars and len(current) >= min_chars:
            chunks.append(current.strip())
            current = sent
        else:
            current += " " + sent

    # Agregar último fragmento
    if current.strip():
        chunks.append(current.strip())

    return chunks


# =======================
# Funciones para embeddings
# =======================
def get_embedding_model(model_name):
    """
    Carga el modelo y el tokenizer para generar embeddings.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    return tokenizer, model


def embed_texts(texts, tokenizer, model, batch_size=8):
    """
    Genera embeddings promedio (pooling) para cada texto de manera eficiente.
    
    Args:
        texts: Lista de textos a procesar
        tokenizer: Tokenizer del modelo
        model: Modelo de embeddings
        batch_size: Tamaño del lote para procesamiento
    """
    embeddings = []
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        
        # Tokenizar con padding y truncamiento
        inputs = tokenizer(
            batch_texts, 
            return_tensors="pt", 
            truncation=True, 
            padding=True, 
            max_length=512
        )
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Pooling promedio de las representaciones ocultas
        batch_embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
        
        # Manejar caso de batch size 1
        if len(batch_embeddings.shape) == 1:
            batch_embeddings = batch_embeddings.reshape(1, -1)
            
        embeddings.extend(batch_embeddings)
    
    return np.array(embeddings)


# =======================
# Cargar y analizar datos XML
# =======================
def load_documents(data_folder="data"):
    """
    Lee los XML de la carpeta data y retorna los textos con metadatos completos.
    """
    docs = []
    
    if not os.path.exists(data_folder):
        print(f"❌ Error: La carpeta '{data_folder}' no existe")
        return docs
        
    files = [f for f in os.listdir(data_folder) if f.endswith('.xml')]
    
    if not files:
        print(f"❌ No se encontraron archivos XML en '{data_folder}'")
        return docs
        
    print(f"📁 Encontrados {len(files)} archivos XML")

    for file in files:
        try:
            path = os.path.join(data_folder, file)
            tree = etree.parse(path)
            root = tree.getroot()
            etapa = root.attrib.get("etapa", "desconocida")

            for relato in root.xpath("//relato"):
                id_relato = relato.get("id", "sin_id")
                titulo = relato.findtext("titulo", "").strip()
                texto_falso = relato.findtext("texto_falso", "").strip()
                datos_reales = relato.findtext("datos_reales", "").strip()
                
                # Validar que los textos no estén vacíos
                if texto_falso:
                    docs.append({
                        "archivo": file,
                        "etapa": etapa,
                        "id_relato": id_relato,
                        "titulo": titulo,
                        "tipo": "falso",
                        "texto": texto_falso
                    })
                
                if datos_reales:
                    docs.append({
                        "archivo": file,
                        "etapa": etapa,
                        "id_relato": id_relato,
                        "titulo": titulo,
                        "tipo": "real",
                        "texto": datos_reales
                    })
                    
        except Exception as e:
            print(f"❌ Error procesando archivo {file}: {str(e)}")
            continue
            
    return docs


def analyze_dataset(docs):
    """
    Analiza y muestra estadísticas del dataset.
    """
    if not docs:
        print("❌ No hay documentos para analizar")
        return
    
    df = pd.DataFrame(docs)
    
    print("\n" + "="*50)
    print("📊 ANÁLISIS DEL DATASET")
    print("="*50)
    
    # Estadísticas básicas
    print(f"📄 Total de documentos: {len(docs)}")
    print(f"📂 Archivos XML procesados: {df['archivo'].nunique()}")
    print(f"🏷️ Etapas únicas: {df['etapa'].unique().tolist()}")
    
    # Distribución por tipo
    tipo_counts = df['tipo'].value_counts()
    print(f"\n📈 Distribución por tipo:")
    for tipo, count in tipo_counts.items():
        print(f"   - {tipo}: {count} documentos")
    
    # Distribución por etapa
    etapa_counts = df['etapa'].value_counts()
    print(f"\n📊 Distribución por etapa:")
    for etapa, count in etapa_counts.items():
        print(f"   - {etapa}: {count} documentos")
    
    # Estadísticas de longitud de texto
    df['longitud_texto'] = df['texto'].str.len()
    print(f"\n📏 Estadísticas de longitud de texto:")
    print(f"   - Longitud mínima: {df['longitud_texto'].min()} caracteres")
    print(f"   - Longitud máxima: {df['longitud_texto'].max()} caracteres")
    print(f"   - Longitud promedio: {df['longitud_texto'].mean():.1f} caracteres")
    print(f"   - Longitud mediana: {df['longitud_texto'].median():.1f} caracteres")
    
    # Top títulos más comunes
    titulo_counts = df['titulo'].value_counts().head(10)
    print(f"\n🏆 Top 10 títulos más comunes:")
    for titulo, count in titulo_counts.items():
        print(f"   - '{titulo}': {count} menciones")
    
    return df


def visualize_dataset(df):
    """
    Genera visualizaciones del dataset.
    """
    if df.empty:
        return
        
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Distribución por tipo
    df['tipo'].value_counts().plot.pie(ax=axes[0,0], autopct='%1.1f%%', colors=['lightcoral', 'lightblue'])
    axes[0,0].set_title('Distribución por Tipo (Real/Falso)')
    
    # Distribución por etapa
    df['etapa'].value_counts().plot.bar(ax=axes[0,1], color='skyblue')
    axes[0,1].set_title('Distribución por Etapa')
    axes[0,1].tick_params(axis='x', rotation=45)
    
    # Distribución de longitudes
    axes[1,0].hist(df['longitud_texto'], bins=30, alpha=0.7, color='lightgreen')
    axes[1,0].set_title('Distribución de Longitudes de Texto')
    axes[1,0].set_xlabel('Longitud (caracteres)')
    axes[1,0].set_ylabel('Frecuencia')
    
    # Longitud por tipo
    df.boxplot(column='longitud_texto', by='tipo', ax=axes[1,1])
    axes[1,1].set_title('Longitud por Tipo')
    axes[1,1].set_xlabel('Tipo')
    axes[1,1].set_ylabel('Longitud (caracteres)')
    
    plt.tight_layout()
    plt.show()


# =======================
# Construir base en RAM
# =======================
def build_ram_db(docs, modelos):
    """
    Genera embeddings y los almacena en memoria con estadísticas.
    """
    RAM_DB = {model_id: {} for model_id in modelos.keys()}
    stats = {model_id: {"chunks_generados": 0, "errores": 0} for model_id in modelos.keys()}

    for model_id, model_name in modelos.items():
        print(f"\n🧠 Generando embeddings con modelo {model_name} ({model_id})")
        try:
            tokenizer, model = get_embedding_model(model_name)
        except Exception as e:
            print(f"❌ Error cargando modelo {model_name}: {str(e)}")
            stats[model_id]["errores"] += 1
            continue

        for doc in tqdm(docs, desc=f"Procesando documentos con {model_id}"):
            try:
                chunks = chunk_text(doc["texto"])
                if not chunks:
                    continue
                    
                embeddings = embed_texts(chunks, tokenizer, model)
                stats[model_id]["chunks_generados"] += len(chunks)

                for i, emb in enumerate(embeddings):
                    key = f"{model_id}_{doc['id_relato']}_{doc['tipo']}_{i}"
                    RAM_DB[model_id][key] = {
                        "archivo": doc["archivo"],
                        "etapa": doc["etapa"],
                        "relato_id": doc["id_relato"],
                        "tipo": doc["tipo"],
                        "titulo": doc["titulo"],
                        "texto": chunks[i],
                        "longitud_texto": len(chunks[i]),
                        "embedding": emb.tolist(),
                        "modelo": model_id
                    }
            except Exception as e:
                print(f"❌ Error procesando documento {doc['id_relato']}: {str(e)}")
                stats[model_id]["errores"] += 1
                continue
                
    # Mostrar estadísticas de generación
    print("\n" + "="*50)
    print("📊 ESTADÍSTICAS DE EMBEDDINGS")
    print("="*50)
    for model_id, stat in stats.items():
        print(f"Modelo {model_id}:")
        print(f"  - Chunks generados: {stat['chunks_generados']}")
        print(f"  - Errores: {stat['errores']}")
    
    return RAM_DB


# =======================
# FAISS index (in-memory)
# =======================
def store_in_faiss(embeddings_dict, model_id=None, save=False, save_dir="faiss_data"):
    """
    Crea un índice FAISS en memoria (y opcionalmente lo guarda en disco).
    """
    if not embeddings_dict:
        raise ValueError("El diccionario de embeddings está vacío")
        
    first_key = next(iter(embeddings_dict.keys()))
    sample_emb = np.array(embeddings_dict[first_key]["embedding"], dtype="float32")
    dim = sample_emb.shape[0]
    
    index = faiss.IndexFlatL2(dim)
    keys = []
    vectors = []

    for k, v in embeddings_dict.items():
        keys.append(k)
        vectors.append(np.array(v["embedding"], dtype="float32"))

    matrix = np.vstack(vectors)
    index.add(matrix)

    if save and model_id:
        save_faiss_index(model_id, index, keys, save_dir)
    
    return index, keys



# =======================
# Búsqueda semántica mejorada
# =======================
def semantic_search(query, model_id, modelos, RAM_DB, top_k=5):
    """
    Busca el texto más similar en el índice del modelo elegido.
    
    Args:
        query: Texto de consulta
        model_id: ID del modelo a usar
        modelos: Diccionario de modelos
        RAM_DB: Base de datos en memoria
        top_k: Número de resultados a retornar
    """
    if model_id not in RAM_DB or not RAM_DB[model_id]:
        print(f"❌ No hay embeddings disponibles para el modelo {model_id}")
        return
        
    print(f"\n🔍 Buscando con modelo {modelos[model_id]}...")
    print(f"📝 Consulta: '{query}'")
    
    try:
        tokenizer, model = get_embedding_model(modelos[model_id])
        query_emb = embed_texts([query], tokenizer, model)[0].astype("float32")

        # Intentar cargar índice desde disco
        index, keys = load_faiss_index(model_id)

        # Si no existe, crear y guardar
        if index is None:
            index, keys = store_in_faiss(RAM_DB[model_id], model_id=model_id, save=True)

        distances, idxs = index.search(np.array([query_emb]), k=top_k)

        print(f"\n🎯 Top {top_k} resultados más relevantes:")
        print("-" * 80)
        
        results = []
        for i, (idx, dist) in enumerate(zip(idxs[0], distances[0])):
            entry = RAM_DB[model_id][keys[idx]]
            results.append({
                "rank": i + 1,
                "score": float(1 / (1 + dist)),  # Convertir distancia a similitud
                "distance": float(dist),
                **entry
            })
            
            print(f"\n#{i+1} | Similitud: {1/(1+dist):.3f} | Distancia: {dist:.4f}")
            print(f"📖 Tipo: {entry['tipo'].upper()} | Etapa: {entry['etapa']}")
            print(f"🏷️ Título: {entry['titulo']}")
            print(f"📝 Texto: {entry['texto'][:200]}...")
            print(f"📊 Longitud: {entry['longitud_texto']} caracteres")
            print(f"🔧 Modelo: {entry['modelo']}")
            print("-" * 60)
            
        return results
        
    except Exception as e:
        print(f"❌ Error en la búsqueda: {str(e)}")
        return None


def compare_models(query, modelos, RAM_DB, top_k=3):
    """
    Compara resultados de búsqueda entre diferentes modelos.
    """
    print(f"\n🔬 COMPARANDO MODELOS PARA: '{query}'")
    print("="*100)
    
    all_results = {}
    for model_id in modelos.keys():
        if model_id in RAM_DB and RAM_DB[model_id]:
            print(f"\n🧩 Modelo: {modelos[model_id]} ({model_id})")
            results = semantic_search(query, model_id, modelos, RAM_DB, top_k)
            all_results[model_id] = results
    
    return all_results


# =======================
# MAIN mejorado
# =======================
if __name__ == "__main__":
    # Cargar y analizar datos
    print("🚀 INICIANDO SISTEMA DE BÚSQUEDA SEMÁNTICA")
    docs = load_documents("data")
    
    if not docs:
        print("❌ No se pudieron cargar documentos. Saliendo...")
        exit(1)
    
    # Análisis de datos
    df = analyze_dataset(docs)
    visualize_dataset(df)
    
    # Construir base de datos de embeddings
    RAM_DB = build_ram_db(docs, MODELOS)
    
    # Ejemplos de búsqueda
    consultas = [
        "Batallas importantes durante la independencia de México",
        "Personajes históricos de la revolución mexicana",
        "Eventos culturales en el México prehispánico"
    ]
    
    for consulta in consultas:
        # Búsqueda individual
        semantic_search(consulta, model_id="m1", modelos=MODELOS, RAM_DB=RAM_DB, top_k=3)
        
        # Comparación entre modelos (opcional - comentado por rendimiento)
        # compare_models(consulta, MODELOS, RAM_DB, top_k=2)