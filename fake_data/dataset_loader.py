import os
import re
import pandas as pd
from lxml import etree
import matplotlib.pyplot as plt

class DatasetLoader:
    def __init__(self, data_folder="data"):
        self.data_folder = data_folder
        self.docs = []

    def load_documents(self):
        if not os.path.exists(self.data_folder):
            raise FileNotFoundError(f"La carpeta '{self.data_folder}' no existe.")
        
        files = [f for f in os.listdir(self.data_folder) if f.endswith('.xml')]
        if not files:
            raise FileNotFoundError(f"No hay archivos XML en '{self.data_folder}'")

        for file in files:
            tree = etree.parse(os.path.join(self.data_folder, file))
            root = tree.getroot()
            etapa = root.attrib.get("etapa", "desconocida")

            for relato in root.xpath("//relato"):
                self.docs.append({
                    "archivo": file,
                    "etapa": etapa,
                    "id_relato": relato.get("id", "sin_id"),
                    "titulo": relato.findtext("titulo", "").strip(),
                    "tipo": "falso",
                    "texto": relato.findtext("texto_falso", "").strip()
                })
                self.docs.append({
                    "archivo": file,
                    "etapa": etapa,
                    "id_relato": relato.get("id", "sin_id"),
                    "titulo": relato.findtext("titulo", "").strip(),
                    "tipo": "real",
                    "texto": relato.findtext("datos_reales", "").strip()
                })
        return self.docs

    def chunk_text(self, text, min_chunk_size=50, max_chunk_size=1500):
        sentences = re.split(r'[.!?¡¿]+\s*', text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
        if not sentences:
            return []

        chunks, current_chunk = [], ""
        for s in sentences:
            if len(current_chunk) + len(s) > max_chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = s
            else:
                current_chunk = f"{current_chunk}. {s}" if current_chunk else s
        if current_chunk and len(current_chunk) >= min_chunk_size:
            chunks.append(current_chunk.strip())
        return chunks

    def analyze_dataset(self):
        df = pd.DataFrame(self.docs)
        print(f"Total documentos: {len(df)} | Etapas: {df['etapa'].unique()}")
        print(df['tipo'].value_counts())
        return df

    def visualize_dataset(self, df):
        df['longitud'] = df['texto'].str.len()
        df.boxplot(column='longitud', by='tipo')
        plt.title("Distribución de longitudes por tipo")
        plt.show()
