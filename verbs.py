import spacy
from collections import Counter
import json
from pathlib import Path
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Cargar modelo español
nlp = spacy.load("es_core_news_lg")

def extraer_noun_verbos(texto):
    """Extrae sustantivos y verbos (en infinitivo, combinando auxiliares)."""
    doc = nlp(texto)
    verbos_contador = Counter()
    sustantivos_contador = Counter()

    for token in doc:
        if token.is_punct or token.is_space:
            continue

        # Sustantivos
        if token.pos_ == "NOUN":
            sustantivos_contador[token.lemma_.lower()] += 1

        # Verbos o auxiliares
        elif token.pos_ in ["VERB", "AUX"]:
            aux_tokens = [child for child in token.children if child.pos_ == "AUX"]
            if aux_tokens:
                aux_lemmas = " ".join(sorted({aux.lemma_.lower() for aux in aux_tokens}))
                verbo_compuesto = f"{aux_lemmas} {token.lemma_.lower()}"
                verbos_contador[verbo_compuesto] += 1
            else:
                verbos_contador[token.lemma_.lower()] += 1

    return sustantivos_contador, verbos_contador


def procesar_carpeta(ruta_carpeta):
    """Procesa todos los archivos .txt de una carpeta."""
    ruta = Path(ruta_carpeta)
    todos_sustantivos = Counter()
    todos_verbos = Counter()

    for archivo in ruta.glob("*.txt"):
        print(f"📄 Procesando: {archivo.name}")
        texto = archivo.read_text(encoding="utf-8", errors="ignore")
        sustantivos, verbos = extraer_noun_verbos(texto)
        todos_sustantivos.update(sustantivos)
        todos_verbos.update(verbos)

    resultado = {
        "nouns": dict(todos_sustantivos),
        "verbs": dict(todos_verbos)
    }
    return resultado


def generar_wordcloud(diccionario, titulo, nombre_archivo):
    """Genera y guarda una nube de palabras."""
    texto_frecuencias = " ".join(
        [f"{palabra} " * freq for palabra, freq in diccionario.items()]
    )

    wc = WordCloud(
        width=1000,
        height=600,
        background_color="white",
        colormap="plasma",
        collocations=False
    ).generate(texto_frecuencias)

    plt.figure(figsize=(10, 6))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.title(titulo, fontsize=16)
    plt.tight_layout()
    plt.savefig(nombre_archivo, dpi=300)
    plt.show()
    print(f"✅ WordCloud guardado en {nombre_archivo}")


if __name__ == "__main__":
    carpeta = "ruta/a/tu/carpeta"  # 🔁 Cambia esta ruta
    resultado = procesar_carpeta(carpeta)

    # Guardar resultados
    with open("resultado.json", "w", encoding="utf-8") as f:
        json.dump(resultado, f, indent=4, ensure_ascii=False)

    print("\n✅ Análisis completado. Resultados guardados en 'resultado.json'")

    # Crear nubes de palabras
    generar_wordcloud(resultado["nouns"], "Nube de Sustantivos", "wordcloud_nouns.png")
    generar_wordcloud(resultado["verbs"], "Nube de Verbos", "wordcloud_verbs.png")
