import wikipedia
import os

wikipedia.set_lang("es")

os.makedirs("corpus", exist_ok=True)

info = [
    "Juan de Palafox y Mendoza",
    "Biblioteca Palafoxiana",
    "Consejo de Indias",
    "Puebla de Zaragoza"
    "Ignacio Zaragoza",
    "Porfirio Díaz",
    "Catedral metropolitana de Puebla",
    "Constitución Política de los Estados Unidos Mexicanos",
    "Independencia de México",
    "Etapa de Resistencia de la Independencia de México",
    "Revolución mexicana",
    "Benito Juárez",
    "Miguel Hidalgo y Costilla",
    "José María Morelos",
    "Porfiriato",
    "Leyes de Reforma",
    "Guerra de Reforma"
]

# for i in info:
#     print(
#         i, "\n",
#         wikipedia.search(i), "\n"
#     )


for i in info:
    try:
        page = wikipedia.page(i)
        
        filename = i.replace(" ", "_").replace("/", "-") + ".txt"
        filepath = os.path.join("corpus", filename)
        
        # Guardar el contenido en el archivo
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(f"Título: {page.title}\n")
            f.write(f"URL: {page.url}\n")
            f.write(f"Resumen: {page.summary}\n")
            f.write("\n" + "="*50 + "\n")
            f.write("CONTENIDO COMPLETO:\n")
            f.write("="*50 + "\n")
            f.write(page.content)
        
        print(f"✓ Guardado: {filename}")
        
    except wikipedia.exceptions.DisambiguationError as e:
        print(f"Error de ambigüedad para '{i}': {e.options}")
    except wikipedia.exceptions.PageError:
        print(f"Página no encontrada: '{i}'")
    except Exception as e:
        print(f"Error al procesar '{i}': {e}")

print("\n¡Proceso completado! Los archivos se guardaron en la carpeta 'corpus'")