import pandas as pd
import nltk
from sklearn.model_selection import train_test_split

nltk.download('punkt')

# ahora cargo los datos csv, use la varible datos para guardarlos
# si se desea cargar con otra ubicacion del archivo colocarla en las comillas simples ('') 
# y tener encuenta que windos coloca / invertidos se deben cambiar para que no arroje falla cuando compile
# si copias y pegas la ruta

datos = pd.read_csv('D:/Users/Maycol Roa/Desktop/prueba_it power_BI/twitter_training_csv.csv')

# con los datos cargados realizo la divicion de los datos en 60% 20% 20%

train_data, temp_data = train_test_split(datos, test_size=0.4, random_state=42)
valid_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)

# ahora asignamos variables para verificar el total de datos y las diviciones de los datos en 60%, 20%, 20%
# use len para delvolver la cantidad total de filas de mi dataframe

total_samples = len(datos)
train_samples = len(train_data)
valid_samples = len(valid_data)
test_samples = len(test_data)

# print para mostrar las longitudes y mensajes 

print(f"Total de muestras: {total_samples}")
print(f"Muestras en el conjunto de entrenamiento: {train_samples} ({(train_samples/total_samples)*100:.2f}%)")
print(f"Muestras en el conjunto de validación: {valid_samples} ({(valid_samples/total_samples)*100:.2f}%)")
print(f"Muestras en el conjunto de prueba: {test_samples} ({(test_samples/total_samples)*100:.2f}%)")

# tarea 2 ¿Cuántos tweets se obtienen por cada conjunto? 
# Imprima la codificación de los encabezados para el conjunto de entrenamiento.

# debido a que hay caracteres que no se dejan imprimir se debe realizar una limpieza de los datos
# debido a su tamaño intente guardarlos en una lista vacia "problematic_texts" para analizarlos y reemplazarlos
# imprimi las cadenas de texto con problemas 

# Identificar las cadenas problemáticas en el conjunto de entrenamiento
problematic_texts = []
for text in train_data.iloc[:, 0]:
    try:
        nltk.word_tokenize(text)
    except (TypeError, LookupError):
        problematic_texts.append(text)

# Imprimir las cadenas problemáticas para identificar y modificarlas 
print("Cadenas de texto problemáticas:")
for text in problematic_texts:
    print(text)

# Realizar limpieza de texto por saltos de linea por espacio y eliminar espacios al final 
def clean_text(text):
    text = text.replace('\n', ' ')  # Reemplazar saltos de línea por espacios
    text = text.strip()  # Eliminar espacios en blanco al principio y al final
    return text

# Aplicar limpieza a las cadenas problemáticas pero hace falta agregar mas casos 
cleaned_problematic_texts = [clean_text(text) for text in problematic_texts]

# Modelado en n-gramas de las cadenas de texto (ejemplo con bigramas)
n = 2
train_ngrams = [' '.join(nltk.ngrams(nltk.word_tokenize(text), n)) for text in train_data.iloc[:, 0]]
valid_ngrams = [' '.join(nltk.ngrams(nltk.word_tokenize(text), n)) for text in valid_data.iloc[:, 0]]
test_ngrams = [' '.join(nltk.ngrams(nltk.word_tokenize(text), n)) for text in test_data.iloc[:, 0]]

# Codificación one-hot para las variables
train_encoded = pd.get_dummies(train_data['clase'])
valid_encoded = pd.get_dummies(valid_data['clase'])
test_encoded = pd.get_dummies(test_data['clase'])

# Obtener el número de tweets en cada conjunto
print(f"Número de tweets en el conjunto de entrenamiento: {len(train_ngrams)}")
print(f"Número de tweets en el conjunto de validación: {len(valid_ngrams)}")
print(f"Número de tweets en el conjunto de prueba: {len(test_ngrams)}")

# Imprimir la codificación de los encabezados para el conjunto de entrenamiento
print("Codificación de los encabezados para el conjunto de entrenamiento:")
print(train_encoded.columns)

