#Paso 1: importar librerias
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

#Paso 2: cargar datos
iris = load_iris()
x = iris.data     #caracteristicas (largo y anicho de sepalos y petalos)
y = iris.target   #etiquetas(tipo de flor)

#Paso 3: dividir datos en entrenamiento y prueba
x_train, x_test, y_train, y_test = train_test_split(
  x, y, test_size=0.2, random_state=42
)

#Paso 4: Elegir modelo (K-Nereast Neighbors)
modelo = KNeighborsClassifier(n_neighbors=3)

#Paso 5: entrenar
modelo.fit(x_train, y_train)

#Paso 6: Predecir
y_pred = modelo.predict(x_test)

#Paso 7: medir precisión
print("Precisión: ", accuracy_score(y_test, y_pred))