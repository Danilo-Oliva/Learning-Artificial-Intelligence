#paso 1 librerias
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Paso 2: generar datos sintéticos
# reproducibilidad
np.random.seed(42)

# personas que hacen deporte (1): suelen ser más jóvenes y con peso medio
edad_deportistas = np.random.randint(18, 35, 50)
peso_deportistas = np.random.randint(55, 80, 50)

# personas que NO hacen deporte (0): suelen ser mayores y con más peso
edad_no = np.random.randint(30, 60, 50)
peso_no = np.random.randint(70, 100, 50)

# unimos en una sola tabla
X = np.column_stack((
    np.concatenate([edad_deportistas, edad_no]),
    np.concatenate([peso_deportistas, peso_no])
))
y = np.array([1]*50 + [0]*50)

#paso 3 dividir datos
x_train, x_test, y_train, y_test = train_test_split(
  X, y, test_size= 0.3, random_state=42 
)

for k in range(1, 6):
    modelo = KNeighborsClassifier(n_neighbors=k)
    modelo.fit(x_train, y_train)
    y_pred = modelo.predict(x_test)
    print(f"k={k} → Precisión: {accuracy_score(y_test, y_pred)}")


#paso 4 elegir modelo
modelo = KNeighborsClassifier(n_neighbors=3)

#paso 5 entrenar
modelo.fit(x_train, y_train)

#paso 6 predecir
y_pred = modelo.predict(x_test)

#paso 7 medir precision
print("Precisión: ", accuracy_score(y_test, y_pred))

#datos nuevos
nuevo = [[25, 68], [45, 85], [19, 60]]
print("Predicciones:", modelo.predict(nuevo))

# --------- GRAFICAR FRONTERA ----------
# Definir los límites del gráfico
x_min, x_max = X[:,0].min() - 1, X[:,0].max() + 1
y_min, y_max = X[:,1].min() - 1, X[:,1].max() + 1

# Crear una cuadrícula de puntos
xx, yy = np.meshgrid(
    np.arange(x_min, x_max, 0.5),
    np.arange(y_min, y_max, 0.5)
)

# Predecir la clase de cada punto de la cuadrícula
Z = modelo.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Colorear el fondo según la clase
plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)

# Graficar los puntos de entrenamiento
plt.scatter(x_train[y_train==1,0], x_train[y_train==1,1], c="blue", label="Deportistas")
plt.scatter(x_train[y_train==0,0], x_train[y_train==0,1], c="red", label="No deportistas")

# Graficar puntos de test
plt.scatter(x_test[:,0], x_test[:,1], c="yellow", edgecolors="k", marker="o", label="Test")

plt.xlabel("Edad")
plt.ylabel("Peso")
plt.legend()
plt.title("Frontera de decisión KNN")
plt.show()