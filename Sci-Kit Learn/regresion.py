import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Paso 1: Datos (metros cuadrados y precios en miles de dólares)
X = np.array([[30], [50], [70], [90], [110], [130], [150]])  # m²
y = np.array([80, 120, 150, 180, 200, 220, 250])             # precio

X_train, X_test, y_train, y_test = train_test_split(
  X, y, test_size=0.3, random_state=42
)

modelo = LinearRegression()
modelo.fit(X_train, y_train)

y_pred = modelo.predict(X_test)

print("Error cuadratico medio: ", mean_squared_error(y_test, y_pred))
plt.scatter(X, y, color="blue", label="Datos reales")
plt.plot(X, modelo.predict(X), color="red", label="Recta de regresión")
plt.xlabel("Metros cuadrados")
plt.ylabel("Precio (miles de USD)")
plt.legend()
plt.show()