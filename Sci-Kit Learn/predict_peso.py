import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

x = np.array([[10], [15], [20], [25], [30], [35], [40]])
y = np.array([30, 40, 55, 65, 70, 72, 75])

modelo = LinearRegression()
modelo.fit(x, y)

print("Peso estimado: ", modelo.predict([[28]]))

# Graficar
def showPlot():
  plt.scatter(x, y, color="blue", label="Datos reales")
  plt.plot(x, modelo.predict(x), color="red", label="Regresión lineal")
  plt.xlabel("Edad (años)")
  plt.ylabel("Peso (kg)")
  plt.legend()
  plt.show()