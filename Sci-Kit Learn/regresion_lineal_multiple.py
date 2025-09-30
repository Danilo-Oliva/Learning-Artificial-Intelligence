import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

X = np.array([
    [10, 1.40],  # edad 10 años, altura 1.40 m
    [15, 1.55],
    [20, 1.70],
    [25, 1.75],
    [30, 1.80],
    [35, 1.82],
    [40, 1.85]
])
y = np.array([30, 45, 60, 68, 72, 75, 78])  # peso en kg

modelo = LinearRegression()
modelo.fit(X, y)

print("Peso estimado:", modelo.coef_)

# Como ahora tenemos 2 variables (edad y altura), ya no se puede graficar con una recta simple.
# Solo mostramos los coeficientes:
print("Coeficientes:", modelo.coef_)
print("Intercepto:", modelo.intercept_)
