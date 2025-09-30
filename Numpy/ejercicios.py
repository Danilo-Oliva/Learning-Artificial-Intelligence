import numpy as np
import matplotlib.pyplot as plt

#Array
a = np.arange(1, 100)
print("Array:", a)

#Calcular el cuadrado de cada elemento
print("Cuadrado de cada elemento:", a**2)

#Generar una matriz de 3x3 con valores aleatorios entre 0 y 1
rg = np.random.default_rng(2)
b = rg.random((3,3))
print("Matriz 3x3 con valores aleatorios entre 0 y 1:\n", b)

print(f"Suma de cada fila: {b.sum(axis=1)}")