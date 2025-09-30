import numpy as np
import matplotlib.pyplot as plt
from numpy import pi

#Array
a = np.array([1, 2, 3, 4, 5])
print("Array:", a)

#Matriz
b = np.array([[1, 2, 3],[4, 5, 6]])
print("Matriz:\n", b)

#Tamaño del array (filas, columas)
print(f"Tamaño del array: {b.shape}")

#Dimension del array
print(f"Dimension del array: {b.ndim}")

#Tipo de array
print(f"Tipo de array: {a.dtype}")

#Tamaño de elementos del array
print(f"Tamaño de elementos: {a.size}")

#Array vacia de ceros
print("Matriz de ceros: ")
c = np.zeros((3,2))
print(c)

#Array vacia de unos
print("Matriz de unos: ")
d = np.ones((3,3))
print(d)

#Array hasta numero indice 0
e = np.arange(10)
print(f"Array de 10 elementos: {e}")

#Matriz hasta numero
f = np.arange(10).reshape(2,5)
print(f"Matriz hasta 10: {f}")

#Array de numero a numero, saltando de numero
g = np.arange(10,20,2)
print(f"Array de 10 a 20, de 2 en 2: {g}")

#Array de numero a numero, te devuelve numero de elementos
h = np.linspace(10, 20, 11)
print(f"11 elementos dentro de 10 hasta 20: {h}")

#Array de 100 elementos
rad = np.linspace(0, 2*pi, 100)
print(rad)

seno = np.sin(rad)
coseno = np.cos(rad)

plt.plot(seno, coseno)
plt.show()

#suma de arrays
j = np.linspace(10, 20, 6)
k = np.linspace(5, 25, 6)
suma = j+k

print(suma)

#Concatenar
c = np.concatenate((j, k))
c.sort()
print(c)

#Numeros aleatorios
rg = np.random.default_rng(2)
aleatorio = rg.random(1000)
print(aleatorio)

plt.hist(aleatorio, bins = 100)
plt.show()

#Numeros aleatorios distribuidos
normal = rg.normal(10, 5, 100000)

plt.hist(normal, bins = 1000)
plt.show()

#Numeros aleatorios sin repetir
sin_repetir = rg.choice(21, 10, replace = False)
print(sin_repetir)

#Estadisticas y rg solo con enteros
print("Estadisticas: ")
estadisticas = rg.integers(20, size=8)
print(estadisticas)

#Estadisticas(rg) minimo, maximo, promedio, distribucion standard, suma de todos los rg
print(f"Min: {estadisticas.min()}")
print(f"Max: {estadisticas.max()}")
print(f"Promedio: {estadisticas.mean()}")
print(f"Distribucion promedio: {estadisticas.std()}")
print(f"Suma: {estadisticas.sum()}")

#Rg en una matriz
estadisticas_2d = rg.integers(20, size=(5,4))
print("Matriz estadisticas: ")
print(estadisticas_2d)

#Matriz min, max, prom, distri std, suma -> axis = 0 significa columnas, axis = 1 filas
print(f'Min columnas: {estadisticas_2d.min(axis=0)}')
print(f'Min filas: {estadisticas_2d.min(axis=1)}')
print(f'Max columnas: {estadisticas_2d.max(axis=0)}')
print(f'Max filas: {estadisticas_2d.max(axis=1)}')

#Saber condiciones, solo con uno devuelve true o false en la posicion
print(f'Stats menores a 12: {estadisticas_2d[estadisticas_2d < 12]}')

#Apilar uno sobre otro a las matrices, vertical u horizontalmente
np1 = rg.integers(20, size=(3,3))
np2 = rg.integers(20, size=(3,3))

print(np.vstack(np1,np2))
print(np.hstack(np1,np2))

#Seleccionar valores
enteros = rg.integers(20, size=10)

#selecciono (inicio : fin(sin tocar) : saltando)
print(f'Enteros de 0 - 5: {enteros[0:6]}')

#Selecciono el 0, 2 y 4
print(f'0 - 2 - 4: {enteros[0:6:2]}')

#Selecciono por par de indice( : : saltando) -> no me importa donde empieces, hasta donde vayas
print(f'Indice de 2 en 2: {enteros[::2]}')

#Seleccion de valores en matriz
enteros_2d = rg.integers(20, size = (8, 5))

#traer el cuarto elemento -> [fila, columna]
print(f'Traer cuarto elemento: {enteros_2d[1 , 3]}')

#Seleccionar el segundo valor con indice 3,4,5, [Empeza en la tercera fila, hasta la quinta : agarra la segunda columna en ese rango]
print(f'Indice 3, 4, 5: {enteros_2d[3:6.2]}')

#Seleccionar los dos primeros valores de indice 4, 5, 6
print(f'Segundo valor, indice 4, 5, 6: {enteros_2d[4:7, 0:2]}')