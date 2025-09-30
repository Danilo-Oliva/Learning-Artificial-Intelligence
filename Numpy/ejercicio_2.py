import numpy as np
import matplotlib.pyplot as plt

#matriz 3x2 con valores entre 1 y 6
a = np.arange(1, 7).reshape(3, 2)
print("Matriz 3x2 con valores entre 1 y 6:\n", a)

#matriz 3x2 con valores entre 7 y 12
b = np.arange(7, 13).reshape(2, 3)
print("Matriz 3x2 con valores entre 7 y 12:\n", b)

print(f"Transpuesta de a: \n {a.T}")
print(f"Producto matricial entre a y b: \n {a @ b}")
print(f"Producto matricial entre b y a: \n {b @ a}")

c = a @ b
print(f"Determinante de a@b: \n {np.linalg.det(c)}")
print(f"Inversa de a@b: \n {np.linalg.inv(c)}")