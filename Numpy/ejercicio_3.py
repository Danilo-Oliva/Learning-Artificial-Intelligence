import numpy as np

a = np.arange(1, 21)
print(a)
print(f"Primeros 5 elementos: {a[:5]}")
print(f"Ultimos 5 elementos {a[-5:]}")
print(f"De par a par: {a[a % 2 == 0]}")

m = a.reshape(4,5)
print("Matriz: \n", m)
print(f"Segunda fila: {m[1, :]}")
print(f"Tercera columna: {m[ :, 2]}")
print(f"Suma: {m.sum(axis=1)}")

rng = np.random.default_rng(0)
X = rng.integers(1, 10, size=(3,3))
det = np.linalg.det(X)
print("X:\n", X)
print("det(X):", det)
if not np.isclose(det, 0):
    print("inv(X):\n", np.linalg.inv(X))
else:
    print("X no es invertible")
