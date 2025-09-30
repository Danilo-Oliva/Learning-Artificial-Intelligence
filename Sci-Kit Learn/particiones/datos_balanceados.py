import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier

#Cargamos el archivo
X = np.load("Ejercicios/GPT/Sci-Kit Learn/particiones-datos-balanceados.npz")['X']
Y = np.load("Ejercicios/GPT/Sci-Kit Learn/particiones-datos-balanceados.npz")['Y']

#Partimos los datos en 60% entrenamiento y 40% de resto
x_train, x_resto, y_train, y_resto = train_test_split(
  X, Y, test_size=0.4, random_state=42
)

#Partimos el resto en 50% prueba, 50% validación
x_test, x_val, y_test, y_val = train_test_split(
  x_resto, y_resto, test_size=0.5, random_state=321
)

print("Tamaños: ")
print("Dataset original: ",X.shape, Y.shape)
print("Entrenamiento: ", x_train.shape, y_train.shape)
print("Validación: ", x_val.shape, y_val.shape)
print("Prueba: ", x_test.shape, y_test.shape)

print("\nProporciones categorias(0s/1s)")
print(f"Dataset Original: , {np.sum(Y==0)/len(Y)}/{np.sum(Y==1)/len(Y)}")
print(f"Entrenamiento: {np.sum(y_train == 0)/len(y_train)}/{np.sum(y_train==1)/len(y_train)}")
print(f"Validación: {np.sum(y_val == 0)/len(y_val)}/{np.sum(y_val==1)/len(y_val)}")
print(f"Prueba: {np.sum(y_test == 0)/len(y_test)}/{np.sum(y_test==1)/len(y_test)}")

#transformo los datos, el rango entre los numeros que hay
print(f'Train: {x_train.min(axis=0)}/{x_train.max(axis=0)}')
print(f'Validacion: {x_val.min(axis=0)}/{x_val.max(axis=0)}')
print(f'Test: {x_test.min(axis=0)}/{x_test.max(axis=0)}')

scaler = MinMaxScaler(feature_range=(-1, 1))
x_train_s = scaler.fit_transform(x_train)

print(f'Train s: {x_train_s.min(axis=0)}/{x_train_s.max(axis=0)}')

#Transformamos el resto de set
x_val_s = scaler.transform(x_val)
x_test_s = scaler.transform(x_test)

#Creamos estimadores
bosque = RandomForestClassifier()

#entrenamos
bosque.fit(x_train, y_train)

bosque.score
print(f"Exactitud promedio train: {bosque.score(x_train, y_train)}")
print(f"Exactitud promedio validación: {bosque.score(x_val, y_val)}")

y_pred = bosque.predict(x_test)
print("Categorias reales: ",y_test)
print("Categoria predicha: ", y_pred)