import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

X = np.load("Ejercicios/GPT/Sci-Kit Learn/particiones-datos-desbalanceados.npz")['X']
Y = np.load("Ejercicios/GPT/Sci-Kit Learn/particiones-datos-desbalanceados.npz")['Y']

#Dividimos los datos en 60% train, 40% test
x_train, x_resto, y_train, y_resto = train_test_split(
  X, Y, test_size=0.4, random_state=42, stratify=Y,
)

#Dividimos el test en validacion y prueba
x_test, x_val, y_test, y_val = train_test_split(
  x_resto, y_resto, test_size=0.5, random_state=42, stratify=y_resto,
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

#pipeline
pipeline = Pipeline([
  ('scaler', MinMaxScaler(feature_range=(-1,1))),
  ('clasifier', RandomForestClassifier())
])
pipeline.fit(x_train, y_train)

print(pipeline.score(x_test, y_test))
print(pipeline.predict(x_test))