import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

x, y = make_classification(
  n_samples=20,
  n_features=2,
  n_classes=2,
  n_informative=2,
  n_redundant=0,
  random_state=42
)

x_train, x_test, y_train, y_test = train_test_split(
  x, y, test_size=0.3, random_state=42, stratify=y 
)

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(x_train, y_train)

y_pred = knn.predict(x_test)

print("Predicciones: ", y_pred)
print("Reales: ", y_test)
print("Accuaracy: ", accuracy_score(y_test, y_pred))
print("Matriz de confusión: ", confusion_matrix(y_test, y_pred))