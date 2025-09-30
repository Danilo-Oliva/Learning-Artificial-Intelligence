import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.neighbors import KNeighborsClassifier


x = np.array([
 [320451, 54782],
 [145873, 99432],
 [276512, 30219],
 [398722, 45312],
 [120384, 87321],
 [305671, 12543],
 [210987, 67890],
 [380124, 49283],
 [199998, 80123],
 [145678, 23456]])

y = np.array([0, 1, 0, 0, 1, 0, 1, 0, 1, 0])

x_train, x_test, y_train, y_test = train_test_split(
  x, y, test_size=0.3, random_state=42
)

knn =KNeighborsClassifier(n_neighbors=1)

knn.fit(x_train, y_train)

y_pred = knn.predict(x_test)

print("Predicciones:", y_pred)
print("Reales:", y_test)
print("Accuaracy:", accuracy_score(y_test, y_pred))
print("Matriz de confusión:\n", confusion_matrix(y_test, y_pred))