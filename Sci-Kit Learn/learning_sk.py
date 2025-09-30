import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.neighbors import KNeighborsClassifier
import warnings

warnings.filterwarnings("ignore")

x = np.array([[170, 65], [160, 55], [180, 80],[175, 77],[155,50],[185, 90]])

y = np.array([0, 0, 1, 1, 0, 1])

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

knn = KNeighborsClassifier(n_neighbors=4)

knn.fit(x_train, y_train)

y_pred = knn.predict(x_test)

print("Predicciones:", y_pred)
print("Reales:", y_test)
print("Accuaracy:", accuracy_score(y_test, y_pred))
print("Matriz de confusión:\n", confusion_matrix(y_test, y_pred))
