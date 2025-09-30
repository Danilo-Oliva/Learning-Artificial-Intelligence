import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

x = [[20, 60], [22, 65], [45, 90], [50, 85], [30, 70], [35, 95]]
y = [1, 1, 0, 0, 1, 0]

x_train, x_test, y_train, y_test = train_test_split(
  x, y, test_size=0.3, random_state=23
)

modelo_simple = DecisionTreeClassifier(max_depth=1, random_state=23)
modelo_simple.fit(x_train, y_train)
pred_simple = modelo_simple.predict(x_test)
print("Precisión (arbol simple): ", accuracy_score(y_test, pred_simple))

#arbol mas profundo
modelo_profundo = DecisionTreeClassifier(max_depth=5, random_state=23)
modelo_profundo.fit(x_train, y_train)
pred_profundo = modelo_profundo.predict(x_test)
print("Precisión (árbol profundo): ", accuracy_score(y_test, pred_profundo))

def showModeloSimple():
  plt.figure(figsize=(10,6))
  plot_tree(modelo_simple, feature_names=["Edad", "Peso"], class_names=["No", "Sí"], filled=True)
  plt.show()

def showModeloProfundo():
  plt.figure(figsize=(10,6))
  plot_tree(modelo_profundo, feature_names=["Edad", "Peso"], class_names=["No", "Sí"], filled=True)
  plt.show()