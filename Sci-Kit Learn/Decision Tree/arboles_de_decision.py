import numpy as np
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

np.random.seed(23)

edad_deportistas = np.random.randint(18, 30, 50)
peso_deportistas = np.random.randint(55, 75, 50)

edad_no = np.random.randint(30, 60, 50)
peso_no = np.random.randint(70, 100, 50)

x = np.column_stack((np.concatenate([edad_deportistas, edad_no]), np.concatenate([peso_deportistas, peso_no])))
y = np.array([1]*50 + [0]*50)

x_train, x_test, y_train, y_test = train_test_split(
  x, y, test_size=0.3, random_state=23
)

modelo = DecisionTreeClassifier(max_depth=3, random_state=23)
modelo.fit(x_train, y_train)

y_pred = modelo.predict(x_test)
print('Precisión: ', accuracy_score(y_test, y_pred))

def showPlot():
  plt.figure(figsize=(10, 6))
  plot_tree(modelo, feature_names=["Edad", "Peso"], class_names=["No deporte", "Deporte"], filled=True)
  plt.show()

showPlot()