import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

data = pd.read_csv("Ejercicios/GPT/Sci-Kit Learn/Video de 5hs/Advertising.csv")
print(data.head())

#Eliminamos la fila unnamed
data = data.iloc[:, 1:]
print(data.head())

#creamos graficos
def ShowPlt(data):
  cols = ['TV', 'Radio', 'Newspaper']
  for col in cols:
    plt.plot(data[col], data['Sales'], 'ro')
    plt.title('Ventas respecto a la publicidad en %s' %col)
    plt.show()

x = data['TV'].values.reshape(-1, 1)
y = data['Sales'].values

x_train, x_test, y_train, y_test = train_test_split(
  x, y, test_size=0.3, random_state=23
)

lin_reg = LinearRegression()
lin_reg.fit(x_train, y_train)

y_pred = lin_reg.predict(x_test)

print('Pred: {}, reales: {}'.format(y_pred[:4], y_test[:4]))

RMSE = root_mean_squared_error(y_test, y_pred)

print('RMSE', RMSE)
print('R2', r2_score(y_test, y_pred))