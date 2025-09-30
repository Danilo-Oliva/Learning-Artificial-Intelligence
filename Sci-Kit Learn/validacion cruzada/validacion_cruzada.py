import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, cross_validate

x = np.load("Ejercicios/GPT/Sci-Kit Learn/validacion cruzada/dataset_digitos.npz")['X']
y = np.load("Ejercicios/GPT/Sci-Kit Learn/validacion cruzada/dataset_digitos.npz")['Y']

#cambio el tamaño de las imagenes a un vector
x = x.reshape(x.shape[0], -1)
kf = KFold(n_splits=5, shuffle=True, random_state=42)

modelo = RandomForestClassifier(n_estimators=40)

k_fold = KFold(n_splits=10, shuffle=True, random_state=42)

#score promedio de train y validation
score_prom = cross_val_score(modelo, x, y, cv= k_fold)
#score separado de train y validation
score_train_val = cross_validate(modelo, x, y, cv=k_fold, return_train_score=True)

print("Validacion cruzada - Desempeño en cada iteracion:", score_prom)

print("Validacion cruzada - Promedio de los desempeños:", np.mean(score_prom))
print("Validacion cruzada - desviacion estandar de los desempeños:", np.std(score_prom))

print("Validacion separada: ")
print(score_train_val)

#una vez separado el train y validation, tomamos el promedio de cada uno
scores_tr = score_train_val['train_score']
scores_vl = score_train_val['test_score']
print(f'Desempeño de entrenamiento(media +/- desviacion): {scores_tr.mean():.3f} +/- {scores_tr.std():.3f} ')
print(f'Desempeño de validación(media +/- desviacion): {scores_vl.mean():.3f} +/- {scores_vl.std():.3f}')

#alerta de overfittin, hay que averiguar el valor exacto de k
SCR_STD_TR = []
SCR_MEAN_TR = []
SCR_STD_VL = []
SCR_MEAN_VL = []
K = []

for k in range(5, 95, 10):
  print(f'Validacion cruzada con k: {k}')
  modelo = RandomForestClassifier(n_estimators=40)
  k_fold = KFold(n_splits=k, shuffle=True, random_state=42)
  scores = cross_validate(modelo, x, y, cv=k_fold, return_train_score=True)
  
  #calculamos y almacenamos desempeños(medias y desviaciones)
  #asi como el valor de k
  SCR_MEAN_TR.append(scores['train_score'].mean())
  SCR_STD_TR.append(scores['train_score'].std())
  SCR_MEAN_VL.append(scores['test_score'].mean())
  SCR_STD_VL.append(scores['test_score'].std())
  K.append(k)
  
  plt.errorbar(K, SCR_MEAN_TR, yerr= SCR_STD_TR, fmt='0', capsize=5, label = 'Entrenamiento')
  plt.errorbar(K, SCR_MEAN_VL, yerr=SCR_STD_TR, fmt= '0', capsize=5, label = 'Validación')
  plt.xlabel('k')
  plt.ylabel('Promedio +/- Desviación')
  plt.legend()