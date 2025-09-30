import numpy as np
from sklearn.model_selection import KFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier

X = np.array([
    [170, 65], [160, 55], [180, 80], [175, 77],
    [155, 50], [185, 90], [165, 70], [172, 68]
])
y = np.array([0, 0, 1, 1, 0, 1, 0, 1])

knn = KNeighborsClassifier(n_neighbors=3)

# Definimos K-Fold con 4 particiones
kf = KFold(n_splits=4, shuffle=True, random_state=42)

# Evaluamos con cross_val_score
scores = cross_val_score(knn, X, y, cv=kf, scoring="accuracy")

print("Accuracy en cada fold:", scores)
print("Accuracy promedio:", scores.mean())