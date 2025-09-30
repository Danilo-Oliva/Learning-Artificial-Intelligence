import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

rg = np.random.default_rng(2)

altura_cm = rg.integers(150, 191, 8) 

print(altura_cm)

#Crear una serie, es una columna
altura = pd.Series(altura_cm)
print(altura)

#Agarrar los primeros 5 elementos o lo que querramo especificando en el ()
print(altura.head())

#Ordenar de menor a mayor, pero no cambia la lista original
print(altura.sort_values())

#maximo, minimo y promedio
print(f"Min: {altura.min()}")
print(f"Max: {altura.max()}")
print(f"Promedio: {altura.mean()}")

#Obtener descripciones generales
print(f"\fDescripción:\f{altura.describe()}")

#Filtros
print("\fAltura mayor a 170: ")
print(altura[altura > 170])

print("\f")
#Datos
datos = {"Nombre" : ["Juan", "Pedro", "Mateo", "Lucas", "Marcos"],
         "Edad" : rg.integers(20, 40, 5),
         "Altura" : rg.integers(160, 190, 5),
         "Peso" : rg.integers(60, 90, 5)
}
amigos = pd.DataFrame(datos)
print(amigos)

print(f"\fDescripción:\f{amigos.describe()}")

print("\f")
#Filtrar datos, si quiero que solo me muestren la edad y no el resto aplico = amigos[amigos.Edad < 30].Edad
print("Personas menores a 30 años: ")
print(amigos[amigos.Edad < 30])

#Abrir tablas de otro tipo de archivo
print("\f")
print("Amigos de excel:")
amigos_exl = pd.read_excel("C:/Users/danil/Downloads/amigos.xlsx", index_col= "Id")
print(amigos_exl.head())
print("\fTotal de amigos:")
print(amigos_exl.count())

#Estudiantes
print("\f")
universidad = pd.read_excel("C:/Users/danil/Downloads/universidad.xlsx", index_col= "Id Persona")
print(universidad.head())

#Unir tablas
estudiantes = amigos_exl.merge(universidad, left_on = "Id", right_on= "Id Persona")
print("\f")
#cambiar una columna
estudiantes.loc[:, "Num Control"].str.upper()

#Cambiar el original
estudiantes.loc[:, "Num Control"] = estudiantes.loc[:, "Num Control"].str.upper()
print(estudiantes.head())

print("\f")

#Cambiar a mayusculas el titulo de la carrera
estudiantes.loc[:, "Carrera"] = estudiantes.loc[:, "Carrera"].str.title()
print(estudiantes.head())

print("\f")

#Agregar columna nueva de año
estudiantes["Año"] = estudiantes.Inscripción.dt.year
print(estudiantes.head())

print("\f")

#Agregar columna de año/mes
estudiantes["Año/Mes"] = estudiantes.Inscripción.dt.strftime("%Y/%m")
print(estudiantes.head())

print("\f")

#Filtrar solo los licenciados
licenciados = estudiantes[estudiantes.loc[:, "Carrera"].str.contains("Lic")]
print(licenciados.head())

#Filtrar licenciados por año
lic_x_año = licenciados.groupby(by = "Año").size()
print(lic_x_año)

print("\f")
print("Ingresos por año")
ingresos = licenciados.pivot_table(index="Año", columns="Carrera", values="Edad", aggfunc= "count", fill_value=0)
print(ingresos)