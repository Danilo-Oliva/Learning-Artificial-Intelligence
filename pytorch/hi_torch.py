import torch
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch import nn#red neuronal
from torch.utils.data import DataLoader #para los batches
import matplotlib.pyplot as plt

#creando un tensor
arreglo = [[2, 3, 4], [1, 5, 6]]
tensor1 = torch.tensor(arreglo)

#en donde está conectado
tensor1.device

#ver si se está usando la gpu o la cpu
device = (
  "cuda" if torch.cuda.is_available()
  else "cpu"
)

#cambiar a GPU o cpu
#tensor1 = tensor1.to("cuda")
#tensor1 = tensor1.to("cpu")

#saber el tamaño del tensor
tensor1.shape

data_mnist = datasets.MNIST(
  root= "datos", #carpeta donde almaceno el dataset
  train=True,
  download=True,
  transform=ToTensor()
)

figure = plt.figure(figsize=(8, 8))
fils, cols = 3, 3

for i in range(1, cols * fils + 1):
    # Escoger una imagen aleatoria
    sample_idx = torch.randint(len(data_mnist), size=(1,)).item()

    # Extraer imagen y categoría
    img, label = data_mnist[sample_idx]

    # Dibujar
    figure.add_subplot(fils, cols, i)
    plt.title(str(label)) # Categoría
    plt.axis("off")
    plt.imshow(img.squeeze(), cmap="gray") # Imagen
plt.show()

torch.manual_seed(123)

train, val, test = torch.utils.data.random_split(
  data_mnist, [0.8, 0.1, 0.1]
)

def SplitSize():
  print(f"Cant de datos en train: {len(train)}")
  print(f"Cant de datos en validacion: {len(val)}")
  print(f"Cant de datos en prueba: {len(test)}")

class RedNeuronal(nn.Module):
  #Primer paso: metodo "init"
  def __init__(self):
    super().__init__()
    #agregar secuencialmente las capas
    self.aplanar = nn.Flatten() #aplana los datos en vectores
    self.red = nn.Sequential(
      nn.Linear(28*28, 15),#capa de entrada, capa oculta de neuronas
      nn.ReLU(),#Esto oculta la capa
      nn.Linear(15, 10),#Capa de salida. Viene 15 datos de 10 neuronas
    )
  #Segundo Paso: metodo "forward" (x = dato de entrada)
  def forward(self, x):
    #definir secuencialmente las operaciones a aplicar
    x = self.aplanar(x) #aplanamos los datos
    logits = self.red(x) #logits son los 10 valores numericos que vamos a obtener en la salida
    return logits
  
modelo = RedNeuronal().to(device)#que vaya a la CPU o la GPU
#print(modelo) -> saber el conteo del modelo

total_params = sum(p.numel() for p in modelo.parameters())
#print("Numero de parametros a entrenar: ", total_params) -> sirve para saber el total de parametros a entrenar

#extraer una imagen y su categoria del set de entrenamiento
img, lbl = train[200]
#redefinimos el lbl a tensor
lbl = torch.tensor(lbl).reshape(1)

img, lbl = img.to(device), lbl.to(device)

logits = modelo(img) #logits eran los 10 numeros de salida
#print(logits)

#como saber la categoria predicha
y_pred = logits.argmax(1) #que busque la posicion del maximo logit

# Mostremos la imagen original
plt.imshow(img.cpu().squeeze(), cmap="gray");

print(f"Logits {logits}")
print(f"Categoria predicha: {y_pred[0]}")
print(f"Categoria real: {lbl[0]}")

#En este punto se termina el forward propagation (adelante) y tiene que hacer el backprop asi obtengo su gradiente (error entre la prediccion y el real)

# 1 - Calculamos perdida y optimizamos el modelo
fn_perdida = nn.CrossEntropyLoss() #entropia que mide el error
optimizador = torch.optim.SGD(modelo.parameters(), lr = 0.2) #SGD es el gradiente descentente y le presentamos los parametros y el learning rate(tasa de aprendizaje)

#2 - Cacular la perdida como numero
loss = fn_perdida(logits, lbl)
print(loss)

#3 Calcular los gradientes de la perdida, es decir, cuanto cambia la pedida por iteracion
loss.backward()

#4 - Actualice los parametros
optimizador.step() #toma los gradientes y los actualiza pero no los elimina
optimizador.zero_grad()#borra los gradientes anteriores

#como tenemos muchos datos, podemos llegar a tener problemas con la memoria RAM, por eso dividimos los datos en sets mas pequeños llamados batches

#definimos el tamaño del batch
TAM_LOTE = 1000

#crear los "dataloaders" para los sets de entrenamiento y validacion
train_loader = DataLoader(
  dataset=train,
  batch_size= TAM_LOTE,
  shuffle=True
)
val_loader = DataLoader(
  dataset=val,
  batch_size= TAM_LOTE,
  shuffle=False
)

#Hiperparametros
TASA_APRENDIZAJE = 0.1 #learning rate
EPOCHS = 10 #numero de iteraciones

#redefinimos la optimizacion
fn_perdida = nn.CrossEntropyLoss()
optimizador = torch.optim.SGD(modelo.parameters(), lr= TASA_APRENDIZAJE)

def train_loop(dataloader, model, loss_fn, optimizer):
  #Cantidad
  train_size = len(dataloader.dataset)
  nlotes = len(dataloader)
  
  #indicarle a pytorch 
  model.train()
  
  #inicializar acumuladores perdida y exactitud
  perdida_train, exactitud = 0, 0
  
  #Presentar los datos al modelo por lotes
  for nlote, (x, y) in enumerate(dataloader): #x almacena las imagenes e y almacena su categoria
    #mover x e y a la gpu
    x, y = x.to(device), y.to(device)
    
    #forward propagation
    logits = model(x)
    
    #backpropagation
    loss = loss_fn(logits, y)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    
    #acumular valores de perdida y exactitud
    #perdida_train <-perdida_train + perdida_actual
    #exactitud <- exactitud + numero_aciertos_actuales
    perdida_train += loss.item() #extraigo el numero de perdida con .item()
    exactitud += (logits.argmax(1) == y).type(torch.float).sum().item() #devuelve 1 si encontró igual o 0 si no lo hizo, por eso lo cambiamos a float para que de por ejemplo 84.35
    
    #imprimimos en pantalla la evolucion
    if nlote % 10 == 0: #por cada 10 lotes
      #obtener el valor de la perdida y la cant de datos procesados
      ndatos = nlote*TAM_LOTE
      
      #imprimir
      print(f"\nPerdida: {loss.item():>7f} [{ndatos:>5d}/{train_size:>5d}]")
  #al terminar de presentar los datos, promediar perdida y exactitud
  perdida_train /= nlotes
  exactitud /= train_size
  
  #imprimir informacion
  print(f"\tExactitud/perdida promedio:")
  print(f"\t\tEntrenamiento: {(100*exactitud):>0.1f}% / {perdida_train:>8f}")
  
def val_loop(dataloader, model, loss_fn):
  #cantidad de datos de validacion y cantidad de lotes
  val_size = len(dataloader.dataset)
  nlotes = len(dataloader)
  
  model.eval()
  
  perdida_val, exactitud = 0, 0
  
  #evaluar
  with torch.no_grad():
    for x, y in dataloader:
      x, y = x.to(device), y.to(device)
      
      logits = model(x)
      
      perdida_val += loss_fn(logits, y).item()
      exactitud += (logits.argmax(1) == y).type(torch.float).sum().item()
  perdida_val /= nlotes
  exactitud /= val_size
  
  print(f"\t\tValidacion: {(100*exactitud):>0.1f}%/ {perdida_val:>8f} \n")

#combinamos todo
for t in range(EPOCHS):
  print(f"Iteracion: {t+1}/{EPOCHS}\n-----------------------")
  #entrenar
  train_loop(train_loader, modelo, fn_perdida, optimizador)
  #validar
  val_loop(val_loader, modelo, fn_perdida)
print("Modelo entrenado")

#probamos predicciones con el modelo ya entrenado
def predecir(model, img):
    # agregar dimensión batch
    img = img.unsqueeze(0).to(device)
    
    # generar predicción
    with torch.no_grad():  # desactiva el cálculo de gradientes
        logits = model(img)
        y_pred = logits.argmax(1).item()
    
    # mostrar imagen y categoría
    plt.imshow(img.cpu().squeeze(), cmap="gray")
    plt.title(f"Categoría predicha: {y_pred}")
    plt.axis("off")
    plt.show()

#tomar imagen del set de prueba
img, lbl = test[211]

#generamos prediccion
predecir(modelo, img)