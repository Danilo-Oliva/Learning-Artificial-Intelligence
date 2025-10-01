import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

#download dataset mnist
train_data = datasets.MNIST(root = "data", train = True, download=True, transform=ToTensor())
test_data = datasets.MNIST(root = "data", train=False, download=True, transform=ToTensor())

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64)

#def red densa
class RedDensa(nn.Module):
  def __init__(self):
    super().__init__()
    self.red = nn.Sequential(
      nn.Flatten(),
      nn.Linear(28*28, 128),
      nn.ReLU(),
      nn.Linear(128, 64),
      nn.ReLU(),
      nn.Linear(64, 10)
    )
  def forward(self, x):
    return self.red(x)

#crear modelo, perdida y optimizador
model = RedDensa()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)

#funcion de entrenamiento y prueba
def entrenar(dataloader, model, loss_fn, optimizer):
  size = len(dataloader.dataset)
  model.train()
  total_loss = 0
  for batch, (x, y) in enumerate(dataloader):
    #forward
    pred = model(x)
    loss = loss_fn(pred, y)
    total_loss += loss.item()
    
    #backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

  print(f"Loss promedio: {total_loss/len(dataloader):4f}") 

def probar(dataloader, model, loss_fn):
  size = len(dataloader.dataset)
  num_batches = len(dataloader)
  model.eval()
  test_loss, correct = 0, 0
  
  with torch.no_grad():
    for x, y in dataloader:
      pred = model(x)
      test_loss += loss_fn(pred, y).item()
      correct += (pred.argmax(1) == y).type(torch.float).sum().item()
      
  test_loss /= num_batches
  accuaracy = correct / size
  print(f"Precision: {(100*accuaracy):.1f}% | Loss: {test_loss:.4f}")
  
epochs = 10
for t in range(epochs):
  print(f"Epoca {t +1}\n--------------------")
  entrenar(train_loader, model, criterion, optimizer)
  probar(test_loader, model, criterion)