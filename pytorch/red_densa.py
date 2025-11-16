import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

# Descarga del dataset MNIST (sin cambios)
train_data = datasets.MNIST(root="data", train=True, download=True, transform=ToTensor())
test_data = datasets.MNIST(root="data", train=False, download=True, transform=ToTensor())

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64)

# --- CAMBIO 1: Se reemplaza la Red Densa por la Red Convolucional ---
class RedConvolucional(nn.Module):
    def __init__(self):
        super().__init__()
        self.red = nn.Sequential(
            # Capa 1: Convolución + Activación + Pooling
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Capa 2: Convolución + Activación + Pooling
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Aplanamos para las capas densas finales
            nn.Flatten(),
            
            # Capas densas para la clasificación
            nn.Linear(16 * 7 * 7, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )
    
    def forward(self, x):
        return self.red(x)

# --- CAMBIO 2: Se crea una instancia del nuevo modelo ---
model = RedConvolucional() 

# Crear pérdida y optimizador (sin cambios)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Función de entrenamiento (sin cambios)
def entrenar(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    total_loss = 0
    for batch, (x, y) in enumerate(dataloader):
        # Forward
        pred = model(x)
        loss = loss_fn(pred, y)
        total_loss += loss.item()
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Loss promedio: {total_loss / len(dataloader):.4f}")

# Función de prueba (sin cambios)
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
    accuracy = correct / size
    print(f"Precisión: {(100 * accuracy):.1f}% | Loss: {test_loss:.4f}")
    
# Bucle de entrenamiento (sin cambios)
epochs = 10
for t in range(epochs):
    print(f"Época {t + 1}\n--------------------")
    entrenar(train_loader, model, criterion, optimizer)
    probar(test_loader, model, criterion)

# Guardado del modelo (sin cambios, pero ahora guarda el modelo correcto)
torch.save(model.state_dict(), "modelo_mnist_cnn.pth")
print("\n¡Modelo Convolucional guardado como 'modelo_mnist_cnn.pth'!")