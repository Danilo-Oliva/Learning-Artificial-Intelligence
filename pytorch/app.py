import torch
from torch import nn
import torchvision.transforms as transforms
from PIL import Image
import streamlit as st
from streamlit_drawable_canvas import  st_canvas

class RedConvolucional(nn.Module):
  def __init__(self):
    super().__init__()
    self.red = nn.Sequential(
      nn.Conv2d(1, 8, kernel_size=3, padding=1),
      nn.ReLU(),
      nn.MaxPool2d(2, 2),
      nn.Conv2d(8, 16, kernel_size=3, padding=1),
      nn.ReLU(),
      nn.MaxPool2d(2, 2),
      nn.Flatten(),
      nn.Linear(16 * 7 * 7, 64),
      nn.ReLU(),
      nn.Linear(64, 10)
    )
    
  def forward(self, x):
    return self.red(x)
  
model = RedConvolucional()
model.load_state_dict(torch.load("modelo_mnist_cnn.pth"))
model.eval()

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
])

st.title("Reconocedor de Dígitos MNIST")
st.write("Dibuja un dígito del 0 al 9 en el recuadro y presiona 'Predecir'.")

canvas_result = st_canvas(
    fill_color="rgba(0, 0, 0, 0)",  # Fondo transparente
    stroke_width=15,               # Grosor del pincel
    stroke_color="#FFFFFF",        # Color del pincel (blanco)
    background_color="#000000",    # Color de fondo (negro)
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas",
)

if st.button("Predecir"):
    if canvas_result.image_data is not None:
        # Convertimos la imagen del lienzo a un formato que PyTorch entiende
        # El canvas devuelve una imagen RGBA, la convertimos a RGB
        img_pil = Image.fromarray(canvas_result.image_data.astype('uint8'), 'RGBA').convert('RGB')
        
        # Aplicamos las transformaciones
        img_tensor = transform(img_pil)
        
        # El modelo espera un "batch" de imágenes, así que añadimos una dimensión extra
        img_tensor = img_tensor.unsqueeze(0)

        # Hacemos la predicción
        with torch.no_grad():
            pred = model(img_tensor)
            predicted_digit = torch.argmax(pred, 1).item()
            
        st.success(f"## ¡Creo que es un: {predicted_digit}!", icon="🎉")
    else:
        st.warning("Por favor, dibuja un número primero.", icon="⚠️")