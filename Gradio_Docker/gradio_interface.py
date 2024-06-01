import torch
import gradio as gr
import pandas as pd
import numpy as np
from PIL import Image
import my_model
import io

# Wczytaj model
# torch.jit.pickle.ignore_pickle_module()
cnn_model = my_model.DiceClassifier()
cnn_model.load_state_dict(torch.load('dice_classifier.pth', map_location=torch.device('cpu')))
cnn_model.eval()

# Definiowanie transformacji obrazu
# preprocess = transforms.Compose([
#     transforms.Resize(256),
#     transforms.CenterCrop(224),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
# ])


def classify(csv_string):
    # Wczytaj dane z pliku CSV
    data = pd.read_csv(io.StringIO(csv_string), header=None)
    # Konwertuj dane do numpy array
    data_array = data.to_numpy().astype(np.float32) / 255
    # Reshape do rozmiaru 28x28
    reshaped_data = data_array.reshape(28, 28)
    # Konwertuj do tensora
    tensor = torch.tensor(reshaped_data).unsqueeze(0).unsqueeze(0)
    # Przekaż dane przez model
    with torch.no_grad():
        output = cnn_model(tensor)
    # Pobierz przewidywaną klasę
    _, predicted = torch.max(output, 1)
    predicted_class = predicted.item()
    # Utwórz obraz z reshaped_data
    image = Image.fromarray((reshaped_data * 255).astype(np.uint8), mode='L')
    return image, predicted_class + 1


# Tworzenie interfejsu Gradio
# Zdefiniuj komponenty wejścia i wyjścia Gradio
input_component = gr.Textbox(lines=10, label="CSV Data")
output_components = [gr.Image(type="pil", label="28x28 Image"), gr.Textbox(label="Predicted Class")]

# Stwórz interfejs
iface = gr.Interface(fn=classify, 
    inputs=input_component, 
    outputs=output_components, 
    title="CSV Classifier",
    description="Upload a CSV string with values separated by commas. The model will classify the reshaped 28x28 image."
    )

# Uruchom interfejs
iface.launch()
