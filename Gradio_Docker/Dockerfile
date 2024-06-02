FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime
# Skopiuj plik requirements.txt do obrazu
COPY requirements.txt /app/requirements.txt
# Przejdź do katalogu roboczego
WORKDIR /app
# Zainstaluj zależności
RUN pip install --no-cache-dir -r requirements.txt
# Skopiuj kod aplikacji do obrazu
COPY gradio_interface.py /app/
# COPY dice_transfer_classifier.pth /app/dice_classifier.pth
COPY dice_classifier.pth /app/dice_classifier.pth
COPY my_model.py /app/

# Uruchom aplikację
CMD ["python", "gradio_interface.py"]