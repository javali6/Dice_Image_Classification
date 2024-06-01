FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY dice_train.py dice_train_transfer.py utilities.py my_model.py /app/
COPY data/dice.csv /app/data/

CMD ["python", "dice_train.py"]
# CMD ["sh", "-c", "python dice_train_transfer.py"]