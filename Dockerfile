FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime

WORKDIR /app

COPY requirements.txt /app/requirements.txtgit s
RUN pip install --no-cache-dir -r requirements.txt


COPY dice_train.py /app/dice_train.py
COPY utilities.py /app/utilities.py
COPY data/dice.csv /app/data/dice.csv

CMD ["python", "dice_train.py"]
