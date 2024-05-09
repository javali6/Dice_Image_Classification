FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime

WORKDIR /app

RUN pip install --no-cache-dir numpy matplotlib scikit-learn seaborn neptune-client

COPY dice_train.py /app/dice_train.py
COPY utilities.py /app/utilities.py
COPY data/dice.csv /app/data/dice.csv

CMD ["python", "dice_train.py"]
