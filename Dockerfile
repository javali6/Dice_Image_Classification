FROM pytorch/pytorch:latest

WORKDIR /app

RUN pip install numpy matplotlib scikit-learn seaborn neptune-client

COPY dice_train.py /app/dice_train.py
COPY utilities.py /app/utilities.py
COPY data/dice.csv /app/data/dice.csv

CMD ["python", "dice_train.py"]
