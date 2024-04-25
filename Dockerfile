FROM ubuntu:latest
LABEL authors="milosz"

RUN apt-get update && apt-get install -y python3 python3-pip sudo
RUN useradd -m jamal

WORKDIR /app
COPY requirements.txt ./
RUN pip3 install --no-cache-dir -r requirements.txt
COPY . .

RUN chown -R jamal:jamal /app
USER jamal

CMD ["python3", "dice_train.py"]