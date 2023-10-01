FROM python:3.9

# Path: /app
WORKDIR /app

# Path: /app
RUN apt-get update

# Path: /app
RUN apt-get install -y libgtk2.0-dev pkg-config libgl1-mesa-dev libglib2.0-0

# Path: /app/requirements.txt
COPY requirements.txt .

RUN pip install --upgrade pip setuptools wheel

# Path: /app
RUN pip install -r requirements.txt

# Path: /app
CMD ["python3", "train_model.py"]
