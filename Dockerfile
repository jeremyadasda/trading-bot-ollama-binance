FROM python:3.12
WORKDIR /app

# Upgrade build tools
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
# Comando para que el bot se quede ejecutando
CMD ["python", "bot_logic/main.py"]