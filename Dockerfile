# Dockerfile para o servidor de missões Python
FROM python:3.9-slim

# Definir diretório de trabalho
WORKDIR /app

# Instalar dependências do sistema
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copiar requirements primeiro (para cache do Docker)
COPY requirements.txt .

# Instalar dependências Python
RUN pip install --no-cache-dir -r requirements.txt

# Instalar dependências adicionais para o servidor
RUN pip install --no-cache-dir \
    flask \
    flask-cors \
    flask-socketio \
    requests

# Copiar código fonte
COPY . .

# Expor porta
EXPOSE 5000

# Comando de inicialização
CMD ["python", "mission_server.py"]
