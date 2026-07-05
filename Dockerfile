# Dockerfile
FROM nvidia/cuda:13.3.0-cudnn-runtime-ubuntu24.04

# Устанавливаем системные зависимости
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3.12-venv \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Устанавливаем рабочую директорию
WORKDIR /app

# Создаём виртуальное окружение
RUN python3 -m venv /venv
ENV PATH="/venv/bin:$PATH"

# Копируем зависимости и устанавливаем
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.lock

# Копируем код приложения и лицензию
COPY examples ./examples
COPY app.py .
COPY LICENSE .
COPY THIRD_PARTY_LICENSES .

EXPOSE 7860

# Запускаем приложение
CMD ["python3", "app.py"]