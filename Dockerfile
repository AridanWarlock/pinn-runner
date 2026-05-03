FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /app

ENV PYTHONPYCACHEPREFIX=/tmp/pycache


# 1. Только необходимые системные библиотеки
RUN apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0 git && rm -rf /var/lib/apt/lists/*

# 2. Устанавливаем зависимости по одной, чтобы не перегружать память
# Сначала критически важный NumPy нужной версии
RUN pip install --no-cache-dir "numpy<2.0.0"

# 3. Копируем файл зависимостей
COPY requirements.txt .

# 4. Устанавливаем остальное, игнорируя попытки обновить numpy
RUN pip install --no-cache-dir -r requirements.txt || true
RUN pip install --no-cache-dir "numpy<2.0.0"

COPY . .
RUN pip install -e .

CMD ["/bin/bash"]
