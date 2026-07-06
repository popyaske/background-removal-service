# Background Removal Tool (BiRefNet ONNX)

Сервис для удаления фона с изображений на основе модели **BiRefNet** в формате ONNX. Предоставляет веб-интерфейс на базе Gradio с тремя режимами работы: загрузка файла, ссылка на изображение и экспорт в PNG.

## 📋 Оглавление

- [Особенности](#особенности)
- [Структура проекта](#структура-проекта)
- [Требования](#требования)
- [Установка и запуск](#установка-и-запуск)
  - [Сборка Docker-образа](#сборка-docker-образа)
  - [Запуск через docker-compose](#запуск-через-docker-compose)
  - [Запуск вручную](#запуск-вручную)
- [Использование](#использование)
- [Конфигурация](#конфигурация)
- [Лицензии](#лицензии)
- [Благодарности](#благодарности)

---

## 🚀 Особенности

- **Удаление фона** с высоким качеством (BiRefNet, SOTA на DIS5K/COD10K/HRSOD).
- **Поддержка GPU** через ONNX Runtime (CUDA) – ускорение инференса.
- **Три режима ввода**:
  - Загрузка изображения (через интерфейс)
  - URL изображения
  - Обработка файла с сохранением PNG
- **Примеры** для быстрого старта.
- **Docker-контейнеризация** для простого развертывания.
- **Лицензия MIT** для модели (исходный код обёртки – коммерческая лицензия).

---

## 📁 Структура проекта

```
background_removal_service/
├── app.py                      # Основной код Gradio-приложения
├── Dockerfile                  # Инструкция сборки Docker-образа
├── docker-compose.yml          # Оркестрация контейнера
├── requirements.lock           # Зафиксированные версии
├── LICENSE                     # Коммерческая лицензия для обёртки
├── THIRD_PARTY_LICENSES        # Лицензии сторонних компонентов
├── README.md                   # Этот файл
├── models/                     # Папка с ONNX-моделью (монтируется)
│   └── model_birefnet_fp16.onnx
└── examples/                   # Примеры изображений
    └── butterfly.jpg
```

---

## ⚙️ Требования

- **Docker** (20.10+)
- **Docker Compose** (2.0+)
- **NVIDIA GPU** (для ускорения) с драйверами и **NVIDIA Container Toolkit** (установлен на хосте)
- Для запуска без GPU – достаточно CPU (но будет медленнее)

---

## 🛠️ Установка и запуск

### 1. Сборка Docker-образа

```bash
docker-compose build
```

Или напрямую:

```bash
docker build -t birefnet-service:latest .
```

### 2. Запуск сервиса

```bash
docker-compose up -d
```

После запуска сервис будет доступен по адресу: [http://localhost:7860](http://localhost:7860)

### 3. Проверка статуса

```bash
docker-compose logs -f   # Просмотр логов
docker-compose ps        # Статус контейнера
```

### 4. Остановка

```bash
docker-compose down
```

---

## 🖥️ Использование

Откройте браузер и перейдите на `http://localhost:7860`. Вы увидите три вкладки:

1. **Image Upload** – загрузите изображение с компьютера. После обработки появится результат с прозрачным фоном.
2. **URL Input** – вставьте ссылку на изображение в интернете.
3. **File Output** – загрузите файл, сервис вернёт готовый PNG с прозрачностью (скачивается автоматически).

В каждой вкладке есть примеры для быстрого тестирования.

---

## 🔧 Конфигурация

### Ограничение памяти GPU

В `app.py` настроен `gpu_mem_limit = 5 ГБ`. При необходимости измените этот параметр:

```python
providers = [
    ('CUDAExecutionProvider', {
        'gpu_mem_limit': 8 * 1024 * 1024 * 1024,  # 8 ГБ
        # ...
    }),
    'CPUExecutionProvider'
]
```

### Порт

По умолчанию используется порт `7860`. Измените в `docker-compose.yml` при необходимости:

```yaml
ports:
  - "8080:7860"   # будет доступно на порту 8080
```

---

## 📜 Лицензии

### Код обёртки (this repository)

Copyright (c) 2026 triumphgroup  
Все права защищены. Использование, копирование, модификация или распространение без письменного разрешения запрещены.  
Коммерческая лицензия доступна по запросу: legal@yourcompany.com

### Модель BiRefNet

Модель распространяется под **лицензией MIT** (автор: ZhengPeng).  
Подробности см. в файле [THIRD_PARTY_LICENSES](./THIRD_PARTY_LICENSES) и на [официальном репозитории](https://github.com/ZhengPeng7/BiRefNet).

### Используемые Python-пакеты

Все зависимости имеют собственные лицензии (MIT, BSD, Apache и др.). Полный перечень с версиями указан в `requirements.lock`.

---

## 🙏 Благодарности

- [BiRefNet](https://github.com/ZhengPeng7/BiRefNet) – оригинальная модель (авторы: Peng Zheng, Dehong Gao, Deng-Ping Fan, Li Liu, Jorma Laaksonen, Wanli Ouyang, Nicu Sebe)
- [ONNX Community](https://huggingface.co/onnx-community) – конвертация в ONNX
- [Gradio](https://gradio.app) – создание веб-интерфейса
- [ONNX Runtime](https://onnxruntime.ai) – высокопроизводительный инференс

---

## ❓ Часто задаваемые вопросы

**1. Можно ли запустить без GPU?**  
Да, замените в `requirements.lock` `onnxruntime-gpu` на `onnxruntime` (CPU-версия) и удалите секцию `deploy` из `docker-compose.yml`. Инференс будет выполняться на CPU, но медленнее.

**2. Почему используется так много видеопамяти?**  
ONNX Runtime по умолчанию резервирует память для ускорения. В `app.py` установлен лимит 5 ГБ – этого достаточно для модели. При необходимости увеличить `gpu_mem_limit`.

**3. Где взять другую ONNX-модель BiRefNet?**  
На Hugging Face доступны версии FP16 и FP32:  
`https://huggingface.co/onnx-community/BiRefNet-ONNX/tree/main`

---

## 🧪 Разработка и тестирование

Для локальной разработки (без Docker):

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python app.py
```

Убедитесь, что модель лежит по пути `models/model_birefnet_fp16.onnx`.

---

## 📞 Контакты

По вопросам коммерческого использования обращайтесь:  
**Email:** legal@yourcompany.com

---

**Copyright © 2026 triumphgroup. All rights reserved.**
