# Sidekick-VLM: On-Device Vision-Language Assistant

Проект по оптимизации и переносу мультимодальной языковой модели (Vision-Language Model, VLM) на устройства с ограниченными ресурсами (Edge AI / Mobile).

**Главная цель** — создание полностью локального интеллектуального ассистента, способного анализировать видеопоток / кадры с камеры в реальном времени, отвечать на вопросы о пространственном окружении и помогать человеку в выполнении узкоспециализированной деятельности.

---

## Архитектура и фазы проекта

Проект состоит из двух ключевых этапов:

### Фаза 1: Оптимизация инференса и бенчмаркинг (Inference & Quantization)
- Снятие baseline-метрик базовой модели на ПК
- Квантизация модели (**INT4 / GGUF / AWQ**) и адаптация под мобильные фреймворки (**llama.cpp / ExecuTorch / MLC-LLM**)
- Замеры скорости (**TTFT**, **TPS**), потребления памяти (**RAM / VRAM**) и качества (**VQA Accuracy**)

### Фаза 2: Дообучение под конкретную деятельность (Fine-Tuning & Domain Assistant)
- Выбор предметной области (например, ассистент по сборке/пайке, ремонту или настольным играм)
- Сбор и разметка мультимодального датасета (*Image-Instruction Pairs*)
- Дообучение адаптеров через **QLoRA / LoRA** с сохранением общей эрудиции модели
- Создание клиентского приложения с фильтрацией ключевых кадров (Keyframe Selection)

---

## Результаты экспериментов

### Baseline 1.0: Qwen2-VL-2B-Instruct (GGUF INT4 / CPU)

| Параметр                  | Значение                          |
|---------------------------|-----------------------------------|
| **Дата**                  | 6 Октября 2026                      |
| **Модель**                | `Qwen2-VL-2B-Instruct` (Q4_K_M)   |
| **Инференс-движок**       | `llama.cpp` (HTTP Streaming Server) |
| **Аппаратное обеспечение**| CPU (AVX2), 8 GB RAM (без CUDA)   |

| Метрика                        | Значение     | Описание                                              |
|--------------------------------|--------------|-------------------------------------------------------|
| **TTFT (Time-To-First-Token)** | **0.470 s**  | Время обработки кадра и промпта (~700 токенов)        |
| **Generation Speed (TPS)**     | **7.67 tok/s** | Чистая скорость генерации текста на CPU             |
| **Total Inference Time**       | **33.85 s**  | Вывод 256 токенов                                     |
| **Качество (VQA)**             | **5 / 5**    | Точно распознаны технические схемы (2D/3D свёртки)    |

---

## Быстрый запуск

### 1. Подготовка окружения

```bash
# Клонирование репозитория
git clone https://github.com/your-username/Sidekick-VLM.git
cd Sidekick-VLM

# Создание и активация Conda-окружения
conda create -n sidekick python=3.10 -y
conda activate sidekick

# Установка зависимостей
pip install llama-cpp-python requests psutil pillow
```

### 2. Загрузка квантованной модели (GGUF INT4)

Для загрузки модели можно использовать зеркало Hugging Face.

**Linux / macOS:**

```bash
export HF_ENDPOINT="https://hf-mirror.com"

huggingface-cli download ggml-org/Qwen2-VL-2B-Instruct-GGUF \
  Qwen2-VL-2B-Instruct-Q4_K_M.gguf \
  --local-dir ./models

huggingface-cli download ggml-org/Qwen2-VL-2B-Instruct-GGUF \
  mmproj-Qwen2-VL-2B-Instruct-Q8_0.gguf \
  --local-dir ./models
```

**Windows (PowerShell):**

```powershell
$env:HF_ENDPOINT = "https://hf-mirror.com"

huggingface-cli download ggml-org/Qwen2-VL-2B-Instruct-GGUF `
  Qwen2-VL-2B-Instruct-Q4_K_M.gguf `
  --local-dir ./models

huggingface-cli download ggml-org/Qwen2-VL-2B-Instruct-GGUF `
  mmproj-Qwen2-VL-2B-Instruct-Q8_0.gguf `
  --local-dir ./models
```

После загрузки структура проекта должна выглядеть примерно так:

```text
models/
├── Qwen2-VL-2B-Instruct-Q4_K_M.gguf
└── mmproj-Qwen2-VL-2B-Instruct-Q8_0.gguf
```

### 3. Запуск инференса

Для мультимодального инференса используется локальный сервер `llama.cpp`.

**Linux / macOS:**

```bash
./llama-server \
  -m ./models/Qwen2-VL-2B-Instruct-Q4_K_M.gguf \
  --mmproj ./models/mmproj-Qwen2-VL-2B-Instruct-Q8_0.gguf \
  --port 8080 \
  -t 4
```

**Windows (PowerShell):**

```powershell
llama-server `
  -m ".\models\Qwen2-VL-2B-Instruct-Q4_K_M.gguf" `
  --mmproj ".\models\mmproj-Qwen2-VL-2B-Instruct-Q8_0.gguf" `
  --port 8080 `
  -t 4
```

Оставьте сервер запущенным и откройте **второй терминал**.

Активируйте Conda-окружение:

```powershell
conda activate sidekick
```

Затем запустите benchmark:

```powershell
python Stage-1/run_baseline.py
```

Benchmark отправляет изображение и запрос в локальный `llama-server` и измеряет время инференса, количество сгенерированных токенов, скорость генерации и другие показатели.

---

## Структура репозитория

```
Sidekick-VLM/
├── data/                     # Тестовые изображения и датасеты
│   └── convs.png
├── models/                   # Локальные веса моделей и mmproj (.gguf, .bin)
├── Stage-1/                  # Фаза 1: Бенчмаркинг и квантизация
│   ├── run_baseline.py       # Скрипт замера TTFT / TPS со стримингом
│   └── benchmark_set.json    # Калибровочный набор VQA-вопросов
├── Stage-2/                  # Фаза 2: Дообучение и адаптеры (LoRA)
│   ├── dataset/              # Инструкционный датасет
│   └── train_qlora.py        # Скрипт Fine-Tuning
├── README.md
└── requirements.txt
```

---

## Дорожная карта

- [x] Настройка локального окружения и бенчмарк-пайплайна
- [x] Скачивание и запуск Qwen2-VL-2B в формате GGUF INT4
- [x] Замер TTFT и Generation TPS в режиме стриминга
- [ ] Сбор калибровочного датасета `benchmark_set.json` (10+ сценариев VQA/OCR)
- [ ] Оптимизация под мобильные устройства (экспорт в ExecuTorch / Android APK)
- [ ] Сбор датасета и QLoRA-дообучение под предметную деятельность
- [ ] Публикация результатов и научной статьи / отчёта