# LangGraph Stateless Chat (Python Test Task)

Минимальный чат-бот на LangGraph + Google Gemini с поддержкой инструмента `get_current_time`.

## Файлы проекта

- `main.py` — основной код чат-бота
- `requirements.txt` — зависимости
- `.env` — ваш API-ключ Google

## Быстрый старт

1. Клонируйте репозиторий и перейдите в папку проекта:
    ```bash
    git clone <your-repo-url>
    cd <repo-folder>
    ```

2. Получите API-ключ Google (AI Studio) и создайте файл `.env` в корне проекта:
    ```
    GOOGLE_API_KEY="ваш_ключ"
    ```

3. Создайте и активируйте виртуальное окружение:
    ```bash
    python -m venv .venv
    source .venv/bin/activate
    ```

4. Установите зависимости:
    ```bash
    pip install -r requirements.txt
    ```

5. Запустите бота:
    ```bash
    python main.py
    ```

## Как это работает

- Бот поддерживает диалог без хранения состояния между сессиями.
- Для ответа на вопросы о текущем времени использует встроенный инструмент.
