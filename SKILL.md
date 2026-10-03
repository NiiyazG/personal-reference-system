---
name: personal-reference-system
description: Развернуть с нуля локальную справочно-информационную систему personal-reference-system (Personal Reference) — неизменяемое хранилище документов с полнотекстовым поиском, опциональным OCR (Windows) и семантическим поиском (bge-m3). Использовать при установке, запуске, импорте документов, настройке конфигурации или диагностике этой системы.
---

# Personal Reference System — развёртывание с нуля

Навык описывает, как поставить и запустить локальную справочную систему из
репозитория `NiiyazG/personal-reference-system`. Система индексирует документы и
ищет по ним **локально, без сети**; OCR и векторный поиск — опциональны.

## Когда применять

- Нужно развернуть систему на новой машине.
- Нужно добавить документ в корпус или найти по нему.
- Нужно настроить путь хранилища, квоты, OCR или модель векторов.
- Система не работает: не создаётся база, поиск падает, OCR/векторы недоступны.

## Ключевые факты

- Пакет запускается как `python -m reference_system.cli`.
- Хранилище — один каталог (по умолчанию `./data`); `init` создаёт структуру, манифест
  и **пустую** базу со свободным `knowledge_base_id`.
- Поиск по пустой базе не падает — возвращает пустой список.
- Конфигурация: файл `reference.config.json` (`--config` / `REFERENCE_CONFIG`),
  переменные окружения `REFERENCE_*`, флаги CLI. Пример — `reference.config.example.json`.
- OCR и embeddings по умолчанию `none`: система ставится и ищет без скачивания модели.
- Сеть системой не используется; модель векторов скачивается вручную.

## Шаги

### 1. Установка

```bash
git clone https://github.com/NiiyazG/personal-reference-system.git
cd personal-reference-system
python -m venv .venv
# Windows:  .venv\Scripts\activate
# Linux/macOS:  source .venv/bin/activate
pip install -r requirements.txt
```

Проверка:

```bash
python -m tools.security_scan
python -m unittest discover -s tests -p "test_*.py"
```

### 2. Инициализация хранилища

```bash
python -m reference_system.cli --root data init
```

Создаётся пустая база. Проверить состояние и убедиться, что поиск не падает:

```bash
python -m reference_system.cli --root data status
python -m reference_system.cli --root data search "проверка"
```

### 3. Импорт и поиск

```bash
python -m reference_system.cli --root data add "/путь/к/файлу.pdf" --title "Название"
python -m reference_system.cli --root data search "насос вибрирует"
python -m reference_system.cli --root data search "P-101" --exact
python -m reference_system.cli --root data integrity-check
```

### 4. Настройка (по желанию)

```bash
cp reference.config.example.json reference.config.json
```

Настраиваются `root`, `mode`, квоты (`total_gib`/`live_gib`/`backup_gib`/
`temporary_gib`/`minimum_free_disk_gib`), `ocr_engine`, `ocr_language`,
`embedding_model`, `allowed_profile`. Либо переменные `REFERENCE_*`, либо флаги CLI.

### 5. OCR (опционально, Windows)

```bash
pip install winsdk
python -m reference_system.cli --root data add scan.pdf --ocr windows
```

### 6. Семантический поиск (опционально)

```bash
pip install huggingface_hub
huggingface-cli download Xenova/bge-m3 --local-dir "$HOME/.cache/huggingface/hub/models--Xenova--bge-m3"
pip install onnxruntime tokenizers numpy
python -m reference_system.cli --root data embed --embeddings bge-m3
python -m reference_system.cli --root data search "смысл запроса" --semantic --embeddings bge-m3
```

Модель в репозиторий не входит; без неё `embed` и `search --semantic` отказывают.

## Диагностика

| Симптом | Причина / действие |
|---|---|
| `search` падает с `OperationalError` | проверьте `integrity-check`; пустой запрос из одних метасимволов возвращает `[]`, а не ошибку |
| `embed`/`search --semantic` отказывают | модель не найдена — скачайте `bge-m3` в кэш HF и установите `onnxruntime`, `tokenizers`, `numpy` |
| OCR недоступен | установлен ли `winsdk`; язык задан через `ocr_language` |
| «сканированный PDF отклонён» | подключите OCR: `--ocr windows` |
| Импорт блокируется по квоте | мало свободного места (< `minimum_free_disk_gib`) или превышен `live_gib` |

## Границы

- Не публикуйте каталог `data/` — это личный корпус, он исключён в `.gitignore`.
- Удаление необратимо (резервной копии нет), требует `--confirm`.
- Не меняйте настройки Hermes и не трогайте чужие профили.
