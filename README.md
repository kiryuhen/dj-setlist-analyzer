# DJ Setlist Analyzer

DJ Setlist Analyzer — это профессиональный инструмент для анализа и нормализации громкости аудиофайлов, разработанный специально для диджеев. Приложение позволяет проанализировать все треки в вашем сетлисте, нормализовать их громкость с учетом жанра, а также получить рекомендации по построению оптимального сетлиста.

## Основные возможности

- **Анализ аудио**: извлечение BPM, тональности, громкости, жанра и других характеристик
- **Умная нормализация**: настройка громкости с учетом жанра музыки
- **Многопоточная обработка**: быстрая работа с большими коллекциями
- **Кэширование результатов**: ускорение повторного анализа
- **Детальные отчеты**: визуализация характеристик треков
- **Рекомендации по сетлисту**: оптимальный порядок треков и советы по микшированию

## Установка

### Требования

- Python 3.8 или выше
- Зависимости из файла `requirements.txt`

### Шаги установки

1. Клонируйте репозиторий:
```bash
git clone https://github.com/username/dj-setlist-analyzer.git
cd dj-setlist-analyzer
```

2. Установите зависимости:
```bash
pip install -r requirements.txt
```

## Использование

### Подготовка

1. Поместите аудиофайлы для анализа в папку `setlist/`
2. Убедитесь, что у вас есть права на чтение и запись в директории проекта

### Командная строка

#### Анализ аудиофайлов

```bash
python cli.py analyze --folder path/to/files --report
```

#### Нормализация громкости

```bash
python cli.py normalize --target-lufs -14 --format mp3
```

#### Пакетная обработка (анализ + нормализация)

```bash
python cli.py batch --folder path/to/files --target-lufs -14 --format wav --report
```

#### Информация о файле

```bash
python cli.py info --file path/to/track.mp3
```

#### Управление кэшем

```bash
python cli.py cache --info
python cli.py cache --clear
```

### Программный интерфейс

Вы также можете использовать библиотеку программно:

```python
from src.analyzer import AudioAnalyzer
from src.normalizer import AudioNormalizer

# Анализ файлов
analyzer = AudioAnalyzer("path/to/files")
results = analyzer.analyze_all()

# Нормализация громкости
normalizer = AudioNormalizer(analyzer)
normalized = normalizer.normalize_all(target_lufs=-14, output_format="wav")
```

## Настройка

Основные настройки находятся в файле `config.py`. Вы можете изменить:

- Целевые уровни громкости для разных жанров
- Параметры кэширования
- Количество потоков для параллельной обработки
- Поддерживаемые форматы файлов

## Структура проекта

```
dj-setlist-analyzer/
│
├── setlist/                 # Папка для аудиофайлов
│
├── src/                     # Исходный код
│   ├── __init__.py          # Инициализация пакета
│   ├── analyzer.py          # Анализ аудио параметров
│   ├── normalizer.py        # Нормализация громкости
│   ├── metadata.py          # Извлечение метаданных
│   ├── cache.py             # Кэширование результатов
│   └── utils.py             # Вспомогательные функции
│
├── cache/                   # Папка для кэша результатов
├── output/                  # Папка для нормализованных файлов
├── logs/                    # Папка для логов
│
├── config.py                # Конфигурационные параметры
├── main.py                  # Точка входа в приложение
├── cli.py                   # Интерфейс командной строки
├── requirements.txt         # Зависимости
└── README.md                # Документация
```

## Особенности нормализации по жанрам

DJ Setlist Analyzer использует специальные профили нормализации для разных жанров электронной музыки:

| Жанр | Целевой LUFS | Пиковый уровень | Динамический диапазон |
|------|--------------|-----------------|------------------------|
| Techno | -8.0 | -0.3 | Сжатый |
| House | -9.0 | -0.3 | Сжатый |
| Trance | -8.0 | -0.3 | Сжатый |
| Drum and Bass | -7.0 | -0.2 | Сжатый |
| Dubstep | -6.0 | -0.1 | Очень сжатый |
| Hip-Hop | -9.0 | -0.5 | Средний |
| Pop | -10.0 | -0.5 | Средний |
| Rock | -11.0 | -0.5 | Средний |
| Jazz | -14.0 | -1.0 | Широкий |
| Classical | -18.0 | -1.5 | Очень широкий |
| Ambient | -16.0 | -1.0 | Широкий |

## Советы по использованию

- Для клубной музыки рекомендуется использовать целевой LUFS около -8 dB
- Для предварительного прослушивания дома используйте -14 dB (стандарт потоковых сервисов)
- Используйте формат WAV при подготовке к живому выступлению
- Используйте MP3 320kbps для экономии места при сохранении личной коллекции
- Анализируйте все треки перед нормализацией для оценки необходимых изменений

## Лицензия

MIT License

## Авторы

DJ Tools Team