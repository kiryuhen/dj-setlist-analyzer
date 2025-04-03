"""
Конфигурационный файл для DJ сетлист-анализатора
"""
import os
from pathlib import Path

# Основные директории
BASE_DIR = Path(__file__).resolve().parent
SETLIST_DIR = os.path.join(BASE_DIR, "setlist")
CACHE_DIR = os.path.join(BASE_DIR, "cache")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

# Создаем директории, если они не существуют
os.makedirs(SETLIST_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Поддерживаемые аудио форматы
SUPPORTED_FORMATS = ['.mp3', '.wav', '.flac', '.ogg', '.m4a', '.aac']

# Параметры нормализации по умолчанию
DEFAULT_TARGET_LUFS = -14.0  # Стандарт для стриминговых сервисов
DEFAULT_TARGET_PEAK = -1.0   # дБ

# Настройки нормализации по жанрам
GENRE_PRESETS = {
    "techno": {
        "target_lufs": -8.0,
        "target_peak": -0.3,
        "dynamic_range": "compressed"
    },
    "house": {
        "target_lufs": -9.0,
        "target_peak": -0.3,
        "dynamic_range": "compressed"
    },
    "trance": {
        "target_lufs": -8.0,
        "target_peak": -0.3,
        "dynamic_range": "compressed"
    },
    "dnb": {  # Drum and Bass
        "target_lufs": -7.0,
        "target_peak": -0.2,
        "dynamic_range": "compressed"
    },
    "dubstep": {
        "target_lufs": -6.0,
        "target_peak": -0.1,
        "dynamic_range": "very_compressed"
    },
    "hiphop": {
        "target_lufs": -9.0,
        "target_peak": -0.5,
        "dynamic_range": "medium"
    },
    "pop": {
        "target_lufs": -10.0,
        "target_peak": -0.5,
        "dynamic_range": "medium"
    },
    "rock": {
        "target_lufs": -11.0,
        "target_peak": -0.5,
        "dynamic_range": "medium"
    },
    "jazz": {
        "target_lufs": -14.0,
        "target_peak": -1.0,
        "dynamic_range": "wide"
    },
    "classical": {
        "target_lufs": -18.0,
        "target_peak": -1.5,
        "dynamic_range": "very_wide"
    },
    "ambient": {
        "target_lufs": -16.0,
        "target_peak": -1.0,
        "dynamic_range": "wide"
    },
    "default": {
        "target_lufs": -14.0,
        "target_peak": -1.0,
        "dynamic_range": "medium"
    }
}

# Настройки кэша
CACHE_ENABLED = True
CACHE_EXPIRY_DAYS = 30  # Срок действия кэша в днях

# Многопоточность
MAX_WORKERS = os.cpu_count() or 4  # Используем все доступные ядра процессора
MIN_FILE_SIZE_FOR_PARALLEL = 5 * 1024 * 1024  # 5 MB

# Настройки анализа
ANALYSIS_SAMPLE_RATE = 44100
ANALYSIS_DURATION = None  # None для анализа всего трека

# Пути к FFmpeg
# Путь к папке bin с FFmpeg в директории проекта
FFMPEG_PATH = os.path.join(BASE_DIR, "bin")
FFMPEG_BIN = os.path.join(FFMPEG_PATH, "ffmpeg.exe")
FFPROBE_BIN = os.path.join(FFMPEG_PATH, "ffprobe.exe")

# Проверяем наличие FFmpeg в указанном пути
FFMPEG_AVAILABLE = os.path.exists(FFMPEG_BIN)
if not FFMPEG_AVAILABLE:
    print(f"ВНИМАНИЕ: FFmpeg не найден по пути {FFMPEG_BIN}")
    print("Некоторые функции работы с аудио могут быть недоступны.")
else:
    print(f"FFmpeg найден: {FFMPEG_BIN}")
    # Устанавливаем переменные среды для pydub
    os.environ["PATH"] += os.pathsep + str(FFMPEG_PATH)
    os.environ["FFMPEG_BINARY"] = str(FFMPEG_BIN)
    os.environ["FFPROBE_BINARY"] = str(FFPROBE_BIN)