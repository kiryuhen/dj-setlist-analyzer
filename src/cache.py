"""
Модуль для кэширования результатов анализа аудиофайлов
"""
import os
import json
import time
import logging
from datetime import datetime, timedelta
from pathlib import Path
import hashlib

import config

logger = logging.getLogger(__name__)


class AnalysisCache:
    """Класс для работы с кэшем результатов анализа аудио"""

    def __init__(self, cache_dir=None):
        """
        Инициализация кэша

        Args:
            cache_dir: Директория для хранения кэша (по умолчанию берется из config.CACHE_DIR)
        """
        self.cache_dir = Path(cache_dir or config.CACHE_DIR)
        self.cache_dir.mkdir(exist_ok=True, parents=True)
        self.cache_index_path = self.cache_dir / 'index.json'
        self.cache_index = self._load_cache_index()
        self._clean_expired_cache()

    def _load_cache_index(self):
        """Загрузка индекса кэша"""
        if not self.cache_index_path.exists():
            return {}

        try:
            with open(self.cache_index_path, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"Ошибка при чтении индекса кэша: {e}")
            return {}

    def _save_cache_index(self):
        """Сохранение индекса кэша"""
        try:
            with open(self.cache_index_path, 'w') as f:
                json.dump(self.cache_index, f, indent=2)
        except IOError as e:
            logger.error(f"Ошибка при сохранении индекса кэша: {e}")

    def _get_file_hash(self, file_path):
        """
        Создание хэша файла на основе пути и времени модификации

        Args:
            file_path: Путь к файлу

        Returns:
            str: Хэш файла
        """
        file_path = Path(file_path)
        if not file_path.exists():
            return None

        # Берем полный путь, размер и время модификации файла
        file_stats = file_path.stat()
        file_data = f"{file_path.absolute()}|{file_stats.st_size}|{file_stats.st_mtime}"

        # Создаем MD5 хэш
        return hashlib.md5(file_data.encode()).hexdigest()

    def _clean_expired_cache(self):
        """Очистка устаревших записей кэша"""
        if not config.CACHE_ENABLED:
            return

        current_time = time.time()
        expiry_seconds = config.CACHE_EXPIRY_DAYS * 24 * 60 * 60
        expired_files = []

        for file_hash, cache_data in list(self.cache_index.items()):
            if current_time - cache_data['timestamp'] > expiry_seconds:
                expired_files.append(file_hash)
                cache_path = self.cache_dir / f"{file_hash}.json"

                if cache_path.exists():
                    try:
                        cache_path.unlink()
                    except IOError as e:
                        logger.warning(f"Не удалось удалить устаревший кэш {cache_path}: {e}")

                del self.cache_index[file_hash]

        if expired_files:
            logger.info(f"Удалено {len(expired_files)} устаревших записей кэша")
            self._save_cache_index()

    def get(self, file_path):
        """
        Получение данных из кэша для указанного файла

        Args:
            file_path: Путь к аудиофайлу

        Returns:
            dict: Данные анализа или None, если кэш отсутствует
        """
        if not config.CACHE_ENABLED:
            return None

        file_hash = self._get_file_hash(file_path)
        if not file_hash or file_hash not in self.cache_index:
            return None

        cache_path = self.cache_dir / f"{file_hash}.json"
        if not cache_path.exists():
            # Если файл кэша отсутствует, удаляем запись из индекса
            if file_hash in self.cache_index:
                del self.cache_index[file_hash]
                self._save_cache_index()
            return None

        try:
            with open(cache_path, 'r') as f:
                cached_data = json.load(f)
                logger.debug(f"Загружены данные из кэша для {file_path}")
                return cached_data
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"Ошибка при чтении кэша для {file_path}: {e}")
            return None

    def save(self, file_path, data):
        """
        Сохранение данных анализа в кэш

        Args:
            file_path: Путь к аудиофайлу
            data: Данные анализа для сохранения

        Returns:
            bool: True, если сохранение прошло успешно
        """
        if not config.CACHE_ENABLED:
            return False

        file_hash = self._get_file_hash(file_path)
        if not file_hash:
            return False

        cache_path = self.cache_dir / f"{file_hash}.json"

        try:
            with open(cache_path, 'w') as f:
                json.dump(data, f, indent=2)

            # Обновляем индекс
            self.cache_index[file_hash] = {
                'file_path': str(file_path),
                'timestamp': time.time()
            }
            self._save_cache_index()

            logger.debug(f"Сохранены данные в кэш для {file_path}")
            return True
        except IOError as e:
            logger.error(f"Ошибка при сохранении кэша для {file_path}: {e}")
            return False

    def invalidate(self, file_path):
        """
        Инвалидация кэша для указанного файла

        Args:
            file_path: Путь к аудиофайлу

        Returns:
            bool: True, если инвалидация прошла успешно
        """
        if not config.CACHE_ENABLED:
            return False

        file_hash = self._get_file_hash(file_path)
        if not file_hash or file_hash not in self.cache_index:
            return False

        cache_path = self.cache_dir / f"{file_hash}.json"

        if cache_path.exists():
            try:
                cache_path.unlink()
            except IOError as e:
                logger.warning(f"Не удалось удалить кэш для {file_path}: {e}")
                return False

        # Удаляем запись из индекса
        if file_hash in self.cache_index:
            del self.cache_index[file_hash]
            self._save_cache_index()

        logger.debug(f"Кэш для {file_path} инвалидирован")
        return True

    def clear_all(self):
        """
        Очистка всего кэша

        Returns:
            int: Количество удаленных файлов кэша
        """
        file_count = 0

        for cache_file in self.cache_dir.glob('*.json'):
            if cache_file.name != 'index.json':
                try:
                    cache_file.unlink()
                    file_count += 1
                except IOError as e:
                    logger.warning(f"Не удалось удалить кэш {cache_file}: {e}")

        # Очищаем индекс, оставляя только структуру
        self.cache_index = {}
        self._save_cache_index()

        logger.info(f"Удалено {file_count} файлов кэша")
        return file_count