"""
Модуль для анализа аудиофайлов
"""
import os
import logging
from pathlib import Path
import concurrent.futures
import time

import numpy as np
import pandas as pd
import librosa
import soundfile as sf
import pyloudnorm as pyln
from tqdm import tqdm

import config
from src.metadata import MetadataExtractor
from src.cache import AnalysisCache

logger = logging.getLogger(__name__)


class AudioAnalyzer:
    """Класс для анализа аудиофайлов"""

    def __init__(self, folder_path=None):
        """
        Инициализация анализатора

        Args:
            folder_path: Путь к папке с аудиофайлами (по умолчанию берется из config.SETLIST_DIR)
        """
        self.folder_path = Path(folder_path or config.SETLIST_DIR)
        self.audio_files = []
        self.metadata_extractor = MetadataExtractor()
        self.cache = AnalysisCache()
        self.scan_folder()

    def scan_folder(self):
        """
        Сканирование папки на наличие аудиофайлов

        Returns:
            list: Список найденных аудиофайлов
        """
        if not self.folder_path.exists():
            logger.error(f"Папка не найдена: {self.folder_path}")
            return []

        self.audio_files = []

        for root, _, files in os.walk(self.folder_path):
            for file in files:
                file_path = Path(root) / file
                if file_path.suffix.lower() in config.SUPPORTED_FORMATS:
                    self.audio_files.append(file_path)

        logger.info(f"Найдено {len(self.audio_files)} аудиофайлов в {self.folder_path}")
        return self.audio_files

    def analyze_track(self, file_path, force_analyze=False):
        """
        Анализ одного трека

        Args:
            file_path: Путь к аудиофайлу
            force_analyze: Принудительный анализ, игнорируя кэш

        Returns:
            dict: Результаты анализа
        """
        file_path = Path(file_path)
        logger.debug(f"Анализ трека: {file_path}")

        # Проверяем кэш
        if not force_analyze and config.CACHE_ENABLED:
            cached_results = self.cache.get(file_path)
            if cached_results:
                logger.debug(f"Использованы кэшированные результаты для {file_path}")
                return cached_results

        try:
            # Извлекаем метаданные
            metadata = self.metadata_extractor.extract_metadata(file_path)

            # Получаем аудио данные
            y, sr = librosa.load(file_path, sr=None)
            duration = librosa.get_duration(y=y, sr=sr)

            # Базовые характеристики аудио
            analysis_results = {
                'file': file_path.name,
                'file_path': str(file_path),
                'duration': duration,
            }

            # Добавляем метаданные
            analysis_results.update(metadata)

            # Анализ громкости
            loudness_metrics = self._analyze_loudness(file_path, y, sr)
            analysis_results.update(loudness_metrics)

            # Анализ спектра
            spectral_metrics = self._analyze_spectrum(y, sr)
            analysis_results.update(spectral_metrics)

            # Анализ динамики
            dynamic_metrics = self._analyze_dynamics(y)
            analysis_results.update(dynamic_metrics)

            # Анализ начала и конца трека (для микширования)
            mixing_metrics = self._analyze_for_mixing(y, sr)
            analysis_results.update(mixing_metrics)

            # Сохраняем результаты в кэш
            if config.CACHE_ENABLED:
                self.cache.save(file_path, analysis_results)

            return analysis_results

        except Exception as e:
            logger.error(f"Ошибка при анализе {file_path}: {e}")
            return {
                'file': file_path.name,
                'file_path': str(file_path),
                'error': str(e)
            }

    def _analyze_loudness(self, file_path, y, sr):
        """
        Анализ громкости трека

        Args:
            file_path: Путь к аудиофайлу
            y: Аудио данные
            sr: Частота дискретизации

        Returns:
            dict: Метрики громкости
        """
        try:
            # Рассчитываем RMS (среднеквадратичную громкость)
            rms = np.sqrt(np.mean(y ** 2))

            # Рассчитываем пиковые значения
            peak = np.max(np.abs(y))
            peak_db = 20 * np.log10(peak) if peak > 0 else -120

            # Рассчитываем LUFS (более точная мера воспринимаемой громкости)
            try:
                meter = pyln.Meter(sr)
                # Интегральная (общая) громкость
                lufs_integrated = meter.integrated_loudness(y)

                # Кратковременная (short-term) громкость - используем блоки по 3 секунды
                block_size = sr * 3
                blocks = [y[i:i + block_size] for i in range(0, len(y), block_size) if len(y[i:i + block_size]) >= sr]

                lufs_blocks = [meter.integrated_loudness(block) for block in blocks]
                lufs_short_term_max = max(lufs_blocks) if lufs_blocks else lufs_integrated
                lufs_short_term_min = min(lufs_blocks) if lufs_blocks else lufs_integrated

                # Громкость вступления (первые 20 секунд)
                intro_samples = min(sr * 20, len(y))
                y_intro = y[:intro_samples] if intro_samples > sr else y
                lufs_intro = meter.integrated_loudness(y_intro)

                # Громкость окончания (последние 20 секунд)
                outro_samples = min(sr * 20, len(y))
                y_outro = y[-outro_samples:] if outro_samples > sr else y
                lufs_outro = meter.integrated_loudness(y_outro)

            except Exception as e:
                logger.warning(f"Ошибка при расчете LUFS: {e}")
                lufs_integrated = -23.0
                lufs_short_term_max = -23.0
                lufs_short_term_min = -23.0
                lufs_intro = -23.0
                lufs_outro = -23.0

            return {
                'rms': float(rms),
                'rms_db': float(20 * np.log10(rms) if rms > 0 else -120),
                'peak': float(peak),
                'peak_db': float(peak_db),
                'lufs_integrated': float(lufs_integrated),
                'lufs_short_term_max': float(lufs_short_term_max),
                'lufs_short_term_min': float(lufs_short_term_min),
                'lufs_intro': float(lufs_intro),
                'lufs_outro': float(lufs_outro),
                'dynamic_range': float(lufs_short_term_max - lufs_short_term_min) if lufs_short_term_max > -120 else 0,
            }

        except Exception as e:
            logger.error(f"Ошибка при анализе громкости {file_path}: {e}")
            return {
                'rms': 0.0,
                'rms_db': -120.0,
                'peak': 0.0,
                'peak_db': -120.0,
                'lufs_integrated': -23.0,
                'dynamic_range': 0.0,
            }

    def _analyze_spectrum(self, y, sr):
        """
        Анализ спектра трека

        Args:
            y: Аудио данные
            sr: Частота дискретизации

        Returns:
            dict: Спектральные характеристики
        """
        try:
            # Центроид спектра (яркость звука)
            cent = librosa.feature.spectral_centroid(y=y, sr=sr).mean()

            # Спектральный контраст
            contrast = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr))

            # Спектральный баланс - энергия в разных диапазонах
            stft = np.abs(librosa.stft(y))

            # Частотные диапазоны (в бинах)
            # Для sr=44100, размер окна=2048, частотное разрешение ~21.5 Гц на бин
            bin_resolution = sr / (2 * (stft.shape[0] - 1))

            # Определяем границы диапазонов в бинах
            sub_bin = int(60 / bin_resolution)  # < 60 Hz (сабы)
            bass_bin = int(250 / bin_resolution)  # 60-250 Hz (бас)
            low_mid_bin = int(500 / bin_resolution)  # 250-500 Hz (нижняя середина)
            mid_bin = int(2000 / bin_resolution)  # 500-2000 Hz (середина)
            high_mid_bin = int(4000 / bin_resolution)  # 2000-4000 Hz (верхняя середина)
            high_bin = int(10000 / bin_resolution)  # 4000-10000 Hz (высокие)
            # > 10000 Hz (очень высокие)

            # Считаем энергию в каждом диапазоне
            total_energy = np.sum(stft)

            sub_energy = np.sum(stft[:sub_bin, :]) / total_energy if total_energy > 0 else 0
            bass_energy = np.sum(stft[sub_bin:bass_bin, :]) / total_energy if total_energy > 0 else 0
            low_mid_energy = np.sum(stft[bass_bin:low_mid_bin, :]) / total_energy if total_energy > 0 else 0
            mid_energy = np.sum(stft[low_mid_bin:mid_bin, :]) / total_energy if total_energy > 0 else 0
            high_mid_energy = np.sum(stft[mid_bin:high_mid_bin, :]) / total_energy if total_energy > 0 else 0
            high_energy = np.sum(stft[high_mid_bin:high_bin, :]) / total_energy if total_energy > 0 else 0
            very_high_energy = np.sum(stft[high_bin:, :]) / total_energy if total_energy > 0 else 0

            return {
                'spectral_centroid': float(cent),
                'spectral_contrast': float(contrast),
                'sub_energy': float(sub_energy),
                'bass_energy': float(bass_energy),
                'low_mid_energy': float(low_mid_energy),
                'mid_energy': float(mid_energy),
                'high_mid_energy': float(high_mid_energy),
                'high_energy': float(high_energy),
                'very_high_energy': float(very_high_energy),
                'bass_to_highs_ratio': float((sub_energy + bass_energy) / (high_energy + very_high_energy)) if (
                                                                                                                           high_energy + very_high_energy) > 0 else 0,
            }

        except Exception as e:
            logger.error(f"Ошибка при анализе спектра: {e}")
            return {
                'spectral_centroid': 0.0,
                'spectral_contrast': 0.0,
                'sub_energy': 0.0,
                'bass_energy': 0.0,
                'low_mid_energy': 0.0,
                'mid_energy': 0.0,
                'high_mid_energy': 0.0,
                'high_energy': 0.0,
                'very_high_energy': 0.0,
                'bass_to_highs_ratio': 0.0,
            }

    def _analyze_dynamics(self, y):
        """
        Анализ динамики трека

        Args:
            y: Аудио данные

        Returns:
            dict: Характеристики динамики
        """
        try:
            # Вычисляем огибающую сигнала
            envelope = np.abs(y)

            # Вычисляем crest factor (отношение пика к RMS)
            peak = np.max(envelope)
            rms = np.sqrt(np.mean(y ** 2))
            crest_factor = peak / rms if rms > 0 else 0

            # Вычисляем динамический диапазон (распределение громкости)
            # Как отношение 95-го перцентиля к 5-му перцентилю
            percentile_95 = np.percentile(envelope, 95)
            percentile_5 = np.percentile(envelope, 5)
            dynamic_range_ratio = percentile_95 / percentile_5 if percentile_5 > 0 else 0

            # Оценка компрессии (чем ближе к 1, тем больше сжатие)
            # В нормальной динамике crest_factor обычно > 10
            compression_estimate = 1.0 - min(1.0, crest_factor / 15.0)

            return {
                'crest_factor': float(crest_factor),
                'dynamic_range_ratio': float(dynamic_range_ratio),
                'compression_estimate': float(compression_estimate),
            }

        except Exception as e:
            logger.error(f"Ошибка при анализе динамики: {e}")
            return {
                'crest_factor': 0.0,
                'dynamic_range_ratio': 0.0,
                'compression_estimate': 0.0,
            }

    def _analyze_for_mixing(self, y, sr):
        """
        Анализ характеристик для микширования (начало и конец трека)

        Args:
            y: Аудио данные
            sr: Частота дискретизации

        Returns:
            dict: Характеристики для микширования
        """
        try:
            # Длина в сэмплах начала/конца трека для анализа (8 секунд)
            intro_length = min(8 * sr, len(y) // 2)
            outro_length = min(8 * sr, len(y) // 2)

            # Извлекаем фрагменты трека
            y_intro = y[:intro_length]
            y_outro = y[-outro_length:]

            # Анализируем начало и конец
            # RMS для начала и конца
            rms_intro = np.sqrt(np.mean(y_intro ** 2))
            rms_outro = np.sqrt(np.mean(y_outro ** 2))

            # Находим позиции кульминаций (простой метод)
            # Делим трек на 10-секундные сегменты и ищем самый громкий
            segment_length = 10 * sr
            num_segments = max(1, len(y) // segment_length)

            segment_energies = []
            for i in range(num_segments):
                start = i * segment_length
                end = min(start + segment_length, len(y))
                segment = y[start:end]
                segment_energies.append((i, np.mean(segment ** 2)))

            # Находим самый громкий сегмент (первый дроп)
            sorted_segments = sorted(segment_energies, key=lambda x: x[1], reverse=True)

            if len(sorted_segments) > 0:
                first_drop_segment = sorted_segments[0][0]
                first_drop_time = (first_drop_segment * segment_length) / sr
            else:
                first_drop_time = 0

            # Примерное микширование - сколько секунд нужно для микширования в начале/конце
            # Определяем по нарастанию/убыванию энергии
            energy_rise = self._calculate_energy_slope(y_intro)
            energy_fall = self._calculate_energy_slope(y_outro, rising=False)

            # Конвертируем в рекомендуемое время микса (в секундах)
            if energy_rise > 0.5:  # Резкое начало
                intro_mix_time = 4
            elif energy_rise > 0.2:  # Среднее нарастание
                intro_mix_time = 8
            else:  # Плавное начало
                intro_mix_time = 16

            if energy_fall > 0.5:  # Резкий конец
                outro_mix_time = 4
            elif energy_fall > 0.2:  # Среднее затухание
                outro_mix_time = 8
            else:  # Плавный конец
                outro_mix_time = 16

            return {
                'rms_intro': float(rms_intro),
                'rms_outro': float(rms_outro),
                'first_drop_time': float(first_drop_time),
                'energy_rise_rate': float(energy_rise),
                'energy_fall_rate': float(energy_fall),
                'intro_mix_recommendation': intro_mix_time,
                'outro_mix_recommendation': outro_mix_time,
            }

        except Exception as e:
            logger.error(f"Ошибка при анализе для микширования: {e}")
            return {
                'rms_intro': 0.0,
                'rms_outro': 0.0,
                'first_drop_time': 0.0,
                'energy_rise_rate': 0.0,
                'energy_fall_rate': 0.0,
                'intro_mix_recommendation': 8,
                'outro_mix_recommendation': 8,
            }

    def _calculate_energy_slope(self, y, window_size=1024, rising=True):
        """
        Расчет скорости нарастания/убывания энергии

        Args:
            y: Аудио данные
            window_size: Размер окна для анализа
            rising: True для анализа нарастания, False для анализа затухания

        Returns:
            float: Нормализованная скорость изменения энергии
        """
        if len(y) < window_size * 2:
            return 0.0

        # Разбиваем на окна
        num_windows = len(y) // window_size
        energies = []

        for i in range(num_windows):
            start = i * window_size
            end = start + window_size
            window = y[start:end]
            energy = np.mean(window ** 2)
            energies.append(energy)

        if len(energies) < 2:
            return 0.0

        # Если нужно анализировать затухание, инвертируем последовательность
        if not rising:
            energies.reverse()

        # Нормализуем энергии
        max_energy = max(energies)
        if max_energy > 0:
            energies = [e / max_energy for e in energies]

        # Находим скорость нарастания как отношение разницы
        # между конечной и начальной энергией к числу окон
        if len(energies) > 1:
            return (energies[-1] - energies[0]) / len(energies)
        else:
            return 0.0

    def analyze_all(self, force_analyze=False, parallel=True):
        """
        Анализ всех аудиофайлов

        Args:
            force_analyze: Принудительный анализ, игнорируя кэш
            parallel: Использовать параллельную обработку

        Returns:
            list: Список результатов анализа
        """
        if not self.audio_files:
            logger.warning("Нет файлов для анализа")
            return []

        results = []
        start_time = time.time()

        try:
            if parallel and len(self.audio_files) > 1:
                with concurrent.futures.ThreadPoolExecutor(max_workers=config.MAX_WORKERS) as executor:
                    # Создаем задачи анализа
                    future_to_file = {
                        executor.submit(self.analyze_track, file_path, force_analyze): file_path
                        for file_path in self.audio_files
                    }

                    # Обрабатываем результаты по мере их готовности с прогресс-баром
                    for future in tqdm(
                            concurrent.futures.as_completed(future_to_file),
                            total=len(self.audio_files),
                            desc="Анализ аудиофайлов"
                    ):
                        file_path = future_to_file[future]
                        try:
                            result = future.result()
                            results.append(result)
                        except Exception as e:
                            logger.error(f"Ошибка при анализе {file_path}: {e}")
                            results.append({
                                'file': file_path.name,
                                'file_path': str(file_path),
                                'error': str(e)
                            })
            else:
                # Последовательный анализ с прогрессбаром
                for file_path in tqdm(self.audio_files, desc="Анализ аудиофайлов"):
                    result = self.analyze_track(file_path, force_analyze)
                    results.append(result)

        except Exception as e:
            logger.error(f"Ошибка при выполнении анализа: {e}")

        elapsed_time = time.time() - start_time
        logger.info(f"Анализ завершен за {elapsed_time:.2f} сек. Проанализировано {len(results)} файлов.")

        return results

    def get_analysis_dataframe(self, results=None):
        """
        Преобразование результатов анализа в pandas DataFrame

        Args:
            results: Результаты анализа (если None, будет выполнен analyze_all)

        Returns:
            pandas.DataFrame: DataFrame с результатами анализа
        """
        if results is None:
            results = self.analyze_all()

        if not results:
            return pd.DataFrame()

        # Преобразуем в DataFrame
        df = pd.DataFrame(results)

        # Сортируем по имени файла
        if 'file' in df.columns:
            df = df.sort_values('file')

        return df