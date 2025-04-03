"""
Модуль для нормализации громкости аудиофайлов
"""
import os
import logging
from pathlib import Path
import concurrent.futures
import time
import tempfile

import numpy as np
import pyloudnorm as pyln
import soundfile as sf
from pydub import AudioSegment
from tqdm import tqdm

import config
from src.analyzer import AudioAnalyzer

logger = logging.getLogger(__name__)


class AudioNormalizer:
    """Класс для нормализации громкости аудиофайлов"""

    def __init__(self, analyzer=None):
        """
        Инициализация нормализатора

        Args:
            analyzer: Экземпляр AudioAnalyzer (если None, будет создан новый)
        """
        self.analyzer = analyzer or AudioAnalyzer()
        self.output_dir = Path(config.OUTPUT_DIR)
        self.output_dir.mkdir(exist_ok=True, parents=True)

    def normalize_track(self, file_path, target_lufs=None, target_peak=None,
                        genre=None, dynamic_range=None, output_format=None,
                        output_dir=None, dry_run=False):
        """
        Нормализация громкости одного трека

        Args:
            file_path: Путь к аудиофайлу
            target_lufs: Целевой уровень LUFS (если None, берется из жанра или по умолчанию)
            target_peak: Целевой пиковый уровень в dB (если None, берется из жанра или по умолчанию)
            genre: Жанр трека (для определения параметров нормализации)
            dynamic_range: Целевой динамический диапазон ('wide', 'medium', 'compressed', 'very_compressed')
            output_format: Формат выходного файла (wav, mp3, flac, ogg)
            output_dir: Директория для сохранения нормализованных файлов
            dry_run: Если True, только рассчитывает параметры без создания файлов

        Returns:
            dict: Результаты нормализации
        """
        file_path = Path(file_path)
        
        output_dir = Path(output_dir or self.output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)

        # Если формат не указан, используем формат исходного файла
        if not output_format:
            output_format = file_path.suffix.lstrip('.').lower()
            if output_format not in ['wav', 'mp3', 'flac', 'ogg']:
                output_format = 'wav'  # По умолчанию WAV, если формат не поддерживается

        # Анализируем файл
        analysis = self.analyzer.analyze_track(file_path)

        # Определяем параметры нормализации
        norm_params = self._get_normalization_params(
            analysis, target_lufs, target_peak, genre, dynamic_range
        )

        # Обновляем анализ информацией о нормализации
        analysis.update({
            'normalization': norm_params
        })

        if dry_run:
            # Если сухой прогон, просто возвращаем параметры
            return analysis

        # Создаем выходной путь
        output_filename = f"{file_path.stem}_normalized.{output_format}"
        output_path = output_dir / output_filename

        try:
            # Выполняем нормализацию
            self._apply_normalization(file_path, output_path, norm_params, output_format)

            # Добавляем информацию о выходном файле
            analysis['output_file'] = str(output_path)
            analysis['normalization']['success'] = True

            return analysis

        except Exception as e:
            logger.error(f"Ошибка при нормализации {file_path}: {e}")
            analysis['normalization']['success'] = False
            analysis['normalization']['error'] = str(e)
            return analysis
        
    def normalize_track_safe(self, file_path, target_lufs=None, target_peak=None,
                    genre=None, dynamic_range=None, output_format=None,
                    output_dir=None, dry_run=False):
        try:
            return self.normalize_track(file_path, target_lufs, target_peak, genre, 
                                    dynamic_range, output_format, output_dir, dry_run)
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"Ошибка при нормализации {file_path}: {e}")
            
            # Возвращаем базовую информацию о файле с отметкой об ошибке
            file_path = Path(file_path)
            analysis = self.analyzer.analyze_track(file_path)
            analysis['normalization'] = {
                'success': False,
                'error': str(e),
                'target_lufs': target_lufs,
                'target_peak': target_peak,
                'dynamic_range': dynamic_range
            }
            return analysis

# И замените метод normalize_all в том же классе:

    def normalize_all(self, target_lufs=None, target_peak=None, genre=None,
                    dynamic_range=None, output_format='wav', output_dir=None,
                    parallel=True):
        """
        Нормализация всех треков
        
        Args:
            target_lufs: Целевой уровень LUFS
            target_peak: Целевой пиковый уровень в dB
            genre: Жанр треков (для определения параметров нормализации)
            dynamic_range: Целевой динамический диапазон
            output_format: Формат выходных файлов
            output_dir: Директория для сохранения нормализованных файлов
            parallel: Использовать параллельную обработку
            
        Returns:
            list: Результаты нормализации для всех треков
        """
        results = []
        audio_files = self.analyzer.audio_files
        
        if not audio_files:
            logger.warning("Нет файлов для нормализации")
            return []
        
        output_dir = Path(output_dir or self.output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        
        start_time = time.time()
        
        try:
            if parallel and len(audio_files) > 1:
                with concurrent.futures.ThreadPoolExecutor(max_workers=config.MAX_WORKERS) as executor:
                    # Создаем задачи нормализации
                    future_to_file = {}
                    for file_path in audio_files:
                        future = executor.submit(
                            self.normalize_track_safe,  # Используем безопасный метод!
                            file_path=file_path,
                            target_lufs=target_lufs,
                            target_peak=target_peak,
                            genre=genre,
                            dynamic_range=dynamic_range,
                            output_format=output_format,
                            output_dir=output_dir
                        )
                        future_to_file[future] = file_path
                    
                    # Обрабатываем результаты с прогресс-баром
                    for future in tqdm(
                        concurrent.futures.as_completed(future_to_file),
                        total=len(audio_files),
                        desc="Нормализация аудиофайлов"
                    ):
                        file_path = future_to_file[future]
                        try:
                            result = future.result()
                            results.append(result)
                        except Exception as e:
                            logger.error(f"Ошибка при нормализации {file_path}: {e}")
                            results.append({
                                'file': file_path.name,
                                'file_path': str(file_path),
                                'normalization': {
                                    'success': False,
                                    'error': str(e)
                                }
                            })
            else:
                # Последовательная нормализация с прогресс-баром
                for file_path in tqdm(audio_files, desc="Нормализация аудиофайлов"):
                    try:
                        result = self.normalize_track_safe(  # Используем безопасный метод!
                            file_path=file_path,
                            target_lufs=target_lufs,
                            target_peak=target_peak,
                            genre=genre,
                            dynamic_range=dynamic_range,
                            output_format=output_format,
                            output_dir=output_dir
                        )
                        results.append(result)
                    except Exception as e:
                        logger.error(f"Ошибка при нормализации {file_path}: {e}")
                        results.append({
                            'file': file_path.name,
                            'file_path': str(file_path),
                            'normalization': {
                                'success': False,
                                'error': str(e)
                            }
                        })
                        
        except Exception as e:
            logger.error(f"Ошибка при выполнении нормализации: {e}")
            
        elapsed_time = time.time() - start_time
        logger.info(f"Нормализация завершена за {elapsed_time:.2f} сек. Обработано {len(results)} файлов.")
        
        return results

    def _get_normalization_params(self, analysis, target_lufs=None, target_peak=None,
                                  genre=None, dynamic_range=None):
        """
        Определение параметров нормализации

        Args:
            analysis: Результаты анализа файла
            target_lufs: Целевой уровень LUFS
            target_peak: Целевой пиковый уровень в dB
            genre: Жанр трека
            dynamic_range: Целевой динамический диапазон

        Returns:
            dict: Параметры нормализации
        """
        # Если жанр не указан, пытаемся определить из анализа
        if not genre:
            if 'genre' in analysis:
                genre = analysis['genre'].lower() if analysis['genre'] else None
            elif 'detected_genre' in analysis:
                genre = analysis['detected_genre'].lower()

        # Если жанр задан, берем настройки из конфига
        if genre and genre in config.GENRE_PRESETS:
            preset = config.GENRE_PRESETS[genre]
            default_target_lufs = preset['target_lufs']
            default_target_peak = preset['target_peak']
            default_dynamic_range = preset['dynamic_range']
        else:
            # Если жанр не определен, используем настройки по умолчанию
            default_target_lufs = config.DEFAULT_TARGET_LUFS
            default_target_peak = config.DEFAULT_TARGET_PEAK
            default_dynamic_range = 'medium'

        # Используем переданные параметры или значения по умолчанию
        target_lufs = target_lufs if target_lufs is not None else default_target_lufs
        target_peak = target_peak if target_peak is not None else default_target_peak
        dynamic_range = dynamic_range or default_dynamic_range

        # Рассчитываем параметры динамической обработки
        dynamic_params = self._calculate_dynamic_processing(
            analysis, dynamic_range
        )

        # Вычисляем коэффициент усиления для достижения целевого LUFS
        current_lufs = analysis.get('lufs_integrated', -23.0)
        gain_db = target_lufs - current_lufs

        # Проверяем, чтобы пиковый уровень не превышал целевой
        current_peak_db = analysis.get('peak_db', -1.0)
        headroom_db = target_peak - (current_peak_db + gain_db)

        # Если headroom отрицательный, уменьшаем усиление
        if headroom_db < 0:
            gain_db += headroom_db  # Уменьшаем усиление на величину превышения

        # Собираем все параметры нормализации
        return {
            'target_lufs': float(target_lufs),
            'target_peak': float(target_peak),
            'gain_db': float(gain_db),
            'dynamic_range_processing': dynamic_range,
            'compression_params': dynamic_params,
            'genre': genre,
            'original_lufs': float(current_lufs),
            'original_peak_db': float(current_peak_db),
            'expected_output_lufs': float(min(target_lufs, current_lufs + gain_db)),
            'expected_output_peak_db': float(min(target_peak, current_peak_db + gain_db)),
        }

    def _calculate_dynamic_processing(self, analysis, dynamic_range):
        """
        Рассчет параметров динамической обработки

        Args:
            analysis: Результаты анализа файла
            dynamic_range: Целевой динамический диапазон

        Returns:
            dict: Параметры динамической обработки
        """
        # Получаем текущую оценку компрессии из анализа
        current_compression = analysis.get('compression_estimate', 0.5)

        # Определяем целевую компрессию в зависимости от требуемого динамического диапазона
        if dynamic_range == 'very_wide':
            target_compression = 0.1  # Минимальная компрессия
            ratio = 1.1  # Почти без компрессии
            threshold = -30.0  # Очень низкий порог

        elif dynamic_range == 'wide':
            target_compression = 0.3
            ratio = 1.5
            threshold = -24.0

        elif dynamic_range == 'medium':
            target_compression = 0.5
            ratio = 2.0
            threshold = -20.0

        elif dynamic_range == 'compressed':
            target_compression = 0.7
            ratio = 3.0
            threshold = -18.0

        elif dynamic_range == 'very_compressed':
            target_compression = 0.9
            ratio = 4.0
            threshold = -16.0

        else:  # По умолчанию medium
            target_compression = 0.5
            ratio = 2.0
            threshold = -20.0

        # Если текущая компрессия больше целевой, не добавляем дополнительную
        if current_compression >= target_compression:
            apply_compression = False
            ratio = 1.0  # Без компрессии
        else:
            apply_compression = True

        # Собираем параметры компрессии
        return {
            'apply_compression': apply_compression,
            'ratio': float(ratio),
            'threshold': float(threshold),
            'attack': 10.0,  # мс
            'release': 100.0,  # мс
            'current_compression': float(current_compression),
            'target_compression': float(target_compression),
        }

    def _apply_normalization(self, input_path, output_path, norm_params, output_format):
        """
        Применение нормализации к аудиофайлу

        Args:
            input_path: Путь к исходному файлу
            output_path: Путь для сохранения нормализованного файла
            norm_params: Параметры нормализации
            output_format: Формат выходного файла

        Returns:
            bool: True в случае успеха
        """
        # Загружаем аудио
        input_path = Path(input_path)
        output_path = Path(output_path)

        try:
            # Загружаем аудио с помощью pydub для поддержки разных форматов
            audio_format = input_path.suffix.lstrip('.').lower()
            audio = AudioSegment.from_file(input_path, format=audio_format)

            # Применяем усиление (в дБ)
            gain_db = norm_params['gain_db']
            if gain_db != 0:
                audio = audio.apply_gain(gain_db)

            # Если нужно применить компрессию
            compression_params = norm_params['compression_params']
            if compression_params['apply_compression'] and compression_params['ratio'] > 1.0:
                # Преобразуем в WAV для обработки через TempFile
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_in:
                    temp_in_path = temp_in.name
                    audio.export(temp_in_path, format='wav')

                # Создаем еще один временный файл для выхода
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_out:
                    temp_out_path = temp_out.name

                try:
                    # Применяем компрессию с использованием pydub
                    self._apply_compression(
                        temp_in_path, temp_out_path, compression_params
                    )

                    # Загружаем обработанный файл
                    audio = AudioSegment.from_file(temp_out_path, format='wav')

                finally:
                    # Удаляем временные файлы
                    if os.path.exists(temp_in_path):
                        os.unlink(temp_in_path)
                    if os.path.exists(temp_out_path):
                        os.unlink(temp_out_path)

            # Экспортируем в нужный формат
            export_format = output_format.lower()
            export_params = self._get_export_params(export_format)

            audio.export(
                output_path,
                format=export_format,
                **export_params
            )

            return True

        except Exception as e:
            logger.error(f"Ошибка при применении нормализации: {e}")
            raise e

    def _apply_compression(self, input_path, output_path, params):
        """
        Применение компрессии к аудиофайлу

        Args:
            input_path: Путь к исходному файлу
            output_path: Путь для сохранения обработанного файла
            params: Параметры компрессии

        Returns:
            bool: True в случае успеха
        """
        # Загружаем аудио с помощью soundfile
        y, sr = sf.read(input_path)

        # Параметры компрессии
        threshold = params['threshold']
        ratio = params['ratio']
        attack_ms = params['attack']
        release_ms = params['release']

        # Преобразуем мс во фреймы
        attack_frames = int(attack_ms * sr / 1000)
        release_frames = int(release_ms * sr / 1000)

        # Реализация простой компрессии в амплитудной области
        # Преобразуем threshold из дБ в линейную амплитуду
        threshold_amp = 10 ** (threshold / 20)

        # Создаем огибающую сигнала
        envelope = np.abs(y)

        # Применяем сглаживание для огибающей (простой алгоритм)
        smoothed_envelope = np.zeros_like(envelope)
        for i in range(len(y)):
            if i == 0:
                smoothed_envelope[i] = envelope[i]
            else:
                # Если текущая амплитуда больше предыдущей - быстрая атака
                if envelope[i] > smoothed_envelope[i - 1]:
                    smoothed_envelope[i] = envelope[i] * (1 - np.exp(-1 / attack_frames)) + smoothed_envelope[
                        i - 1] * np.exp(-1 / attack_frames)
                # Если меньше - медленное затухание
                else:
                    smoothed_envelope[i] = envelope[i] * (1 - np.exp(-1 / release_frames)) + smoothed_envelope[
                        i - 1] * np.exp(-1 / release_frames)

        # Рассчитываем коэффициент компрессии для каждого сэмпла
        gain_reduction = np.ones_like(smoothed_envelope)

        mask = smoothed_envelope > threshold_amp
        
        # Расчет коэффициента снижения усиления
        if np.any(mask):  # Проверяем, есть ли хоть один True в маске
            above_threshold = smoothed_envelope[mask] / threshold_amp
            gain_reduction[mask] = threshold_amp * (above_threshold ** (1/ratio - 1))

        # Применяем компрессию
        y_compressed = y * gain_reduction

        # Сохраняем результат
        sf.write(output_path, y_compressed, sr)

        return True

    def _get_export_params(self, format_name):
        """
        Получение параметров экспорта для разных форматов

        Args:
            format_name: Имя формата (wav, mp3, flac, ogg)

        Returns:
            dict: Параметры экспорта
        """
        if format_name == 'mp3':
            return {
                'bitrate': '320k',
                'codec': 'libmp3lame',
            }
        elif format_name == 'flac':
            return {
                'codec': 'flac',
                'compression_level': 5,
            }
        elif format_name == 'ogg':
            return {
                'codec': 'libvorbis',
                'quality': '6',  # От 0 до 10
            }
        elif format_name == 'wav':
            return {
                'codec': 'pcm_s16le',  # 16-bit PCM
            }
        else:
            return {}

    def normalize_all(self, target_lufs=None, target_peak=None, genre=None,
                      dynamic_range=None, output_format='wav', output_dir=None,
                      parallel=True):
        """
        Нормализация всех треков

        Args:
            target_lufs: Целевой уровень LUFS
            target_peak: Целевой пиковый уровень в dB
            genre: Жанр треков (для определения параметров нормализации)
            dynamic_range: Целевой динамический диапазон
            output_format: Формат выходных файлов
            output_dir: Директория для сохранения нормализованных файлов
            parallel: Использовать параллельную обработку

        Returns:
            list: Результаты нормализации для всех треков
        """
        results = []
        audio_files = self.analyzer.audio_files

        if not audio_files:
            logger.warning("Нет файлов для нормализации")
            return []

        output_dir = Path(output_dir or self.output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)

        start_time = time.time()

        try:
            if parallel and len(audio_files) > 1:
                with concurrent.futures.ThreadPoolExecutor(max_workers=config.MAX_WORKERS) as executor:
                    # Создаем задачи нормализации
                    future_to_file = {}
                    for file_path in audio_files:
                        future = executor.submit(
                            self.normalize_track,
                            file_path=file_path,
                            target_lufs=target_lufs,
                            target_peak=target_peak,
                            genre=genre,
                            dynamic_range=dynamic_range,
                            output_format=output_format,
                            output_dir=output_dir
                        )
                        future_to_file[future] = file_path

                    # Обрабатываем результаты с прогресс-баром
                    for future in tqdm(
                            concurrent.futures.as_completed(future_to_file),
                            total=len(audio_files),
                            desc="Нормализация аудиофайлов"
                    ):
                        file_path = future_to_file[future]
                        try:
                            result = future.result()
                            results.append(result)
                        except Exception as e:
                            logger.error(f"Ошибка при нормализации {file_path}: {e}")
                            results.append({
                                'file': file_path.name,
                                'file_path': str(file_path),
                                'normalization': {
                                    'success': False,
                                    'error': str(e)
                                }
                            })
            else:
                # Последовательная нормализация с прогресс-баром
                for file_path in tqdm(audio_files, desc="Нормализация аудиофайлов"):
                    try:
                        result = self.normalize_track(
                            file_path=file_path,
                            target_lufs=target_lufs,
                            target_peak=target_peak,
                            genre=genre,
                            dynamic_range=dynamic_range,
                            output_format=output_format,
                            output_dir=output_dir
                        )
                        results.append(result)
                    except Exception as e:
                        logger.error(f"Ошибка при нормализации {file_path}: {e}")
                        results.append({
                            'file': file_path.name,
                            'file_path': str(file_path),
                            'normalization': {
                                'success': False,
                                'error': str(e)
                            }
                        })

        except Exception as e:
            logger.error(f"Ошибка при выполнении нормализации: {e}")

        elapsed_time = time.time() - start_time
        logger.info(f"Нормализация завершена за {elapsed_time:.2f} сек. Обработано {len(results)} файлов.")

        return results