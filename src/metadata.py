"""
Модуль для извлечения метаданных из аудиофайлов (BPM, тональность, жанр)
"""
import os
import logging
from pathlib import Path
import tempfile

import librosa
import numpy as np
# Пробуем импортировать essentia, но готовы работать и без неё
try:
    import essentia.standard as es
    HAVE_ESSENTIA = True
except ImportError:
    HAVE_ESSENTIA = False
    print("Библиотека Essentia не установлена. Будет использоваться упрощенное определение BPM и тональности.")

import mutagen
from pydub import AudioSegment

import config

logger = logging.getLogger(__name__)


class MetadataExtractor:
    """Класс для извлечения метаданных из аудиофайлов"""
    
    def __init__(self):
        """Инициализация экстрактора метаданных"""
        pass
    
    def extract_metadata(self, file_path, extract_bpm=True, extract_key=True, 
                         extract_tags=True, extract_genre=True):
        """
        Извлечение метаданных из аудиофайла
        
        Args:
            file_path: Путь к аудиофайлу
            extract_bpm: Извлекать ли BPM
            extract_key: Извлекать ли тональность
            extract_tags: Извлекать ли теги из файла
            extract_genre: Пытаться ли определить жанр
            
        Returns:
            dict: Словарь с извлеченными метаданными
        """
        file_path = Path(file_path)
        if not file_path.exists():
            logger.error(f"Файл не найден: {file_path}")
            return {}
            
        metadata = {
            'filename': file_path.name,
            'file_path': str(file_path),
            'file_format': file_path.suffix.lower(),
            'file_size': file_path.stat().st_size,
        }
        
        # Извлечение тегов из файла
        if extract_tags:
            file_tags = self._extract_file_tags(file_path)
            if file_tags:
                metadata.update(file_tags)
        
        # Загрузка аудио данных для анализа
        try:
            y, sr = self._load_audio(file_path)
            metadata['sample_rate'] = sr
            metadata['duration'] = librosa.get_duration(y=y, sr=sr)
            
            # Извлечение BPM
            if extract_bpm:
                bpm = self._extract_bpm(y, sr)
                metadata['bpm'] = bpm
            
            # Извлечение тональности
            if extract_key:
                key = self._extract_key(y, sr)
                metadata['key'] = key
            
            # Определение жанра (если запрошено и не найдено в тегах)
            if extract_genre and 'genre' not in metadata:
                genre = self._detect_genre(y, sr)
                if genre:
                    metadata['detected_genre'] = genre
            
        except Exception as e:
            logger.error(f"Ошибка при извлечении аудио-метаданных из {file_path}: {e}")
        
        return metadata
    
    def _load_audio(self, file_path, offset=0.0, duration=None):
        """
        Загрузка аудио данных для анализа
        
        Args:
            file_path: Путь к аудиофайлу
            offset: Смещение в секундах
            duration: Длительность в секундах (None для загрузки всего файла)
            
        Returns:
            tuple: (y, sr) - аудио данные и частота дискретизации
        """
        try:
            # Сначала пробуем загрузить с помощью librosa
            y, sr = librosa.load(file_path, sr=config.ANALYSIS_SAMPLE_RATE, 
                                 offset=offset, duration=duration)
            return y, sr
        except Exception as e:
            logger.warning(f"Не удалось загрузить файл с помощью librosa: {e}")
            
            # Если не получилось, пробуем через pydub + tempfile
            try:
                # Определяем формат для pydub
                format_map = {
                    '.mp3': 'mp3',
                    '.wav': 'wav',
                    '.flac': 'flac',
                    '.ogg': 'ogg',
                    '.m4a': 'mp4',
                    '.aac': 'aac'
                }
                
                file_ext = Path(file_path).suffix.lower()
                audio_format = format_map.get(file_ext, 'mp3')
                
                # Загружаем аудио с помощью pydub
                audio = AudioSegment.from_file(file_path, format=audio_format)
                
                # Применяем offset и duration если указаны
                if offset > 0:
                    audio = audio[int(offset * 1000):]
                if duration:
                    audio = audio[:int(duration * 1000)]
                
                # Конвертируем в моно если нужно
                if audio.channels > 1:
                    audio = audio.set_channels(1)
                
                # Устанавливаем частоту дискретизации
                if audio.frame_rate != config.ANALYSIS_SAMPLE_RATE:
                    audio = audio.set_frame_rate(config.ANALYSIS_SAMPLE_RATE)
                
                # Сохраняем во временный WAV файл
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                    temp_path = temp_file.name
                    audio.export(temp_path, format="wav")
                
                try:
                    # Загружаем временный файл с помощью librosa
                    y, sr = librosa.load(temp_path, sr=config.ANALYSIS_SAMPLE_RATE, mono=True)
                    return y, sr
                finally:
                    # Удаляем временный файл
                    if os.path.exists(temp_path):
                        os.unlink(temp_path)
                        
            except Exception as e2:
                logger.error(f"Не удалось загрузить аудио с помощью pydub: {e2}")
                raise e2
    
    def _extract_file_tags(self, file_path):
        """
        Извлечение тегов из файла метаданных
        
        Args:
            file_path: Путь к аудиофайлу
            
        Returns:
            dict: Словарь с тегами
        """
        try:
            audio_file = mutagen.File(file_path)
            
            if audio_file is None:
                return {}
            
            tags = {}
            
            # Извлекаем базовые теги
            for key in ['title', 'artist', 'album', 'date', 'genre', 'bpm']:
                if key in audio_file:
                    tags[key] = audio_file[key][0]
            
            # Обработка специфичных форматов
            if isinstance(audio_file, mutagen.mp3.MP3):
                # ID3 теги для MP3
                id3 = audio_file.tags
                if id3:
                    mapping = {
                        'TIT2': 'title',
                        'TPE1': 'artist',
                        'TALB': 'album',
                        'TDRC': 'date',
                        'TCON': 'genre',
                        'TBPM': 'bpm',
                        'TKEY': 'key',
                    }
                    
                    for id3_key, tag_key in mapping.items():
                        if id3_key in id3:
                            tags[tag_key] = str(id3[id3_key])
            
            # Преобразуем BPM в число, если это возможно
            if 'bpm' in tags:
                try:
                    tags['bpm'] = float(tags['bpm'])
                except (ValueError, TypeError):
                    pass
            
            return tags
        
        except Exception as e:
            logger.warning(f"Не удалось извлечь теги из {file_path}: {e}")
            return {}
    
    def _extract_bpm(self, y, sr):
        """
        Извлечение BPM (темпа) из аудио
        
        Args:
            y: Аудио данные
            sr: Частота дискретизации
            
        Returns:
            float: Значение BPM
        """
        try:
            # Если доступна Essentia - используем её для более точного определения BPM
            if HAVE_ESSENTIA:
                # Конвертируем из numpy float32 в список float для Essentia
                audio = np.array(y, dtype=np.float32).tolist()
                
                rhythm_extractor = es.RhythmExtractor2013(method="multifeature")
                bpm, _, _, _, _ = rhythm_extractor(audio)
                
                # Верификация с помощью librosa
                tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
                
                # Убедимся, что темп - это скаляр, а не массив
                if isinstance(tempo, np.ndarray):
                    tempo = float(tempo.item() if tempo.size == 1 else tempo[0])
                
                # Если результаты сильно расходятся, возвращаем среднее
                if abs(bpm - tempo) > 10:
                    return float((bpm + tempo) / 2)
                
                return float(bpm)
            else:
                # Если Essentia не доступна, используем только librosa
                tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
                
                # Убедимся, что темп - это скаляр, а не массив
                if isinstance(tempo, np.ndarray):
                    tempo = float(tempo.item() if tempo.size == 1 else tempo[0])
                
                return float(tempo)
            
        except Exception as e:
            logger.warning(f"Ошибка при определении BPM: {e}")
            
            # Fallback на librosa
            try:
                tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
                
                # Убедимся, что темп - это скаляр, а не массив
                if isinstance(tempo, np.ndarray):
                    tempo = float(tempo.item() if tempo.size == 1 else tempo[0])
                
                return float(tempo)
            except Exception as e2:
                logger.error(f"Не удалось определить BPM с помощью librosa: {e2}")
                return 120.0  # Возвращаем разумное значение по умолчанию вместо None
    
    def _extract_key(self, y, sr):
        """
        Извлечение музыкальной тональности
        
        Args:
            y: Аудио данные
            sr: Частота дискретизации
            
        Returns:
            str: Тональность в формате 'C Major' или 'A Minor'
        """
        try:
            # Если доступна Essentia - используем её для определения тональности
            if HAVE_ESSENTIA:
                audio = np.array(y, dtype=np.float32).tolist()
                
                key_extractor = es.KeyExtractor()
                key, scale, strength = key_extractor(audio)
                
                # Форматируем результат
                key_name = f"{key} {scale}"
                
                return key_name
            else:
                # Если Essentia не доступна, используем только librosa
                return self._extract_key_librosa(y, sr)
            
        except Exception as e:
            logger.warning(f"Ошибка при определении тональности: {e}")
            
            # Fallback на librosa
            try:
                return self._extract_key_librosa(y, sr)
            except Exception as e2:
                logger.error(f"Не удалось определить тональность: {e2}")
                return "C Major"  # Возвращаем значение по умолчанию вместо None
    
    def _extract_key_librosa(self, y, sr):
        """
        Извлечение тональности с помощью librosa
        
        Args:
            y: Аудио данные
            sr: Частота дискретизации
            
        Returns:
            str: Тональность в формате 'C Major' или 'A Minor'
        """
        try:
            # Извлекаем хроматограмму
            chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
            
            # Сумма по времени для получения общего профиля тональности
            chroma_sum = np.sum(chroma, axis=1)
            
            # Находим ноту с максимальной энергией
            max_note_idx = np.argmax(chroma_sum)
            
            # Определяем тональность (мажор/минор) на основе профиля
            notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
            
            # Упрощенное определение мажор/минор
            major_profile = librosa.feature.tonnetz(y=y, sr=sr)[5]
            is_major = np.mean(major_profile) > 0
            
            scale = "Major" if is_major else "Minor"
            key_name = f"{notes[max_note_idx]} {scale}"
            
            return key_name
        except Exception as e:
            logger.error(f"Не удалось определить тональность с помощью librosa: {e}")
            return "C Major"  # Возвращаем значение по умолчанию
    
    def _detect_genre(self, y, sr):
        """
        Простое определение жанра на основе аудио характеристик
        
        Args:
            y: Аудио данные
            sr: Частота дискретизации
            
        Returns:
            str: Наиболее вероятный жанр
        """
        try:
            # Извлекаем различные характеристики
            
            # 1. Спектральный центроид (яркость звука)
            cent = librosa.feature.spectral_centroid(y=y, sr=sr).mean()
            if isinstance(cent, np.ndarray):
                cent = float(cent.item() if cent.size == 1 else cent[0])
            
            # 2. Спектральный контраст (разница между пиками и долинами в спектре)
            contrast = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr))
            if isinstance(contrast, np.ndarray):
                contrast = float(contrast.item() if contrast.size == 1 else contrast[0])
            
            # 3. Темп
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
            if isinstance(tempo, np.ndarray):
                tempo = float(tempo.item() if tempo.size == 1 else tempo[0])
            
            # 4. RMS энергия
            rms = librosa.feature.rms(y=y).mean()
            if isinstance(rms, np.ndarray):
                rms = float(rms.item() if rms.size == 1 else rms[0])
            
            # 5. Низкочастотная энергия
            stft = np.abs(librosa.stft(y))
            low_freq_energy = np.sum(stft[:10, :]) / np.sum(stft) if np.sum(stft) > 0 else 0
            
            # Упрощенные правила классификации
            if tempo > 160 and rms > 0.15:
                return "dnb"
            elif tempo > 135 and tempo < 145 and contrast > 0.2:
                return "techno"
            elif tempo > 120 and tempo < 130 and contrast > 0.15:
                return "house"
            elif tempo > 135 and tempo < 150 and cent > 2000:
                return "trance"
            elif tempo > 130 and low_freq_energy > 0.5 and rms > 0.2:
                return "dubstep"
            elif tempo > 85 and tempo < 105 and low_freq_energy > 0.4:
                return "hiphop"
            elif tempo > 100 and tempo < 130 and contrast < 0.2:
                return "pop"
            elif contrast > 0.25 and cent > 1800:
                return "rock"
            elif tempo < 100 and contrast > 0.15 and cent < 1500:
                return "jazz"
            elif cent < 1000 and rms < 0.1:
                return "ambient"
            elif rms < 0.08:
                return "classical"
            else:
                return "default"
                
        except Exception as e:
            logger.warning(f"Ошибка при определении жанра: {e}")
            return "default"