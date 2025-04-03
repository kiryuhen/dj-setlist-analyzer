"""
Вспомогательные функции для DJ сетлист-анализатора
"""
import os
import json
import logging
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

import config


def setup_logging(log_level=logging.INFO):
    """
    Настройка логирования
    
    Args:
        log_level: Уровень логирования
        
    Returns:
        logging.Logger: Настроенный логгер
    """
    # Создаем директорию для логов, если она не существует
    log_dir = Path('logs')
    log_dir.mkdir(exist_ok=True)
    
    # Создаем имя файла лога с текущей датой и временем
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = log_dir / f'dj_analyzer_{timestamp}.log'
    
    # Настраиваем формат логов
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Настраиваем корневой логгер
    logging.basicConfig(
        level=log_level,
        format=log_format,
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()  # Вывод в консоль
        ]
    )
    
    # Уменьшаем уровень логирования для сторонних библиотек
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('numba').setLevel(logging.WARNING)
    logging.getLogger('librosa').setLevel(logging.WARNING)
    
    return logging.getLogger('dj_analyzer')


def format_time(seconds):
    """
    Форматирование времени в формат MM:SS
    
    Args:
        seconds: Время в секундах
        
    Returns:
        str: Отформатированное время
    """
    if seconds is None:
        seconds = 0
    minutes = int(seconds // 60)
    remaining_seconds = int(seconds % 60)
    return f"{minutes}:{remaining_seconds:02d}"


def save_results_to_csv(results, output_path=None):
    """
    Сохранение результатов анализа в CSV-файл
    
    Args:
        results: Результаты анализа
        output_path: Путь для сохранения CSV-файла (по умолчанию 'results.csv')
        
    Returns:
        str: Путь к сохраненному файлу
    """
    if not output_path:
        output_path = 'results.csv'
    
    # Преобразуем в DataFrame
    df = pd.DataFrame(results)
    
    # Фильтруем колонки для CSV (исключаем сложные вложенные структуры)
    simple_cols = []
    for col in df.columns:
        # Проверяем, что в колонке нет сложных типов данных (списки, словари)
        if df[col].dtype == 'object':
            has_complex = any(isinstance(x, (dict, list)) for x in df[col].dropna())
            if not has_complex:
                simple_cols.append(col)
        else:
            simple_cols.append(col)
    
    # Сохраняем только простые колонки
    df[simple_cols].to_csv(output_path, index=False)
    
    return output_path


def save_results_to_json(results, output_path=None):
    """
    Сохранение результатов анализа в JSON-файл
    
    Args:
        results: Результаты анализа
        output_path: Путь для сохранения JSON-файла (по умолчанию 'results.json')
        
    Returns:
        str: Путь к сохраненному файлу
    """
    if not output_path:
        output_path = 'results.json'
    
    # Сериализуем в JSON с отступами для читаемости
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    
    return output_path


def create_loudness_report(results, output_dir=None):
    """
    Создание отчета о громкости треков
    
    Args:
        results: Результаты анализа
        output_dir: Директория для сохранения отчета
        
    Returns:
        tuple: (DataFrame с результатами, путь к сохраненному отчету)
    """
    if not output_dir:
        output_dir = Path('reports')
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Создаем DataFrame для отчета
    loudness_data = []
    
    for track in results:
        if 'error' in track:
            continue
            
        track_data = {
            'file': track.get('file', ''),
            'lufs': track.get('lufs_integrated', 0),
            'peak_db': track.get('peak_db', 0),
            'dynamic_range': track.get('dynamic_range', 0),
            'bpm': track.get('bpm', 0),
            'key': track.get('key', ''),
            'duration': track.get('duration', 0),
            'genre': track.get('genre', track.get('detected_genre', '')),
        }
        
        # Убедимся, что все значения валидны (преобразуем None в 0 или '')
        for k, v in track_data.items():
            if v is None:
                track_data[k] = 0 if k in ['lufs', 'peak_db', 'dynamic_range', 'bpm', 'duration'] else ''
        
        # Если была проведена нормализация, добавляем параметры
        if 'normalization' in track:
            norm = track['normalization']
            norm_data = {
                'gain_db': norm.get('gain_db', 0),
                'target_lufs': norm.get('target_lufs', 0),
                'expected_output_lufs': norm.get('expected_output_lufs', 0),
            }
            
            # Убедимся, что все значения валидны
            for k, v in norm_data.items():
                if v is None:
                    norm_data[k] = 0
                    
            track_data.update(norm_data)
        
        loudness_data.append(track_data)
    
    # Создаем DataFrame
    df = pd.DataFrame(loudness_data)
    
    # Если DataFrame пустой, возвращаем пустой отчет
    if df.empty:
        return df, None
    
    # Сортируем по громкости
    if 'lufs' in df.columns:
        df = df.sort_values('lufs', ascending=False)
    
    # Сохраняем отчет в CSV
    csv_path = output_dir / 'loudness_report.csv'
    df.to_csv(csv_path, index=False)
    
    # Создаем графики
    try:
        # Проверки на наличие данных для графиков
        if df.empty or len(df) < 1:
            return df, csv_path
            
        # График интегральной громкости
        if 'lufs' in df.columns and 'file' in df.columns:
            plt.figure(figsize=(12, 6))
            plt.barh(df['file'], df['lufs'], color='skyblue')
            plt.axvline(x=-14, color='red', linestyle='--', label='Streaming Target (-14 LUFS)')
            plt.xlabel('LUFS')
            plt.ylabel('Track')
            plt.title('Integrated Loudness (LUFS)')
            plt.grid(axis='x', linestyle='--', alpha=0.7)
            plt.legend()
            plt.tight_layout()
            plt.savefig(output_dir / 'loudness_lufs.png')
            plt.close()
        
        # График пиковых значений
        if 'peak_db' in df.columns and 'file' in df.columns:
            plt.figure(figsize=(12, 6))
            plt.barh(df['file'], df['peak_db'], color='lightgreen')
            plt.axvline(x=-1, color='red', linestyle='--', label='Standard Peak Limit (-1 dB)')
            plt.xlabel('dB')
            plt.ylabel('Track')
            plt.title('Peak Levels (dB)')
            plt.grid(axis='x', linestyle='--', alpha=0.7)
            plt.legend()
            plt.tight_layout()
            plt.savefig(output_dir / 'loudness_peaks.png')
            plt.close()
        
        # График динамического диапазона
        if 'dynamic_range' in df.columns and 'file' in df.columns:
            plt.figure(figsize=(12, 6))
            plt.barh(df['file'], df['dynamic_range'], color='salmon')
            plt.xlabel('dB')
            plt.ylabel('Track')
            plt.title('Dynamic Range (dB)')
            plt.grid(axis='x', linestyle='--', alpha=0.7)
            plt.tight_layout()
            plt.savefig(output_dir / 'dynamic_range.png')
            plt.close()
        
        # График BPM
        if 'bpm' in df.columns and 'file' in df.columns:
            plt.figure(figsize=(12, 6))
            plt.barh(df['file'], df['bpm'], color='mediumpurple')
            plt.xlabel('BPM')
            plt.ylabel('Track')
            plt.title('Tempo (BPM)')
            plt.grid(axis='x', linestyle='--', alpha=0.7)
            plt.tight_layout()
            plt.savefig(output_dir / 'bpm.png')
            plt.close()
        
        # Создаем HTML-отчет
        html_path = output_dir / 'loudness_report.html'
        
        # Вычисляем статистику, защищаясь от None и пустых значений
        avg_lufs = df['lufs'].mean() if 'lufs' in df.columns and not df['lufs'].empty else 0
        avg_bpm = df['bpm'].mean() if 'bpm' in df.columns and not df['bpm'].empty else 0
        avg_dr = df['dynamic_range'].mean() if 'dynamic_range' in df.columns and not df['dynamic_range'].empty else 0
        
        # Определяем самый громкий и тихий трек
        loudest_track = ""
        loudest_lufs = 0
        quietest_track = ""
        quietest_lufs = 0
        
        if 'lufs' in df.columns and not df['lufs'].empty and 'file' in df.columns:
            sorted_df = df.sort_values('lufs', ascending=False)
            if len(sorted_df) > 0:
                loudest_track = sorted_df.iloc[0]['file']
                loudest_lufs = sorted_df.iloc[0]['lufs']
                quietest_track = sorted_df.iloc[-1]['file']
                quietest_lufs = sorted_df.iloc[-1]['lufs']

        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>DJ Setlist Analysis Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1, h2 {{ color: #333; }}
                table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                tr:nth-child(even) {{ background-color: #f9f9f9; }}
                img {{ max-width: 100%; height: auto; margin: 10px 0; }}
                .stats {{ background-color: #f0f8ff; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
            </style>
        </head>
        <body>
            <h1>DJ Setlist Analysis Report</h1>
            <div class="stats">
                <h2>Summary Statistics</h2>
                <p>Total Tracks: {len(df)}</p>
                <p>Average LUFS: {avg_lufs:.2f} LUFS</p>
                <p>Loudest Track: {loudest_track} ({loudest_lufs:.2f} LUFS)</p>
                <p>Quietest Track: {quietest_track} ({quietest_lufs:.2f} LUFS)</p>
                <p>Average BPM: {avg_bpm:.1f}</p>
                <p>Average Dynamic Range: {avg_dr:.1f} dB</p>
            </div>
            
            <h2>Loudness Graphs</h2>
            <img src="loudness_lufs.png" alt="LUFS Graph">
            <img src="loudness_peaks.png" alt="Peak Levels Graph">
            <img src="dynamic_range.png" alt="Dynamic Range Graph">
            <img src="bpm.png" alt="BPM Graph">
            
            <h2>Track Details</h2>
            <table>
                <tr>
                    <th>File</th>
                    <th>LUFS</th>
                    <th>Peak (dB)</th>
                    <th>Dynamic Range (dB)</th>
                    <th>BPM</th>
                    <th>Key</th>
                    <th>Duration</th>
                    <th>Genre</th>
                </tr>
        """
        
        for _, row in df.iterrows():
            duration_fmt = format_time(row.get('duration', 0))
            html_content += f"""
                <tr>
                    <td>{row.get('file', '')}</td>
                    <td>{row.get('lufs', 0):.2f}</td>
                    <td>{row.get('peak_db', 0):.2f}</td>
                    <td>{row.get('dynamic_range', 0):.2f}</td>
                    <td>{row.get('bpm', 0):.1f}</td>
                    <td>{row.get('key', '')}</td>
                    <td>{duration_fmt}</td>
                    <td>{row.get('genre', '')}</td>
                </tr>
            """
        
        html_content += """
            </table>
        </body>
        </html>
        """
        
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        # Пытаемся автоматически открыть отчет
        try:
            import os
            import webbrowser
            absolute_path = 'file://' + os.path.abspath(str(html_path))
            webbrowser.open(absolute_path)
            print(f"Открываем HTML-отчет в браузере: {absolute_path}")
        except Exception as e:
            print(f"Не удалось автоматически открыть отчет: {e}")
            print(f"Вы можете открыть отчет вручную: {html_path}")
        
        return df, html_path
        
    except Exception as e:
        logging.error(f"Ошибка при создании отчета: {e}")
        return df, csv_path


def create_setlist_recommendations(results):
    """
    Создание рекомендаций по сетлисту на основе анализа
    
    Args:
        results: Результаты анализа
        
    Returns:
        dict: Рекомендации по сетлисту
    """
    tracks = []
    
    for track in results:
        if 'error' in track:
            continue
            
        # Защита от None-значений
        bpm = track.get('bpm', 0)
        if bpm is None:
            bpm = 0
            
        rms = track.get('rms', 0)
        if rms is None:
            rms = 0
            
        bass_to_highs_ratio = track.get('bass_to_highs_ratio', 1)
        if bass_to_highs_ratio is None:
            bass_to_highs_ratio = 1
        
        track_data = {
            'file': track.get('file', ''),
            'lufs': track.get('lufs_integrated', 0) or 0,
            'bpm': bpm,
            'key': track.get('key', '') or '',
            'duration': track.get('duration', 0) or 0,
            'genre': track.get('genre', track.get('detected_genre', '')) or '',
            'energy': bass_to_highs_ratio * rms,
            'intro_mix_recommendation': track.get('intro_mix_recommendation', 8) or 8,
            'outro_mix_recommendation': track.get('outro_mix_recommendation', 8) or 8,
            'first_drop_time': track.get('first_drop_time', 0) or 0,
        }
        
        tracks.append(track_data)
    
    # Если треков нет, возвращаем пустые рекомендации
    if not tracks:
        return {
            'optimal_track_order': {'by_bpm': [], 'by_energy': []},
            'suggested_intros': [],
            'suggested_peaks': [],
            'transition_recommendations': []
        }
    
    # Сортируем треки по BPM для создания плавного перехода темпа
    tracks_by_bpm = sorted(tracks, key=lambda x: x['bpm'])
    
    # Сортируем треки по энергии для создания динамики сета
    tracks_by_energy = sorted(tracks, key=lambda x: x['energy'])
    
    # Выделяем треки для начала, середины и конца сета
    intro_tracks = tracks_by_energy[:max(1, len(tracks_by_energy)//4)]
    peak_tracks = tracks_by_energy[-max(1, len(tracks_by_energy)//3):]
    
    # Создаем рекомендации
    recommendations = {
        'optimal_track_order': {
            'by_bpm': [{'file': t['file'], 'bpm': t['bpm']} for t in tracks_by_bpm],
            'by_energy': [{'file': t['file'], 'energy_rating': i+1} for i, t in enumerate(tracks_by_energy)],
        },
        'suggested_intros': [{'file': t['file'], 'bpm': t['bpm']} for t in intro_tracks],
        'suggested_peaks': [{'file': t['file'], 'bpm': t['bpm']} for t in peak_tracks],
        'transition_recommendations': []
    }
    
    # Создаем рекомендации по переходам между треками
    for i in range(len(tracks_by_bpm) - 1):
        current = tracks_by_bpm[i]
        next_track = tracks_by_bpm[i+1]
        
        # Вычисляем разницу в BPM
        bpm_diff = next_track['bpm'] - current['bpm']
        
        # Определяем тип перехода
        if abs(bpm_diff) < 3:
            transition_type = "Прямой переход (близкие BPM)"
            mix_time = min(current['outro_mix_recommendation'], next_track['intro_mix_recommendation'])
        elif abs(bpm_diff) < 10:
            transition_type = "Постепенный переход с изменением темпа"
            mix_time = max(current['outro_mix_recommendation'], next_track['intro_mix_recommendation'])
        else:
            transition_type = "Резкий переход или использование эффектов"
            mix_time = 4  # Короткий переход
        
        recommendations['transition_recommendations'].append({
            'from_track': current['file'],
            'to_track': next_track['file'],
            'bpm_diff': bpm_diff,
            'transition_type': transition_type,
            'suggested_mix_time': mix_time,
        })
    
    return recommendations


def print_analysis_summary(results):
    """
    Вывод краткой сводки результатов анализа в консоль
    
    Args:
        results: Результаты анализа треков
    """
    if not results:
        print("\n*** ИТОГИ АНАЛИЗА: Нет данных для отображения ***")
        return
    
    print("\n" + "="*80)
    print("                            ИТОГИ АНАЛИЗА СЕТЛИСТА")
    print("="*80)
    
    # Статистика по трекам
    total_tracks = len(results)
    tracks_with_bpm = sum(1 for track in results if track.get('bpm') not in [None, 0])
    tracks_with_key = sum(1 for track in results if track.get('key'))
    tracks_with_error = sum(1 for track in results if 'error' in track)
    
    # Общая длительность
    total_duration = sum(track.get('duration', 0) or 0 for track in results)
    hours = int(total_duration // 3600)
    minutes = int((total_duration % 3600) // 60)
    seconds = int(total_duration % 60)
    
    # LUFS статистика
    lufs_values = [track.get('lufs_integrated', 0) for track in results if track.get('lufs_integrated') is not None]
    avg_lufs = sum(lufs_values) / len(lufs_values) if lufs_values else 0
    min_lufs = min(lufs_values) if lufs_values else 0
    max_lufs = max(lufs_values) if lufs_values else 0
    lufs_range = max_lufs - min_lufs
    
    # BPM статистика
    bpm_values = [track.get('bpm', 0) for track in results if track.get('bpm') is not None]
    avg_bpm = sum(bpm_values) / len(bpm_values) if bpm_values else 0
    min_bpm = min(bpm_values) if bpm_values else 0
    max_bpm = max(bpm_values) if bpm_values else 0
    bpm_range = max_bpm - min_bpm
    
    # Вывод основной статистики
    print(f"Всего треков: {total_tracks}")
    print(f"Успешно проанализировано: {total_tracks - tracks_with_error}")
    print(f"Общая длительность: {hours}ч {minutes}м {seconds}с")
    print(f"Треки с определенным BPM: {tracks_with_bpm}/{total_tracks}")
    print(f"Треки с определенной тональностью: {tracks_with_key}/{total_tracks}")
    print("-"*80)
    
    # Вывод статистики громкости
    print("СТАТИСТИКА ГРОМКОСТИ:")
    print(f"  Средняя громкость: {avg_lufs:.2f} LUFS")
    print(f"  Диапазон громкости: от {min_lufs:.2f} до {max_lufs:.2f} LUFS (размах {lufs_range:.2f} dB)")
    
    # Треки, требующие нормализации
    if lufs_values:
        threshold = -14.0  # Стандарт для стриминга
        loud_tracks = [r for r in results if r.get('lufs_integrated', 0) and r.get('lufs_integrated', 0) > threshold]
        quiet_tracks = [r for r in results if r.get('lufs_integrated', 0) and r.get('lufs_integrated', 0) < threshold - 3]
        
        if loud_tracks:
            print("\nСлишком громкие треки (требуют понижения громкости):")
            for track in sorted(loud_tracks, key=lambda x: x.get('lufs_integrated', 0), reverse=True)[:3]:
                print(f"  {track.get('file', '')}: {track.get('lufs_integrated', 0):.2f} LUFS")
            if len(loud_tracks) > 3:
                print(f"  ... и еще {len(loud_tracks) - 3} треков")
                
        if quiet_tracks:
            print("\nСлишком тихие треки (требуют повышения громкости):")
            for track in sorted(quiet_tracks, key=lambda x: x.get('lufs_integrated', 0))[:3]:
                print(f"  {track.get('file', '')}: {track.get('lufs_integrated', 0):.2f} LUFS")
            if len(quiet_tracks) > 3:
                print(f"  ... и еще {len(quiet_tracks) - 3} треков")
    
    # Вывод статистики BPM
    if bpm_values:
        print("\nСТАТИСТИКА BPM:")
        print(f"  Средний BPM: {avg_bpm:.1f}")
        print(f"  Диапазон BPM: от {min_bpm:.1f} до {max_bpm:.1f} (размах {bpm_range:.1f})")
        
        # Группировка по BPM диапазонам
        bpm_ranges = {
            "<100": 0,
            "100-120": 0,
            "120-130": 0,
            "130-140": 0,
            ">140": 0
        }
        
        for bpm in bpm_values:
            if bpm < 100:
                bpm_ranges["<100"] += 1
            elif bpm < 120:
                bpm_ranges["100-120"] += 1
            elif bpm < 130:
                bpm_ranges["120-130"] += 1
            elif bpm < 140:
                bpm_ranges["130-140"] += 1
            else:
                bpm_ranges[">140"] += 1
        
        print("  Распределение по BPM:")
        for range_name, count in bpm_ranges.items():
            if count > 0:
                print(f"    {range_name}: {count} треков")
    
    # Распределение треков по жанрам
    genres = {}
    for track in results:
        genre = track.get('genre', track.get('detected_genre', 'Неизвестно'))
        if not genre:
            genre = 'Неизвестно'
        genres[genre] = genres.get(genre, 0) + 1
    
    if genres:
        print("\nРАСПРЕДЕЛЕНИЕ ПО ЖАНРАМ:")
        for genre, count in sorted(genres.items(), key=lambda x: x[1], reverse=True):
            print(f"  {genre}: {count} треков")
    
    # Рекомендации
    print("\nРЕКОМЕНДАЦИИ:")
    if lufs_range > 6:
        print("  • Рекомендуется нормализовать громкость треков для более равномерного звучания")
        target = -14 if avg_lufs < -10 else -8
        print(f"    Целевой уровень: {target} LUFS (используйте команду 'normalize --target-lufs {target}')")
    
    if bpm_range > 20:
        print("  • Рекомендуется учитывать большой разброс BPM при построении сетлиста")
        print("    Изучите рекомендации в файле setlist_recommendations.json")
    
    if tracks_with_error > 0:
        print(f"  • {tracks_with_error} треков не удалось проанализировать полностью.")
        print("    Детали можно найти в логах и в файле результатов.")
    
    print("\nПолные результаты анализа сохранены в файлах JSON и HTML.")
    print("="*80)