"""
Интерфейс командной строки для DJ сетлист-анализатора
"""
import os
import sys
import logging
import click
import json
import webbrowser
from pathlib import Path

import config
from src.analyzer import AudioAnalyzer
from src.normalizer import AudioNormalizer
from src.cache import AnalysisCache
from src.utils import (
    setup_logging, save_results_to_csv, save_results_to_json,
    create_loudness_report, create_setlist_recommendations
)

logger = setup_logging()


@click.group()
@click.version_option(version='1.0.0')
def cli():
    """DJ Setlist Analyzer - инструмент для анализа и нормализации громкости аудиофайлов"""
    pass


@cli.command('analyze')
@click.option('--folder', '-f', default=None,
              help='Путь к папке с аудиофайлами (по умолчанию используется папка setlist)')
@click.option('--output', '-o', default='results.json', help='Путь для сохранения результатов анализа')
@click.option('--format', 'output_format', type=click.Choice(['json', 'csv']), default='json',
              help='Формат вывода (json или csv)')
@click.option('--force', is_flag=True, help='Принудительный анализ, игнорируя кэш')
@click.option('--report', is_flag=True, help='Создать отчет о громкости')
@click.option('--report-dir', default='reports', help='Директория для сохранения отчета')
@click.option('--no-parallel', is_flag=True, help='Отключить параллельную обработку')
def analyze(folder, output, output_format, force, report, report_dir, no_parallel):
    """Анализ аудиофайлов в указанной папке"""
    try:
        folder_path = Path(folder) if folder else Path(config.SETLIST_DIR)

        logger.info(f"Начинаем анализ файлов в папке: {folder_path}")

        # Создаем анализатор
        analyzer = AudioAnalyzer(folder_path=folder_path)

        # Проверяем, найдены ли файлы
        if not analyzer.audio_files:
            logger.error(f"В папке {folder_path} не найдено поддерживаемых аудиофайлов")
            click.echo(f"В папке {folder_path} не найдено поддерживаемых аудиофайлов")
            return

        # Выводим информацию о найденных файлах
        click.echo(f"Найдено {len(analyzer.audio_files)} аудиофайлов в {folder_path}")

        # Выполняем анализ
        results = analyzer.analyze_all(force_analyze=force, parallel=not no_parallel)

        # Сохраняем результаты
        if output_format == 'json':
            file_path = save_results_to_json(results, output)
        else:
            file_path = save_results_to_csv(results, output)

        click.echo(f"Результаты анализа сохранены в {file_path}")

        # Создаем отчет, если запрошено
        if report:
            df, report_path = create_loudness_report(results, report_dir)
            if report_path:
                click.echo(f"Отчет о громкости создан в {report_path}")

                # Открываем отчет в браузере, если это HTML-отчет
                if report_path and str(report_path).endswith('.html'):
                    try:
                        webbrowser.open('file://' + os.path.abspath(report_path))
                    except Exception as e:
                        logger.warning(f"Не удалось открыть отчет в браузере: {e}")

        # Создаем рекомендации по сетлисту
        recommendations = create_setlist_recommendations(results)
        recommendations_path = os.path.join(os.path.dirname(output), 'setlist_recommendations.json')
        with open(recommendations_path, 'w', encoding='utf-8') as f:
            json.dump(recommendations, f, indent=2, ensure_ascii=False)

        click.echo(f"Рекомендации по сетлисту сохранены в {recommendations_path}")

    except Exception as e:
        logger.error(f"Ошибка при анализе файлов: {e}", exc_info=True)
        click.echo(f"Произошла ошибка: {e}")
        sys.exit(1)


@cli.command('normalize')
@click.option('--folder', '-f', default=None,
              help='Путь к папке с аудиофайлами (по умолчанию используется папка setlist)')
@click.option('--output-dir', '-o', default=None, help='Директория для сохранения нормализованных файлов')
@click.option('--target-lufs', type=float, default=None, help='Целевой уровень LUFS')
@click.option('--target-peak', type=float, default=None, help='Целевой пиковый уровень в dB')
@click.option('--genre', type=str, default=None, help='Жанр треков (для определения параметров нормализации)')
@click.option('--dynamic-range', type=click.Choice(['very_wide', 'wide', 'medium', 'compressed', 'very_compressed']),
              default=None, help='Целевой динамический диапазон')
@click.option('--format', 'output_format', type=click.Choice(['wav', 'mp3', 'flac', 'ogg']), default='wav',
              help='Формат выходных файлов')
@click.option('--no-parallel', is_flag=True, help='Отключить параллельную обработку')
@click.option('--report', is_flag=True, help='Создать отчет после нормализации')
def normalize(folder, output_dir, target_lufs, target_peak, genre, dynamic_range,
              output_format, no_parallel, report):
    """Нормализация громкости аудиофайлов"""
    try:
        folder_path = Path(folder) if folder else Path(config.SETLIST_DIR)
        output_dir = Path(output_dir) if output_dir else Path(config.OUTPUT_DIR)

        logger.info(f"Начинаем нормализацию файлов в папке: {folder_path}")

        # Создаем анализатор и нормализатор
        analyzer = AudioAnalyzer(folder_path=folder_path)
        normalizer = AudioNormalizer(analyzer=analyzer)

        # Проверяем, найдены ли файлы
        if not analyzer.audio_files:
            logger.error(f"В папке {folder_path} не найдено поддерживаемых аудиофайлов")
            click.echo(f"В папке {folder_path} не найдено поддерживаемых аудиофайлов")
            return

        # Выводим информацию о найденных файлах
        click.echo(f"Найдено {len(analyzer.audio_files)} аудиофайлов в {folder_path}")
        click.echo(f"Нормализованные файлы будут сохранены в {output_dir}")

        # Параметры нормализации
        params = {}
        if target_lufs is not None:
            params['target_lufs'] = target_lufs
        if target_peak is not None:
            params['target_peak'] = target_peak
        if genre:
            params['genre'] = genre
        if dynamic_range:
            params['dynamic_range'] = dynamic_range

        # Информация о параметрах
        if params:
            click.echo("Параметры нормализации:")
            for key, value in params.items():
                click.echo(f"  {key}: {value}")
        else:
            click.echo("Используются параметры нормализации по умолчанию для каждого трека")

        # Выполняем нормализацию
        results = normalizer.normalize_all(
            **params,
            output_format=output_format,
            output_dir=output_dir,
            parallel=not no_parallel
        )

        # Сохраняем результаты
        results_path = output_dir / 'normalization_results.json'
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, default=str, ensure_ascii=False)

        click.echo(f"Результаты нормализации сохранены в {results_path}")

        # Выводим статистику
        success_count = sum(1 for r in results if r.get('normalization', {}).get('success', False))
        click.echo(f"Нормализовано {success_count} из {len(results)} файлов")

        # Создаем отчет, если запрошено
        if report:
            report_dir = output_dir / 'report'
            df, report_path = create_loudness_report(results, report_dir)
            if report_path:
                click.echo(f"Отчет о громкости после нормализации создан в {report_path}")

                # Открываем отчет в браузере, если это HTML-отчет
                if str(report_path).endswith('.html'):
                    try:
                        webbrowser.open('file://' + os.path.abspath(report_path))
                    except Exception as e:
                        logger.warning(f"Не удалось открыть отчет в браузере: {e}")

    except Exception as e:
        logger.error(f"Ошибка при нормализации файлов: {e}", exc_info=True)
        click.echo(f"Произошла ошибка: {e}")
        sys.exit(1)


@cli.command('cache')
@click.option('--clear', is_flag=True, help='Очистить кэш')
@click.option('--info', is_flag=True, help='Показать информацию о кэше')
def cache_command(clear, info):
    """Управление кэшем анализатора"""
    try:
        cache = AnalysisCache()

        if clear:
            count = cache.clear_all()
            click.echo(f"Кэш очищен. Удалено {count} файлов.")

        if info or not clear:
            # Сбор информации о кэше
            cache_index = cache._load_cache_index()
            cache_files = list(Path(cache.cache_dir).glob('*.json'))
            cache_files = [f for f in cache_files if f.name != 'index.json']

            total_size = sum(f.stat().st_size for f in cache_files)

            click.echo(f"Информация о кэше:")
            click.echo(f"  Директория: {cache.cache_dir}")
            click.echo(f"  Количество файлов: {len(cache_files)}")
            click.echo(f"  Общий размер: {total_size / 1024 / 1024:.2f} МБ")
            click.echo(f"  Количество записей в индексе: {len(cache_index)}")

            # Показываем список кэшированных файлов
            if cache_index and click.confirm("Показать список кэшированных файлов?"):
                click.echo("\nКэшированные файлы:")
                for file_hash, data in cache_index.items():
                    click.echo(f"  {data['file_path']} ({file_hash[:8]}...)")

    except Exception as e:
        logger.error(f"Ошибка при работе с кэшем: {e}", exc_info=True)
        click.echo(f"Произошла ошибка: {e}")
        sys.exit(1)


@cli.command('batch')
@click.option('--analyze', is_flag=True, help='Выполнить анализ')
@click.option('--normalize', is_flag=True, help='Выполнить нормализацию')
@click.option('--folder', '-f', default=None, help='Путь к папке с аудиофайлами')
@click.option('--output-dir', '-o', default=None, help='Директория для сохранения нормализованных файлов')
@click.option('--target-lufs', type=float, default=None, help='Целевой уровень LUFS')
@click.option('--format', 'output_format', type=click.Choice(['wav', 'mp3', 'flac', 'ogg']), default='wav',
              help='Формат выходных файлов')
@click.option('--report', is_flag=True, help='Создать отчет')
def batch(analyze, normalize, folder, output_dir, target_lufs, output_format, report):
    """Пакетная обработка файлов (анализ и нормализация)"""
    try:
        folder_path = Path(folder) if folder else Path(config.SETLIST_DIR)
        output_dir = Path(output_dir) if output_dir else Path(config.OUTPUT_DIR)

        logger.info(f"Начинаем пакетную обработку файлов в папке: {folder_path}")

        # Если не указаны операции, выполняем и анализ, и нормализацию
        if not analyze and not normalize:
            analyze = normalize = True

        # Создаем анализатор
        analyzer = AudioAnalyzer(folder_path=folder_path)

        # Проверяем, найдены ли файлы
        if not analyzer.audio_files:
            logger.error(f"В папке {folder_path} не найдено поддерживаемых аудиофайлов")
            click.echo(f"В папке {folder_path} не найдено поддерживаемых аудиофайлов")
            return

        # Выводим информацию о найденных файлах
        click.echo(f"Найдено {len(analyzer.audio_files)} аудиофайлов в {folder_path}")

        # Выполняем анализ
        if analyze:
            click.echo("Выполняем анализ файлов...")
            results = analyzer.analyze_all()

            # Сохраняем результаты анализа
            analysis_path = output_dir / 'analysis_results.json'
            save_results_to_json(results, analysis_path)
            click.echo(f"Результаты анализа сохранены в {analysis_path}")

            # Создаем отчет, если запрошено
            if report:
                report_dir = output_dir / 'analysis_report'
                df, report_path = create_loudness_report(results, report_dir)
                if report_path:
                    click.echo(f"Отчет о громкости создан в {report_path}")

        # Выполняем нормализацию
        if normalize:
            click.echo("Выполняем нормализацию файлов...")
            normalizer = AudioNormalizer(analyzer=analyzer)

            # Параметры нормализации
            params = {}
            if target_lufs is not None:
                params['target_lufs'] = target_lufs

            # Выполняем нормализацию
            norm_results = normalizer.normalize_all(
                **params,
                output_format=output_format,
                output_dir=output_dir
            )

            # Сохраняем результаты нормализации
            norm_path = output_dir / 'normalization_results.json'
            with open(norm_path, 'w', encoding='utf-8') as f:
                json.dump(norm_results, f, indent=2, default=str, ensure_ascii=False)

            click.echo(f"Результаты нормализации сохранены в {norm_path}")

            # Создаем отчет, если запрошено
            if report:
                report_dir = output_dir / 'normalization_report'
                df, report_path = create_loudness_report(norm_results, report_dir)
                if report_path:
                    click.echo(f"Отчет о громкости после нормализации создан в {report_path}")

    except Exception as e:
        logger.error(f"Ошибка при пакетной обработке файлов: {e}", exc_info=True)
        click.echo(f"Произошла ошибка: {e}")
        sys.exit(1)


@cli.command('info')
@click.option('--file', '-f', required=True, help='Путь к аудиофайлу для анализа')
def file_info(file):
    """Показать детальную информацию об одном аудиофайле"""
    try:
        file_path = Path(file)
        if not file_path.exists():
            click.echo(f"Файл не найден: {file_path}")
            return

        # Создаем анализатор
        analyzer = AudioAnalyzer()

        # Анализируем файл
        click.echo(f"Анализ файла: {file_path}")
        result = analyzer.analyze_track(file_path)

        # Проверяем результат на ошибки
        if 'error' in result:
            click.echo(f"Ошибка при анализе файла: {result['error']}")
            return

        # Выводим информацию
        click.echo("\n---- Основная информация ----")
        click.echo(f"Файл: {result.get('file', '')}")
        click.echo(f"Длительность: {result.get('duration', 0):.2f} сек")
        click.echo(f"Частота дискретизации: {result.get('sample_rate', 0)} Гц")

        click.echo("\n---- Метаданные ----")
        if 'artist' in result:
            click.echo(f"Исполнитель: {result.get('artist', '')}")
        if 'title' in result:
            click.echo(f"Название: {result.get('title', '')}")
        if 'album' in result:
            click.echo(f"Альбом: {result.get('album', '')}")
        if 'genre' in result:
            click.echo(f"Жанр: {result.get('genre', '')}")
        elif 'detected_genre' in result:
            click.echo(f"Определенный жанр: {result.get('detected_genre', '')}")
        if 'bpm' in result:
            click.echo(f"BPM: {result.get('bpm', 0)}")
        if 'key' in result:
            click.echo(f"Тональность: {result.get('key', '')}")

        click.echo("\n---- Характеристики громкости ----")
        click.echo(f"Интегральная громкость (LUFS): {result.get('lufs_integrated', 0):.2f}")
        click.echo(f"Пиковый уровень (dB): {result.get('peak_db', 0):.2f}")
        click.echo(f"Динамический диапазон (dB): {result.get('dynamic_range', 0):.2f}")
        click.echo(f"Оценка компрессии: {result.get('compression_estimate', 0):.2f}")

        click.echo("\n---- Спектральные характеристики ----")
        click.echo(f"Соотношение басов к высоким: {result.get('bass_to_highs_ratio', 0):.2f}")

        click.echo("\n---- Рекомендации для DJ ----")
        click.echo(f"Рекомендуемое время микса на входе: {result.get('intro_mix_recommendation', 0)} сек")
        click.echo(f"Рекомендуемое время микса на выходе: {result.get('outro_mix_recommendation', 0)} сек")
        if 'first_drop_time' in result:
            click.echo(f"Время первого дропа: {result.get('first_drop_time', 0):.2f} сек")

    except Exception as e:
        logger.error(f"Ошибка при получении информации о файле: {e}", exc_info=True)
        click.echo(f"Произошла ошибка: {e}")
        sys.exit(1)


if __name__ == '__main__':
    cli()