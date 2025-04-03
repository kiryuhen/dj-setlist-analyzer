"""
Основной файл для запуска DJ сетлист-анализатора
"""
import os
import sys
import logging
from pathlib import Path

import config
from src.analyzer import AudioAnalyzer
from src.normalizer import AudioNormalizer
from src.utils import (
    setup_logging, save_results_to_json, create_loudness_report,
    create_setlist_recommendations
)
from cli import cli


# Настройка логирования
logger = setup_logging()


def main():
    """Основная функция для запуска приложения"""
    try:
        # Проверка наличия папки для аудиофайлов
        setlist_dir = Path(config.SETLIST_DIR)
        if not setlist_dir.exists():
            setlist_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Создана папка для сетлиста: {setlist_dir}")

        # Запуск CLI интерфейса
        cli()

    except Exception as e:
        logger.error(f"Ошибка при запуске приложения: {e}", exc_info=True)
        print(f"Произошла ошибка: {e}")
        sys.exit(1)


def run_batch_processing(
    folder_path=None,
    output_dir=None,
    target_lufs=None,
    output_format='wav',
    do_analyze=True,
    do_normalize=True,
    create_report=True
):
    """
    Выполнение пакетной обработки файлов программным путем

    Args:
        folder_path: Путь к папке с аудиофайлами
        output_dir: Путь для сохранения результатов
        target_lufs: Целевой уровень LUFS
        output_format: Формат выходных файлов
        do_analyze: Выполнить анализ
        do_normalize: Выполнить нормализацию
        create_report: Создать отчет

    Returns:
        dict: Результаты обработки
    """
    results = {}

    try:
        # Настройка путей
        folder_path = Path(folder_path) if folder_path else Path(config.SETLIST_DIR)
        output_dir = Path(output_dir) if output_dir else Path(config.OUTPUT_DIR)
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Пакетная обработка файлов в {folder_path}")

        # Создаем анализатор
        analyzer = AudioAnalyzer(folder_path=folder_path)

        # Проверяем, найдены ли файлы
        if not analyzer.audio_files:
            logger.warning(f"В папке {folder_path} не найдено аудиофайлов")
            results['error'] = f"В папке {folder_path} не найдено аудиофайлов"
            return results

        logger.info(f"Найдено {len(analyzer.audio_files)} аудиофайлов")
        results['file_count'] = len(analyzer.audio_files)

        # Выполняем анализ
        if do_analyze:
            logger.info("Выполняем анализ файлов...")
            analysis_results = analyzer.analyze_all()

            # Сохраняем результаты анализа
            analysis_path = output_dir / 'analysis_results.json'
            save_results_to_json(analysis_results, analysis_path)
            logger.info(f"Результаты анализа сохранены в {analysis_path}")

            results['analysis'] = {
                'success': True,
                'results_path': str(analysis_path)
            }

            # Создаем отчет, если запрошено
            if create_report:
                report_dir = output_dir / 'analysis_report'
                df, report_path = create_loudness_report(analysis_results, report_dir)
                if report_path:
                    logger.info(f"Отчет о громкости создан в {report_path}")
                    results['analysis']['report_path'] = str(report_path)

            # Создаем рекомендации по сетлисту
            recommendations = create_setlist_recommendations(analysis_results)
            recommendations_path = output_dir / 'setlist_recommendations.json'
            save_results_to_json(recommendations, recommendations_path)
            logger.info(f"Рекомендации по сетлисту сохранены в {recommendations_path}")
            results['analysis']['recommendations_path'] = str(recommendations_path)

        # Выполняем нормализацию
        if do_normalize:
            logger.info("Выполняем нормализацию файлов...")
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
            save_results_to_json(norm_results, norm_path)
            logger.info(f"Результаты нормализации сохранены в {norm_path}")

            # Подсчитываем статистику успешной нормализации
            success_count = sum(1 for r in norm_results if r.get('normalization', {}).get('success', False))

            results['normalization'] = {
                'success': True,
                'results_path': str(norm_path),
                'success_count': success_count,
                'total_count': len(norm_results)
            }

            # Создаем отчет, если запрошено
            if create_report:
                report_dir = output_dir / 'normalization_report'
                df, report_path = create_loudness_report(norm_results, report_dir)
                if report_path:
                    logger.info(f"Отчет о нормализации создан в {report_path}")
                    results['normalization']['report_path'] = str(report_path)

        return results

    except Exception as e:
        logger.error(f"Ошибка при пакетной обработке: {e}", exc_info=True)
        results['error'] = str(e)
        return results


if __name__ == '__main__':
    main()