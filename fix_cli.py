"""
Быстрый скрипт для исправления ошибки в cli.py
"""
import re

# Открываем файл cli.py
with open('cli.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Заменяем все вхождения проблемной строки
fixed_content = content.replace(
    'if report_path.endswith',
    'if report_path and str(report_path).endswith'
)

# Сохраняем исправленный файл
with open('cli.py', 'w', encoding='utf-8') as f:
    f.write(fixed_content)

print("Файл cli.py успешно исправлен!")