# -*- coding: utf-8 -*-
"""
Точка входа для сборки EXE (PyInstaller).
Запускает GUI-приложение расчёта динамики PRV.
"""

import os
import sys

# Убедимся, что модуль engine доступен при запуске из одного каталога с exe
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from prv_app.app import main  # noqa: E402

if __name__ == '__main__':
    main()
