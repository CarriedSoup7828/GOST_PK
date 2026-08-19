# -*- coding: utf-8 -*-
"""
Сценарный запуск расчёта предохранительного клапана по ГОСТ 12.2.085-2017.

Запуск:
    d:\\Studies\\Diplom\\env\\Scripts\\python.exe run_pk.py

Консоль Windows по умолчанию cp1251, поэтому вывод принудительно
переключается на UTF-8 (иначе кириллица в протоколе ломается).
"""
from __future__ import annotations

import sys

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from pk_gost import inputs, regime as rg, report, thermo as th


def main():
    case = inputs.default_case()

    # ---------- основной расчёт ----------
    res = report.run_case(case)
    print(report.format_report(res))

    # ---------- граница применимости режима Г-Г ----------
    fluid = th.Fluid(case.fluid)
    dT_min = rg.min_superheat_for_GG(fluid, res.ps.P1, res.ps.P2)
    print()
    print(report.LINE)
    print('ГРАНИЦА ПРИМЕНИМОСТИ РАСЧЁТА ПО РЕЖИМУ Г-Г')
    print(report.LINE)
    if dT_min is None:
        print('  При любом разумном перегреве изоэнтропа остаётся в двухфазной области.')
    else:
        T_sat = fluid.T_sat(res.ps.P1)
        print(f'  Тнас(Р1)                = {T_sat:.2f} К')
        print(f'  Минимальный перегрев    = {dT_min:.2f} К  '
              f'(Т1 > {T_sat + dT_min:.2f} К)')
        print(f'  Принято в расчёте       = {res.state["superheat"]:.2f} К  '
              f'(Т1 = {case.T1:.2f} К)')
        margin = res.state['superheat'] - dT_min
        if margin < 0:
            print(f'  ! Перегрева НЕ ХВАТАЕТ ({margin:+.2f} К): клапан запирается в')
            print(f'    двухфазной области, режим Г-2Ф (табл. Д.2, п. Е.2.3.2).')
        elif margin < 5.0:
            print(f'  ! Запас всего {margin:+.2f} К — точка близка к границе Е.1.5.')
            print(f'    Результат чувствителен к Т1, см. таблицу ниже.')
        else:
            print(f'  Запас {margin:+.2f} К — режим Г-Г устойчив.')

    # ---------- таблица чувствительности ----------
    T_sat, rows = report.superheat_table(case)
    print(report.format_superheat_table(case, T_sat, rows))


if __name__ == '__main__':
    main()
