# -*- coding: utf-8 -*-
"""
Температуры сбрасываемой среды и стенок — приложение Ж.

Для криогенных сред это не вспомогательный, а определяющий расчёт:
температура стенки клапана и отводящего трубопровода задаёт выбор
материала (8.1.3 требует «предусмотреть меры по исключению резких
изменений температуры стенок (тепловых ударов) при срабатывании клапана»).

  Ж.4  Т0 = T(P0, s1) — температура в седле (течение до седла изоэнтропно)
  Ж.5  h(P, Tторм) = h1 = const — «температура торможения»
  Ж.6  Тст = r*Тторм + (1 - r)*T,  r = Pr^(1/3), допускается r = 0,85
  Ж.9  при критическом течении идеального газа
       Тст = [1 - (1 - r)*(k-1)/(k+1)]*T1
"""
from __future__ import annotations

from dataclasses import dataclass

R_DEFAULT = 0.85          # Ж.6: «Допускается принимать r = 0,85»


@dataclass
class WallTemperatures:
    T1: float             # К, температура перед клапаном
    T0: float             # К, температура среды в седле (Ж.4)
    T_stag: float         # К, температура торможения (Ж.5)
    T_wall: float         # К, температура стенки (Ж.7)
    r: float              # коэффициент восстановления (Ж.8)
    T_wall_ideal: float | None = None   # контроль по Ж.9 для идеального газа
    note: str = ''


def _T_at_h(fluid, P, h_target, T_lo, T_hi, tol=1e-6):
    """
    Решение h(P, T) = h_target бисекцией по температуре.

    Отрезок поиска обязан целиком лежать в однофазной области: внутри
    купола давление и температура не независимы, и P,T-флеш неприменим.
    Ниже линии плавления CoolProp также не считает свойства, поэтому
    нижнюю границу берут не произвольно малой, а от температуры в седле.
    """
    try:
        f_lo = fluid.h_PT(P, T_lo) - h_target
        f_hi = fluid.h_PT(P, T_hi) - h_target
    except Exception:
        return None
    if f_lo * f_hi > 0:
        return None
    for _ in range(200):
        T_mid = 0.5 * (T_lo + T_hi)
        try:
            f_mid = fluid.h_PT(P, T_mid) - h_target
        except Exception:
            return None
        if f_mid * f_lo > 0:
            T_lo = T_mid
        else:
            T_hi = T_mid
        if T_hi - T_lo < tol:
            break
    return 0.5 * (T_lo + T_hi)


def wall_temperatures(fluid, P1, T1, P0, s1=None, r=None, k_ideal=None):
    """
    Температуры среды и стенки по прил. Ж.

    P0 — абсолютное давление на выходе из седла (Ркр при критическом режиме).
    r  — коэффициент восстановления; если не задан, берётся Pr^(1/3) по (Ж.8),
         а при недоступности числа Прандтля — 0,85.
    k_ideal — показатель адиабаты для контроля по (Ж.9).
    """
    if s1 is None:
        s1 = fluid.s_PT(P1, T1)
    h1 = fluid.h_PT(P1, T1)

    # Ж.4: течение до седла считается изоэнтропным
    T0 = fluid.T_Ps(P0, s1)

    note = ''
    phase0 = fluid.phase_Ps(P0, s1)
    two_phase_throat = (phase0 == 'двухфазная смесь')

    # Ж.5: h(P0, Тторм) = h1. Торможение всегда повышает температуру
    # относительно T0 (кинетическая энергия переходит обратно в энтальпию),
    # поэтому отрезок поиска берём вверх от T0 — он гарантированно лежит
    # в однофазной области и выше линии плавления.
    T_stag = None
    if not two_phase_throat:
        T_lo = max(T0, fluid.T_sat(P0) + 1e-3) if P0 < fluid.P_crit else T0
        T_stag = _T_at_h(fluid, P0, h1, T_lo, T1 + 200.0)
    if T_stag is None:
        T_stag = T1
        note = ('Тторм принята равной Т1: '
                + ('в седле двухфазная смесь, P,T-флеш неприменим'
                   if two_phase_throat else 'решение (Ж.5) не найдено')
                + '. Для идеального газа Тторм = Т1 (Ж.5).')

    if r is None:
        try:
            r = float(fluid.Pr_PT(P1, T1)) ** (1.0 / 3.0)
        except Exception:
            r = R_DEFAULT

    # Ж.7: Тст = r*Тторм + (1 - r)*T
    T_wall = r * T_stag + (1.0 - r) * T0

    T_wall_ideal = None
    if k_ideal is not None:
        # Ж.9: контроль по модели идеального газа при критическом течении
        T_wall_ideal = (1.0 - (1.0 - r) * (k_ideal - 1.0) / (k_ideal + 1.0)) * T1

    return WallTemperatures(T1=T1, T0=T0, T_stag=T_stag, T_wall=T_wall,
                            r=r, T_wall_ideal=T_wall_ideal, note=note)
