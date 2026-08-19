# -*- coding: utf-8 -*-
"""
Метод прямого интегрирования — приложение Е.1.

Универсальный метод расчёта массовой скорости; Е.1.1 рекомендует применять
его «когда есть сомнения в применимости других методов». Работает для всех
режимов течения табл. Д.2 без изменения формул.

Основа — уравнение (Е.1.1):
    G*ideal = rho0 * [ -2 * INT(P1..P0) dP/rho ]^(1/2)

численно берётся по формуле трапеций (Е.1.4):
    G*ideal(P(j)) = rho_j * [ SUM_i (1/rho_(i-1) + 1/rho_i) * (P(i-1) - P(i)) ]^(1/2)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

REGIME_CRITICAL = 'критический'
REGIME_SUBCRITICAL = 'докритический'


# --------------------------------------------------------------------------
#  Ядро: чистые численные функции (тестируются без CoolProp)
# --------------------------------------------------------------------------
def mass_flux_curve(P, rho):
    """
    Формула трапеций (Е.1.4).

    P, rho — массивы по убыванию давления, P[0] = P1.
    Возвращает (G*, I), где
        I[j] = -2*INT(P1..P[j]) dP/rho,  Дж/кг
        G*[j] = rho[j]*sqrt(I[j]),       кг/(м2*с)

    Проверка знака: -2*INT(P1..P0) dP/rho = +2*INT(P0..P1) dP/rho, а
    2 * SUM (P(i-1)-P(i)) * (1/rho_(i-1) + 1/rho_i)/2 = SUM (1/rho_(i-1)+1/rho_i)*(P(i-1)-P(i)).
    """
    P = np.asarray(P, dtype=float)
    rho = np.asarray(rho, dtype=float)
    if P.shape != rho.shape:
        raise ValueError('P и rho должны быть одинаковой длины')
    if np.any(np.diff(P) > 0.0):
        raise ValueError('Давления должны идти по невозрастанию: P1 -> P2')
    if np.any(rho <= 0.0):
        raise ValueError('Плотность должна быть положительной')

    terms = (1.0 / rho[:-1] + 1.0 / rho[1:]) * (P[:-1] - P[1:])
    I = np.concatenate(([0.0], np.cumsum(terms)))
    return rho * np.sqrt(I), I


def first_maximum_index(g):
    """
    Индекс первого максимума G*(P0) — признак критического режима (Е.1.2).

    «Если величина G*ideal(P0) имеет максимумы внутри данного отрезка, то первый,
    ближайший к Р1 максимум, соответствует критическому давлению Ркр... Если же
    величина G*ideal(P0) монотонно возрастает на всём отрезке, то имеет место
    докритический режим течения.»

    Возвращает None при монотонном росте (докритический режим).
    """
    g = np.asarray(g, dtype=float)
    falling = np.flatnonzero(np.diff(g) < 0.0)
    if falling.size == 0:
        return None
    return int(falling[0])


def parabolic_peak(P, g, j):
    """
    Уточнение положения и величины максимума параболой по трём точкам.

    Значение максимума на равномерной сетке имеет погрешность O(h^2) и без
    уточнения, но положение Ркр — только O(h); парабола даёт O(h^2) и для него.
    """
    if j <= 0 or j >= len(g) - 1:
        return float(P[j]), float(g[j])
    y0, y1, y2 = float(g[j - 1]), float(g[j]), float(g[j + 1])
    denom = y0 - 2.0 * y1 + y2
    if denom == 0.0:
        return float(P[j]), float(g[j])
    # смещение вершины в долях шага сетки
    delta = 0.5 * (y0 - y2) / denom
    delta = max(-1.0, min(1.0, delta))
    h = float(P[j + 1] - P[j - 1]) / 2.0        # шаг (отрицательный: P убывает)
    return float(P[j] + delta * h), float(y1 - 0.25 * (y0 - y2) * delta)


# --------------------------------------------------------------------------
#  Результат расчёта
# --------------------------------------------------------------------------
@dataclass
class FlowSolution:
    """Результат расчёта массовой скорости по модели идеального сопла."""
    regime: str                # критический / докритический
    P0: float                  # абс. давление на выходе из седла, Па
    G_ideal: float             # массовая скорость G*ideal, кг/(м2*с)
    rho0: float                # плотность в седле, кг/м3
    integral: float            # -2*INT dP/rho в точке P0, Дж/кг
    P1: float
    rho1: float
    beta_cr: float | None      # Ркр/Р1 (только критический режим)
    on_phase_boundary: bool    # Ркр совпал с границей фазовой диаграммы (Е.1.5)
    n_plus: float | None       # показатель изоэнтропы со стороны P > Ркр
    n_minus: float | None      # показатель изоэнтропы со стороны P < Ркр
    grid_points: int

    @property
    def Kn(self):
        """Безразмерная массовая скорость Кп = G*ideal/sqrt(P1*rho1)  (Д.3, Д.4)."""
        return self.G_ideal / np.sqrt(self.P1 * self.rho1)

    @property
    def Kb(self):
        """Кb = G*ideal / G*ideal кр (Д.4); при критическом режиме = 1."""
        return 1.0 if self.regime == REGIME_CRITICAL else None


# --------------------------------------------------------------------------
#  Драйвер
# --------------------------------------------------------------------------
def build_grid(P1, P2, breakpoints=(), n_points=20000):
    """
    Сетка давлений от Р1 до Р2 с обязательным включением точек пересечения
    изоэнтропы с границами фазовой диаграммы.

    Примечание к Е.1.4: «при расчёте таких случаев методом прямого
    интегрирования в набор точек P(i) следует включать точки пересечения
    изоэнтропы s = s1 с границами фазовой диаграммы».
    """
    edges = [P1] + [p for p in sorted(breakpoints, reverse=True) if P2 < p < P1] + [P2]
    n_seg = len(edges) - 1
    per_seg = max(2, int(n_points // n_seg))
    parts = []
    for k in range(n_seg):
        seg = np.linspace(edges[k], edges[k + 1], per_seg)
        parts.append(seg if k == 0 else seg[1:])
    return np.concatenate(parts)


def solve(fluid, P1, s1, P2, n_points=20000, boundary_rtol=2e-3):
    """
    Расчёт массовой скорости методом прямого интегрирования (Е.1.2).

    fluid    — объект с методами rho_Ps, phase_crossings, n_isentropic
    P1, P2   — абсолютные давления до и за клапаном, Па
    s1       — энтропия среды перед клапаном, Дж/(кг*К)
    boundary_rtol — относительный допуск, при котором Ркр считается совпавшим
                    с границей фазовой диаграммы (особый случай Е.1.5)
    """
    if P2 >= P1:
        raise ValueError(f'Должно быть P2 < P1, получено P1={P1:.1f}, P2={P2:.1f} Па')

    crossings = fluid.phase_crossings(P1, P2, s1)
    P_grid = build_grid(P1, P2, crossings, n_points)
    rho_grid = np.asarray(fluid.rho_Ps(P_grid, s1), dtype=float)

    g, I = mass_flux_curve(P_grid, rho_grid)
    j = first_maximum_index(g)

    rho1 = float(rho_grid[0])

    if j is None:
        # Докритический режим: расход определяется давлением за клапаном
        return FlowSolution(
            regime=REGIME_SUBCRITICAL, P0=float(P_grid[-1]), G_ideal=float(g[-1]),
            rho0=float(rho_grid[-1]), integral=float(I[-1]), P1=P1, rho1=rho1,
            beta_cr=None, on_phase_boundary=False, n_plus=None, n_minus=None,
            grid_points=len(P_grid))

    P_cr, G_cr = parabolic_peak(P_grid, g, j)

    # Совпало ли Ркр с границей фазовой диаграммы? (особый случай Е.1.5)
    on_boundary = any(abs(P_cr - pc) <= boundary_rtol * P1 for pc in crossings)

    n_plus = n_minus = None
    try:
        if on_boundary:
            # На границе n терпит разрыв: центральную разность брать нельзя
            n_plus = fluid.n_isentropic(P_cr, s1, side='gas')
            n_minus = fluid.n_isentropic(P_cr, s1, side='low')
        else:
            n_plus = n_minus = fluid.n_isentropic(P_cr, s1, side='center')
    except Exception:
        pass                      # контроль Е.1.5 необязателен, расчёт не срывать

    return FlowSolution(
        regime=REGIME_CRITICAL, P0=P_cr, G_ideal=G_cr,
        rho0=float(fluid.rho_Ps(P_cr, s1)), integral=float(I[j]), P1=P1, rho1=rho1,
        beta_cr=P_cr / P1, on_phase_boundary=on_boundary,
        n_plus=n_plus, n_minus=n_minus, grid_points=len(P_grid))


def check_sonic(sol: FlowSolution):
    """
    Контроль Е.1.5.

    В общем случае при критическом режиме
        G*ideal = sqrt(n_кр * Ркр * rho_кр),
    но если точка критического истечения совпадает с границей фазовой
    диаграммы, показатель изоэнтропы испытывает скачок и выполняется лишь
        sqrt(n- * Ркр * rho_кр) <= G*ideal <= sqrt(n+ * Ркр * rho_кр).

    Возвращает словарь с границами и признаком выполнения.
    """
    if sol.regime != REGIME_CRITICAL or sol.n_plus is None:
        return None
    lo = float(np.sqrt(sol.n_minus * sol.P0 * sol.rho0))
    hi = float(np.sqrt(sol.n_plus * sol.P0 * sol.rho0))
    lo, hi = min(lo, hi), max(lo, hi)
    tol = 5e-3 * hi
    return {
        'equality': not sol.on_phase_boundary,
        'n_plus': sol.n_plus,
        'n_minus': sol.n_minus,
        'lower': lo,
        'upper': hi,
        'G_ideal': sol.G_ideal,
        'ok': (lo - tol) <= sol.G_ideal <= (hi + tol),
        'rel_dev': abs(sol.G_ideal - hi) / hi if not sol.on_phase_boundary else None,
    }
