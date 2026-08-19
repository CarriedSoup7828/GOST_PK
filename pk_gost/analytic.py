# -*- coding: utf-8 -*-
"""
Аналитические методы расчёта массовой скорости на основе упрощённых
уравнений состояния — приложение Е.2.

  Е.2.1  несжимаемая жидкость (Д.27): rho = const
  Е.2.2  постоянный показатель изоэнтропы (Д.28) и омега-метод (Д.29),
         таблица Е.1 — коэффициенты Кп, beta_кр, Кп кр, Кb
  Е.2.3  комбинация уравнений состояния, таблица Е.2 — те же коэффициенты
         с учётом ненулевой скорости в начале интервала (число Маха М)

Все Кп безразмерны: G*ideal = Кп * sqrt(P1*rho1)  (Д.3, Д.4).
В таблице Е.2 коэффициенты отнесены к началу текущего интервала:
G*ideal = Кп М * sqrt(P(j)*rho(j))  (Е.2.16).

О численных методах вместо закрытых формул
------------------------------------------
Критическое отношение давлений для омега-метода ГОСТ разрешает находить
«из решения уравнения (Е.2.11)... решаемого численно либо с использованием
формулы (Е.2.12)», а в Е.5.2.2 прямо предписывает определять его
«из условия первого максимума Кп... для чего могут использоваться
стандартные численные методы». Здесь beta_кр для омега-метода находится
именно поиском максимума Кп(beta) — это работает и при M != 0, и
проверяется в тестах против аппроксимации (Е.2.12) и против предельного
случая n = 1 (Е.2.5: beta_кр = 1/sqrt(e) = 0,60653).
"""
from __future__ import annotations

import numpy as np

# =========================================================================
#  Е.2.1  Несжимаемая жидкость
# =========================================================================
def Kn_incompressible(beta):
    """
    (Е.2.1) Кп = sqrt(2*(1 - beta)).

    Течение несжимаемой жидкости всегда докритическое.
    """
    return np.sqrt(2.0 * (1.0 - np.asarray(beta, float)))


# =========================================================================
#  Таблица Е.1 / Е.2 — постоянный показатель изоэнтропы
# =========================================================================
_N1_TOL = 1e-9        # окрестность n = 1, где формулы вырождаются


def Kn_const_n(beta, n, M=0.0):
    """
    Безразмерная массовая скорость при постоянном показателе изоэнтропы.

    (Е.2.17) Кп М = sqrt( 2n/(n-1) * [ (1 + (n-1)/2*M^2)*beta^(2/n) - beta^((n+1)/n) ] )
    (Е.2.18) при n = 1:  Кп М = beta * sqrt(-2*ln(beta) + M^2)

    При M = 0 переходит в (Е.2.2) / (Е.2.3) таблицы Е.1.
    """
    beta = np.asarray(beta, float)
    M2 = float(M) ** 2
    if abs(n - 1.0) < _N1_TOL:
        return beta * np.sqrt(np.maximum(-2.0 * np.log(beta) + M2, 0.0))
    A = 1.0 + 0.5 * (n - 1.0) * M2
    inner = A * beta ** (2.0 / n) - beta ** ((n + 1.0) / n)
    return np.sqrt(np.maximum(2.0 * n / (n - 1.0) * inner, 0.0))


def beta_cr_const_n(n, M=0.0):
    """
    Критическое отношение давлений при постоянном показателе изоэнтропы.

    (Е.2.19) beta_крМ = (2/(n+1))^(n/(n-1)) * (1 + (n-1)/2*M^2)^(n/(n-1))
    (Е.2.20) при n = 1:  beta_крМ = exp((M^2 - 1)/2)

    При M = 0 — (Е.2.4) / (Е.2.5): beta_кр = (2/(n+1))^(n/(n-1)); при n = 1
    beta_кр = 1/sqrt(e) = 0,60653.
    """
    M2 = float(M) ** 2
    if abs(n - 1.0) < _N1_TOL:
        return float(np.exp(0.5 * (M2 - 1.0)))
    A = 1.0 + 0.5 * (n - 1.0) * M2
    return float((2.0 / (n + 1.0)) ** (n / (n - 1.0)) * A ** (n / (n - 1.0)))


def Kn_cr_const_n(n, M=0.0):
    """
    Безразмерная массовая скорость при критическом истечении.

    (Е.2.21) Кп крМ = sqrt(n * beta_крМ^((n+1)/n))
                    = sqrt(2n/(n+1)) * (2/(n+1))^(1/(n-1)) * (1+(n-1)/2*M^2)^((n+1)/(2(n-1)))
    (Е.2.22) при n = 1:  Кп крМ = exp((M^2 - 1)/2)

    При M = 0 — (Е.2.6) / (Е.2.7): при n = 1 Кп кр = 1/sqrt(e) = 0,60653.
    """
    M2 = float(M) ** 2
    if abs(n - 1.0) < _N1_TOL:
        return float(np.exp(0.5 * (M2 - 1.0)))
    b = beta_cr_const_n(n, M)
    return float(np.sqrt(n * b ** ((n + 1.0) / n)))


def Kb_const_n(beta, n, M=0.0):
    """
    (Е.2.8) / (Е.2.23) Кb = Кп/Кп кр < 1 — отличие докритической массовой
    скорости от критической (Д.4).
    """
    return Kn_const_n(beta, n, M) / Kn_cr_const_n(n, M)


def Kb_const_n_literal(beta, n):
    """
    (Е.2.8) в буквальной записи стандарта — только для сверки в тестах:
        Кb = sqrt( (n+1)/(n-1) * [beta^(2/n) - beta^((n+1)/n)] ) * ((n+1)/2)^(1/(n-1))
    (Е.2.9) при n = 1:  Кb = beta * sqrt(-2*e*ln(beta))
    """
    beta = np.asarray(beta, float)
    if abs(n - 1.0) < _N1_TOL:
        return beta * np.sqrt(np.maximum(-2.0 * np.e * np.log(beta), 0.0))
    inner = beta ** (2.0 / n) - beta ** ((n + 1.0) / n)
    return (np.sqrt(np.maximum((n + 1.0) / (n - 1.0) * inner, 0.0))
            * ((n + 1.0) / 2.0) ** (1.0 / (n - 1.0)))


# =========================================================================
#  Таблица Е.1 / Е.2 — омега-метод
# =========================================================================
def Kn_omega(beta, omega, M=0.0):
    """
    (Е.2.25) Кп М = (M^2/omega - 2*[omega*ln(beta) + (omega-1)*(1-beta)])^(1/2)
                    / (omega*(1/beta - 1) + 1)

    При M = 0 переходит в (Е.2.10) таблицы Е.1.
    """
    beta = np.asarray(beta, float)
    M2 = float(M) ** 2
    num = M2 / omega - 2.0 * (omega * np.log(beta) + (omega - 1.0) * (1.0 - beta))
    den = omega * (1.0 / beta - 1.0) + 1.0
    return np.sqrt(np.maximum(num, 0.0)) / den


def beta_cr_omega_approx(omega):
    """
    (Е.2.12) аппроксимация критического отношения давлений для омега-метода:

        beta_кр = [1 + (1,044600 - 0,009343*sqrt(omega)) * omega^(-0,56261)]
                  ^ (-0,703560 + 0,014685*ln(omega))

    Проверка при omega = 1 даёт 0,60660 против точного 1/sqrt(e) = 0,60653.
    """
    omega = float(omega)
    base = 1.0 + (1.044600 - 0.009343 * np.sqrt(omega)) * omega ** (-0.56261)
    expo = -0.703560 + 0.014685 * np.log(omega)
    return float(base ** expo)


def _golden_max(f, lo, hi, iters=200):
    """Поиск максимума унимодальной функции золотым сечением на [lo, hi]."""
    inv_phi = (np.sqrt(5.0) - 1.0) / 2.0
    a, b = lo, hi
    c = b - inv_phi * (b - a)
    d = a + inv_phi * (b - a)
    fc, fd = f(c), f(d)
    for _ in range(iters):
        if fc < fd:
            a, c, fc = c, d, fd
            d = a + inv_phi * (b - a)
            fd = f(d)
        else:
            b, d, fd = d, c, fc
            c = b - inv_phi * (b - a)
            fc = f(c)
        if b - a < 1e-14:
            break
    x = 0.5 * (a + b)
    return x, f(x)


def _first_max_of_Kn(kn_func, beta_lo=1e-9, n_scan=4000):
    """
    Первый (ближайший к beta = 1) максимум Кп(beta) на (beta_lo, 1).

    Именно этот приём предписывает Е.5.2.2 для случаев, когда закрытая формула
    для beta_кр неприменима.
    """
    grid = np.linspace(1.0 - 1e-12, beta_lo, n_scan)
    vals = np.asarray(kn_func(grid), float)
    falling = np.flatnonzero(np.diff(vals) < 0.0)
    if falling.size == 0:
        # Кп монотонно растёт до beta_lo — критического режима нет
        return None, float(vals[-1])
    j = int(falling[0])
    lo = grid[min(j + 1, n_scan - 1)]
    hi = grid[max(j - 1, 0)]
    return _golden_max(lambda b: float(kn_func(b)), lo, hi)


def beta_cr_omega(omega, M=0.0):
    """
    Критическое отношение давлений для омега-метода — численно, как первый
    максимум Кп(beta) (Е.2.11 / Е.2.26 «решаемое численно»; Е.5.2.2).
    """
    b, _ = _first_max_of_Kn(lambda x: Kn_omega(x, omega, M))
    return b


def Kn_cr_omega(omega, M=0.0):
    """
    (Е.2.13) Кп кр = beta_кр/sqrt(omega) — при M = 0.
    (Е.2.29) Кп крМ = beta_крМ/sqrt(omega).

    Для контроля рядом возвращается значение из максимума Кп(beta):
    они должны совпадать (это тождество омега-метода).
    """
    b = beta_cr_omega(omega, M)
    if b is None:
        return None
    return float(b / np.sqrt(omega))


def Kb_omega(beta, omega, M=0.0):
    """
    (Е.2.14) Кb = (-2*omega*[omega*ln(beta) + (omega-1)*(1-beta)])^(1/2)
                  / ([omega*(1/beta - 1) + 1] * beta_кр)

    Тождественно равно Кп/Кп кр.
    """
    kcr = Kn_cr_omega(omega, M)
    if kcr is None:
        return None
    return Kn_omega(beta, omega, M) / kcr


def omega_ideal_gas_liquid(x, cp_g, cp_l, rho_g, rho_l, Z_g, R_spec, Lam):
    """
    (Е.3.6) Параметр омега для парожидкостной смеси однокомпонентной среды
    по модели идеального газа и несжимаемой жидкости (при Pr < 0,5):

        omega = [x + (1-x)*cpl/cpg] / [x + (1-x)*rho_g/rho_l]
                * cpg/(Zg*Rуд) * Lam^2 + eps*(1 - 2*Lam)

    eps — объёмная доля газа (п. 3.4):  eps = [1 + (1-x)/x * rho_g/rho_l]^(-1)
    Lam — температурный показатель изоэнтропы двухфазной смеси (Е.3.4);
          для большинства веществ 0,04...0,16.
    """
    eps = 1.0 / (1.0 + (1.0 - x) / x * rho_g / rho_l) if x > 0 else 0.0
    num = x + (1.0 - x) * cp_l / cp_g
    den = x + (1.0 - x) * rho_g / rho_l
    return float(num / den * cp_g / (Z_g * R_spec) * Lam ** 2 + eps * (1.0 - 2.0 * Lam))


# =========================================================================
#  Е.2.3  Расчёт по комбинации уравнений состояния
# =========================================================================
class Interval:
    """
    Один интервал изоэнтропы с собственным уравнением состояния.

    kind  : 'n' — постоянный показатель изоэнтропы (Д.28)
            'omega' — омега-метод (Д.29)
            'rho' — несжимаемая жидкость (Д.27), допустима только первой (Е.2.3)
    value : n либо omega (для 'rho' не используется)
    P_start, rho_start — параметры в начале интервала (базовая точка)
    P_end   — давление конца интервала
    """

    def __init__(self, kind, value, P_start, rho_start, P_end, label=''):
        self.kind = kind
        self.value = value
        self.P_start = float(P_start)
        self.rho_start = float(rho_start)
        self.P_end = float(P_end)
        self.label = label

    # --- коэффициенты интервала как функции beta = P/P_start и M на входе ---
    def Kn(self, beta, M):
        if self.kind == 'n':
            return Kn_const_n(beta, self.value, M)
        if self.kind == 'omega':
            return Kn_omega(beta, self.value, M)
        if self.kind == 'rho':
            # (Е.2.1) с учётом ненулевой скорости на входе: Кп = sqrt(2(1-beta) + M^2)
            return np.sqrt(np.maximum(2.0 * (1.0 - np.asarray(beta, float)) + M ** 2, 0.0))
        raise ValueError(f'Неизвестный тип интервала: {self.kind!r}')

    def beta_cr(self, M):
        if self.kind == 'n':
            return beta_cr_const_n(self.value, M)
        if self.kind == 'omega':
            return beta_cr_omega(self.value, M)
        if self.kind == 'rho':
            return None                  # несжимаемая жидкость всегда докритична
        raise ValueError(f'Неизвестный тип интервала: {self.kind!r}')

    def Kn_cr(self, M):
        if self.kind == 'n':
            return Kn_cr_const_n(self.value, M)
        if self.kind == 'omega':
            b = beta_cr_omega(self.value, M)
            return None if b is None else float(b / np.sqrt(self.value))
        return None

    def n_local(self):
        """Показатель изоэнтропы интервала (для числа Маха: omega = 1/n)."""
        if self.kind == 'n':
            return self.value
        if self.kind == 'omega':
            return 1.0 / self.value
        return None

    def __repr__(self):
        v = f'{self.kind}={self.value:.4f}' if self.value is not None else self.kind
        return (f'Interval({v}, P: {self.P_start/1e5:.3f}->{self.P_end/1e5:.3f} бар'
                f'{", " + self.label if self.label else ""})')


def solve_chain(intervals, P2):
    """
    Расчёт массовой скорости по цепочке интервалов — алгоритм Е.2.3.

    1) Первый интервал считают по Е.2.1 или Е.2.2 (это случай M = 0 таблицы Е.2).
       Если критическое течение достигнуто внутри интервала — расчёт закончен.
    2) Иначе считают G*ideal в конце интервала и переходят к следующему,
       используя число Маха (Е.2.16):
           M(j) = G*ideal(j) / sqrt(n(j) * rho(j) * P(j))
       где n(j) — показатель изоэнтропы со стороны давлений меньше P(j).
       Если M(j) >= 1, критическое течение имеет место при P = P(j).
    3) На последнем интервале при отсутствии запирания — формула (Е.2.30а).

    Возвращает dict с полями regime, P0, G_ideal, beta_cr (отнесённое к P1),
    Kn (отнесённое к P1*rho1), Mach — числа Маха на границах интервалов.
    """
    if not intervals:
        raise ValueError('Нужен хотя бы один интервал')

    P1 = intervals[0].P_start
    rho1 = intervals[0].rho_start
    G = 0.0
    machs = []

    for k, iv in enumerate(intervals):
        base = np.sqrt(iv.P_start * iv.rho_start)

        # число Маха в начале интервала (Е.2.16)
        n_loc = iv.n_local()
        if k == 0:
            M = 0.0
        elif n_loc is None:
            M = 0.0
        else:
            M = float(G / np.sqrt(n_loc * iv.rho_start * iv.P_start))
        machs.append(M)

        # Е.2.3: «Если M(j) >= 1,0, то критическое течение имеет место при Ркр = P(j)»
        if M >= 1.0:
            return dict(regime='критический', P0=iv.P_start, G_ideal=G,
                        beta_cr=iv.P_start / P1, Kn=G / np.sqrt(P1 * rho1),
                        Mach=machs, interval=k,
                        note='критическое течение на границе фазовой области (M >= 1)')

        b_cr = iv.beta_cr(M)
        beta_end = iv.P_end / iv.P_start

        if b_cr is not None and beta_end < b_cr:
            # запирание внутри интервала
            K_cr = iv.Kn_cr(M)
            G = float(K_cr * base)
            P0 = b_cr * iv.P_start
            return dict(regime='критический', P0=P0, G_ideal=G,
                        beta_cr=P0 / P1, Kn=G / np.sqrt(P1 * rho1),
                        Mach=machs, interval=k, note='')

        # запирания нет — считаем массовую скорость в конце интервала
        G = float(iv.Kn(beta_end, M) * base)

    # последний интервал пройден без запирания — докритический режим (Е.2.30а)
    return dict(regime='докритический', P0=P2, G_ideal=G, beta_cr=None,
                Kn=G / np.sqrt(P1 * rho1), Mach=machs,
                interval=len(intervals) - 1, note='')
