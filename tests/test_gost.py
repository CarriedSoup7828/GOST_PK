# -*- coding: utf-8 -*-
"""
Юнит-тесты по контрольным точкам ГОСТ 12.2.085-2017.

Запуск:
    d:\\Studies\\Diplom\\env\\Scripts\\python.exe -m pytest tests/test_gost.py -v
либо без pytest:
    d:\\Studies\\Diplom\\env\\Scripts\\python.exe tests/test_gost.py

Эталоны берутся непосредственно из текста стандарта:
таблица И.1 (критические отношения давлений), формулы Е.2.4-Е.2.9,
Е.2.17-Е.2.24, таблица Д.1, формулы Д.12/Д.13, п. 5.4, константы Д.7/Д.9.
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pk_gost import analytic as an, coeffs as cf, integrate as integ
from pk_gost import pressures as pr, sizing as sz, thermo as th


def approx(a, b, rel=1e-9, abs_=0.0):
    return abs(a - b) <= max(rel * abs(b), abs_)


# =========================================================================
#  1. Постоянный показатель изоэнтропы: Е.2.4-Е.2.9
# =========================================================================
def test_beta_cr_table_I1():
    """
    Таблица И.1: критическое отношение давлений beta_кр для реальных сред,
    рассчитанное по показателю адиабаты k при 1,013 бар и 0 °С.

    Значения таблицы округлены до трёх знаков, поэтому допуск 1,5e-3:
    максимальное расхождение с формулой (Е.2.4) — 0,00140 (хлор, k = 1,34).
    """
    reference = {
        1.40: 0.528,   # азот, воздух, кислород, окись углерода
        1.67: 0.488,   # аргон
        1.32: 0.543,   # аммиак
        1.30: 0.547,   # метан, сероводород
        1.31: 0.545,   # углекислый газ
        1.23: 0.559,   # ацетилен
        1.10: 0.586,   # бутан
        1.14: 0.576,   # пропан, фреон 22
        1.20: 0.564,   # хлористый метил
        1.22: 0.560,   # этан
        1.24: 0.557,   # этилен
        1.34: 0.540,   # хлор
        1.41: 0.527,   # водород
    }
    for k, beta_ref in reference.items():
        beta = an.beta_cr_const_n(k)
        assert abs(beta - beta_ref) < 1.5e-3, f'k={k}: {beta:.5f} != {beta_ref}'


def test_table_I1_helium_row_is_inconsistent():
    """
    ИЗВЕСТНОЕ РАСХОЖДЕНИЕ В САМОМ СТАНДАРТЕ.

    Для гелия таблица И.1 указывает k = 1,66 и beta_кр = 0,557. Формула
    (Е.2.4) при k = 1,66 даёт 0,4881; значение 0,557 соответствует k = 1,239
    и совпадает со строкой этилена (k = 1,24, beta_кр = 0,557) — похоже на
    ошибку переноса в таблице.

    Практический вывод: для гелия beta_кр следует считать по (Е.2.4), а не
    брать из таблицы И.1. Тест фиксирует расхождение, чтобы оно не всплыло
    неожиданно при смене среды.
    """
    beta_formula = an.beta_cr_const_n(1.66)
    assert abs(beta_formula - 0.4881) < 5e-4, beta_formula
    assert abs(beta_formula - 0.557) > 0.06, 'расхождение с табл. И.1 исчезло?'
    # 0,557 таблицы соответствует показателю адиабаты этилена
    assert abs(an.beta_cr_const_n(1.24) - 0.557) < 1e-3


def test_Kn_cr_nitrogen():
    """(Е.2.6) для азота k = 1,40: Кп кр = 0,6847."""
    assert abs(an.Kn_cr_const_n(1.40) - 0.6847) < 5e-5
    assert abs(an.beta_cr_const_n(1.40) - 0.5283) < 5e-4


def test_n_equal_one_limits():
    """(Е.2.5), (Е.2.7): при n = 1 beta_кр = Кп кр = 1/sqrt(e) = 0,60653."""
    ref = 1.0 / math.sqrt(math.e)
    assert abs(an.beta_cr_const_n(1.0) - ref) < 1e-9
    assert abs(an.Kn_cr_const_n(1.0) - ref) < 1e-9
    assert abs(ref - 0.60653) < 1e-5


def test_Kn_n1_formula():
    """(Е.2.3) при n = 1: Кп = beta*sqrt(-2*ln(beta))."""
    for b in (0.9, 0.7, 0.60653, 0.4):
        assert approx(float(an.Kn_const_n(b, 1.0)), b * math.sqrt(-2.0 * math.log(b)), 1e-12)


def test_Kn_cr_is_max_of_Kn():
    """Кп кр (Е.2.6) должно совпадать с максимумом Кп(beta) (Е.2.2)."""
    for n in (1.0, 1.1, 1.2, 1.3, 1.4, 1.67, 2.0):
        b_cr = an.beta_cr_const_n(n)
        k_cr = an.Kn_cr_const_n(n)
        grid = np.linspace(0.05, 0.999, 40000)
        k_max = float(np.max(an.Kn_const_n(grid, n)))
        assert abs(k_cr - k_max) / k_max < 1e-4, f'n={n}'
        j = int(np.argmax(an.Kn_const_n(grid, n)))
        assert abs(grid[j] - b_cr) < 1e-3, f'n={n}'


def test_Kb_literal_matches_ratio():
    """(Е.2.8)/(Е.2.9) в буквальной записи = Кп/Кп кр."""
    for n in (1.0, 1.2, 1.4, 1.6, 2.0):
        for b in (0.99, 0.9, 0.8, 0.7):
            if b < an.beta_cr_const_n(n):
                continue
            lit = float(an.Kb_const_n_literal(b, n))
            rat = float(an.Kb_const_n(b, n))
            assert abs(lit - rat) < 1e-9, f'n={n}, beta={b}: {lit} != {rat}'
            assert lit <= 1.0 + 1e-12


def test_mach_formulas_reduce_at_M0():
    """Таблица Е.2 при M = 0 переходит в таблицу Е.1."""
    for n in (1.0, 1.2, 1.4, 1.67):
        assert approx(an.beta_cr_const_n(n, 0.0), an.beta_cr_const_n(n), 1e-12)
        assert approx(an.Kn_cr_const_n(n, 0.0), an.Kn_cr_const_n(n), 1e-12)


def test_beta_cr_M_n1():
    """(Е.2.20), (Е.2.22): при n = 1  beta_крМ = Кп крМ = exp((M^2-1)/2)."""
    for M in (0.0, 0.3, 0.7, 1.0):
        ref = math.exp(0.5 * (M ** 2 - 1.0))
        assert approx(an.beta_cr_const_n(1.0, M), ref, 1e-12)
        assert approx(an.Kn_cr_const_n(1.0, M), ref, 1e-12)


def test_Kn_cr_M_identity():
    """(Е.2.21): Кп крМ = sqrt(n*beta_крМ^((n+1)/n)) — сверка с развёрнутой формой."""
    for n in (1.1, 1.4, 1.67, 2.0):
        for M in (0.0, 0.25, 0.5):
            A = 1.0 + 0.5 * (n - 1.0) * M ** 2
            expanded = (math.sqrt(2.0 * n / (n + 1.0))
                        * (2.0 / (n + 1.0)) ** (1.0 / (n - 1.0))
                        * A ** ((n + 1.0) / (2.0 * (n - 1.0))))
            assert approx(an.Kn_cr_const_n(n, M), expanded, 1e-10), f'n={n}, M={M}'


# =========================================================================
#  2. Омега-метод: Е.2.10-Е.2.14
# =========================================================================
def test_omega_equals_const_n_at_omega1():
    """При omega = 1 (n = 1) омега-метод и постоянный n дают одно и то же."""
    ref = 1.0 / math.sqrt(math.e)
    assert abs(an.beta_cr_omega(1.0) - ref) < 1e-6
    assert abs(an.Kn_cr_omega(1.0) - ref) < 1e-6


def test_beta_cr_omega_vs_approximation():
    """
    Численный поиск максимума Кп(beta) против аппроксимации (Е.2.12).
    Аппроксимация стандарта должна совпадать с точным решением
    в пределах инженерной точности.
    """
    for om in (0.5, 1.0, 2.0, 5.0, 10.0, 25.0):
        exact = an.beta_cr_omega(om)
        appr = an.beta_cr_omega_approx(om)
        assert abs(exact - appr) / exact < 5e-3, (
            f'omega={om}: точно {exact:.5f}, (Е.2.12) {appr:.5f}')


def test_Kn_cr_omega_identity():
    """(Е.2.13): Кп кр = beta_кр/sqrt(omega) должно равняться максимуму Кп(beta)."""
    for om in (0.5, 1.0, 2.0, 5.0, 20.0):
        b = an.beta_cr_omega(om)
        k_id = b / math.sqrt(om)
        grid = np.linspace(1e-4, 0.9999, 60000)
        k_max = float(np.max(an.Kn_omega(grid, om)))
        assert abs(k_id - k_max) / k_max < 2e-4, f'omega={om}'


def test_Kb_omega_le_one():
    """(Е.2.14): Кb <= 1 (Д.4: «где Кb < 1»)."""
    for om in (0.5, 1.0, 5.0):
        b_cr = an.beta_cr_omega(om)
        for b in np.linspace(min(0.999, b_cr + 0.3), 0.999, 5):
            assert float(an.Kb_omega(b, om)) <= 1.0 + 1e-9


# =========================================================================
#  3. Интегратор Е.1.4 против аналитики табл. Е.1
# =========================================================================
def test_integrator_reproduces_table_E1():
    """
    Метод прямого интегрирования (Е.1.4) на идеальном газе должен
    воспроизводить beta_кр (Е.2.4) и Кп кр (Е.2.6).
    """
    for n in (1.0, 1.1, 1.2, 1.3, 1.4, 1.6, 1.8, 2.0):
        gas = th.IdealGas(n=n, P1=1.0, rho1=1.0)
        sol = integ.solve(gas, 1.0, 0.0, 0.25, n_points=40000)
        assert sol.regime == integ.REGIME_CRITICAL, f'n={n}'
        assert abs(sol.beta_cr - an.beta_cr_const_n(n)) < 2e-4, (
            f'n={n}: beta_кр {sol.beta_cr:.5f} против {an.beta_cr_const_n(n):.5f}')
        assert abs(sol.Kn - an.Kn_cr_const_n(n)) / an.Kn_cr_const_n(n) < 2e-4, (
            f'n={n}: Кп кр {sol.Kn:.5f} против {an.Kn_cr_const_n(n):.5f}')


def test_integrator_nitrogen_reference():
    """Азот, n = 1,40: beta_кр = 0,528 (табл. И.1), Кп кр = 0,6847 (Е.2.6)."""
    gas = th.IdealGas(n=1.40, P1=1.0, rho1=1.0)
    sol = integ.solve(gas, 1.0, 0.0, 0.30, n_points=40000)
    assert abs(sol.beta_cr - 0.528) < 1e-3
    assert abs(sol.Kn - 0.6847) < 5e-4


def test_integrator_subcritical():
    """При beta > beta_кр интегратор должен дать докритический режим."""
    gas = th.IdealGas(n=1.40, P1=1.0, rho1=1.0)
    sol = integ.solve(gas, 1.0, 0.0, 0.80, n_points=20000)
    assert sol.regime == integ.REGIME_SUBCRITICAL
    # (Е.2.2) Кп при beta = 0,8
    assert abs(sol.Kn - float(an.Kn_const_n(0.8, 1.40))) / sol.Kn < 1e-3


def test_trapezoid_formula_sign():
    """
    (Е.1.4): интеграл -2*INT(P1..P0) dP/rho должен быть положителен и
    для rho = const равняться 2*(P1 - P0)/rho.
    """
    P = np.linspace(10.0, 2.0, 101)
    rho = np.full_like(P, 4.0)
    g, I = integ.mass_flux_curve(P, rho)
    assert approx(I[-1], 2.0 * (10.0 - 2.0) / 4.0, 1e-12)
    # (Е.2.1) Кп = sqrt(2*(1-beta)) для несжимаемой жидкости
    Kn = g[-1] / math.sqrt(P[0] * rho[0])
    assert approx(Kn, float(an.Kn_incompressible(2.0 / 10.0)), 1e-12)


def test_first_maximum_monotonic():
    """Монотонно растущая кривая -> докритический режим (Е.1.2)."""
    assert integ.first_maximum_index(np.array([0.0, 1.0, 2.0, 3.0])) is None
    assert integ.first_maximum_index(np.array([0.0, 1.0, 2.0, 1.5, 3.0])) == 2


# =========================================================================
#  4. Цепочка интервалов Е.2.3
# =========================================================================
def test_chain_single_interval_equals_table_E1():
    """Цепочка из одного интервала = обычный расчёт по табл. Е.1."""
    n = 1.4
    iv = an.Interval('n', n, 1.0, 1.0, 0.2)
    res = an.solve_chain([iv], 0.2)
    assert res['regime'] == 'критический'
    assert abs(res['beta_cr'] - an.beta_cr_const_n(n)) < 1e-9
    assert abs(res['Kn'] - an.Kn_cr_const_n(n)) < 1e-9


def test_chain_two_intervals_continuity():
    """
    Разбиение однородного газа на два интервала с ОДИНАКОВЫМ n не должно
    менять результат: это проверка формул с числом Маха (Е.2.16-Е.2.21).
    """
    n = 1.4
    P1, rho1 = 1.0, 1.0
    Pm = 0.85                                   # точка деления выше beta_кр
    rho_m = rho1 * (Pm / P1) ** (1.0 / n)
    one = an.solve_chain([an.Interval('n', n, P1, rho1, 0.2)], 0.2)
    two = an.solve_chain([an.Interval('n', n, P1, rho1, Pm),
                          an.Interval('n', n, Pm, rho_m, 0.2)], 0.2)
    assert abs(one['G_ideal'] - two['G_ideal']) / one['G_ideal'] < 1e-6
    assert abs(one['beta_cr'] - two['beta_cr']) < 1e-6


def test_chain_integrator_agreement_two_n():
    """
    Цепочка Г-2Ф (n1 в газе, n2 в 2Ф-области) против прямого интегрирования
    на синтетической изоэнтропе с изломом.
    """
    n1, n2, P1, rho1, Pm = 1.40, 1.10, 1.0, 1.0, 0.75
    rho_m = rho1 * (Pm / P1) ** (1.0 / n1)

    class Kinked:
        P_crit, P_triple = float('inf'), 0.0

        def rho_Ps(self, P, s=None):
            P = np.asarray(P, float)
            return np.where(P >= Pm,
                            rho1 * (P / P1) ** (1.0 / n1),
                            rho_m * (P / Pm) ** (1.0 / n2))

        def phase_crossings(self, a, b, s=None):
            return [Pm]

        def n_isentropic(self, P, s=None, side='center', rel_step=2e-4):
            return n1 if P > Pm else n2

    sol = integ.solve(Kinked(), 1.0, 0.0, 0.2, n_points=60000)
    chain = an.solve_chain([an.Interval('n', n1, P1, rho1, Pm),
                            an.Interval('n', n2, Pm, rho_m, 0.2)], 0.2)
    assert abs(sol.G_ideal - chain['G_ideal']) / sol.G_ideal < 2e-3, (
        f'{sol.G_ideal:.5f} против {chain["G_ideal"]:.5f}')


# =========================================================================
#  5. Коэффициент Kv: Д.12, Д.13, Д.7
# =========================================================================
def test_Kv_sewing_at_Re_1000():
    """Сшивка (Д.12) и (Д.13) на границе Re = 1000."""
    hi = cf.Kv_from_Re(1000.0)                       # по (Д.12)
    lo = 0.975 * math.sqrt(1.0 / (170.0 / 1000.0 + 0.980))   # по (Д.13)
    assert abs(hi - 0.9130) < 1e-3, hi
    assert abs(lo - 0.9092) < 1e-3, lo
    assert abs(hi - lo) / hi < 0.01, f'{hi:.4f} против {lo:.4f}'


def test_Kv_never_exceeds_one():
    """Kv — коэффициент УМЕНЬШЕНИЯ пропускной способности (Д.7): Kv <= 1."""
    for Re in (10, 50, 100, 500, 1e3, 5e3, 1e4, 5e4, 9.9e4, 1e5, 1e7):
        assert cf.Kv_from_Re(Re) <= 1.0 + 1e-12, f'Re={Re}'


def test_Kv_monotonic_and_limits():
    """Kv растёт с Re и при Re >= 1e5 равен 1,0 (Д.7)."""
    Re_list = [20, 100, 300, 1e3, 3e3, 1e4, 3e4, 9e4]
    vals = [cf.Kv_from_Re(x) for x in Re_list]
    assert all(b > a for a, b in zip(vals, vals[1:])), vals
    assert cf.Kv_from_Re(1e5) == 1.0
    assert cf.Kv_from_Re(1e9) == 1.0


def test_Kv_reference_points():
    """Контрольные значения (Д.12)/(Д.13)."""
    assert abs(cf.Kv_from_Re(100.0) - 0.5956) < 1e-3
    assert abs(cf.Kv_from_Re(1e4) - 0.9779) < 1e-3
    assert abs(cf.Kv_from_Re(3e4) - 0.9899) < 1e-3


def test_Kv_full_D11_consistency():
    """(Д.11) при типичном d0/DN1 должен быть близок к (Д.12)."""
    for Re in (2e3, 1e4, 5e4):
        a = cf.Kv_full(Re, d0=0.5, DN1=1.0)
        b = cf.Kv_from_Re(Re)
        assert abs(a - b) / b < 0.10, f'Re={Re}: {a:.4f} против {b:.4f}'


def test_reynolds_forms_agree():
    """(Д.14): две записи числа Рейнольдса эквивалентны."""
    G, mu, d0 = 0.16, 7.5e-6, 0.010
    F = math.pi * d0 ** 2 / 4.0
    assert approx(cf.reynolds_from_G(G, d0, mu), cf.reynolds_from_G_F(G, F, mu), 1e-9)


# =========================================================================
#  6. Коэффициент Kw: таблица Д.1
# =========================================================================
def test_Kw_unbalanced_is_one():
    """Примечание 2 к табл. Д.1: для неразгруженных клапанов Kw = 1,0."""
    for r in (0.0, 0.2, 0.45):
        val, _ = cf.Kw(r, 1.15, medium='газ', balanced=False)
        assert val == 1.0


def test_Kw_continuity_at_thresholds():
    """Непрерывность формул табл. Д.1 на порогах 0,150 / 0,300 / 0,377."""
    assert abs(cf._Kw_liquid(0.150) - 1.0) < 1e-4
    assert abs(cf._Kw_gas_110(0.300) - 1.0) < 3e-3
    assert abs(cf._Kw_gas_115(0.377) - 1.0) < 2e-3


def test_Kw_liquid_known_discontinuity():
    """
    В (Д.21) на границе Рп/Рно = 0,250 формулы стандарта дают небольшой
    разрыв (0,9167 против 0,9020). Фиксируем его как свойство стандарта.
    """
    left = 0.8750 + 1.8333 * 0.250 - 6.6667 * 0.250 ** 2
    right = 1.1490 - 0.9880 * 0.250
    assert abs(left - 0.91666) < 1e-4
    assert abs(right - 0.90200) < 1e-4
    assert abs(left - right) < 0.02


def test_Kw_below_threshold_is_one():
    """Ниже порогов Kw = 1,0 (Д.21-Д.24)."""
    assert cf.Kw(0.10, 1.15, medium='газ', balanced=True)[0] == 1.0
    assert cf.Kw(0.30, 1.15, medium='газ', balanced=True)[0] == 1.0
    assert cf.Kw(0.10, 1.10, medium='жидкость', balanced=True)[0] == 1.0
    assert cf.Kw(0.49, 1.25, medium='газ', balanced=True)[0] == 1.0


def test_Kw_interpolation_endpoints():
    """(Д.25), (Д.26): на концах интервалов интерполяция даёт исходные формулы."""
    r = 0.42
    assert abs(cf.Kw(r, 1.10, balanced=True)[0] - cf._Kw_gas_110(r)) < 1e-12
    assert abs(cf.Kw(r, 1.15, balanced=True)[0] - cf._Kw_gas_115(r)) < 1e-12
    assert abs(cf.Kw(r, 1.20, balanced=True)[0] - cf._Kw_gas_120(r)) < 1e-12
    mid = cf.Kw(r, 1.125, balanced=True)[0]
    assert min(cf._Kw_gas_110(r), cf._Kw_gas_115(r)) <= mid <= max(
        cf._Kw_gas_110(r), cf._Kw_gas_115(r))


def test_Kw_decreases_with_backpressure():
    """Рост противодавления не должен увеличивать Kw."""
    vals = [cf.Kw(r, 1.15, balanced=True)[0] for r in (0.30, 0.40, 0.45, 0.50)]
    assert all(b <= a + 1e-12 for a, b in zip(vals, vals[1:])), vals


# =========================================================================
#  7. Давления: п. 5.4
# =========================================================================
def test_P_po_three_ranges():
    """
    (1) Рн < 0,3 МПа  -> Рпо = Рн + 0,05 МПа
    (2) 0,3...6,0 МПа -> Рпо = 1,15*Рн
    (3) Рн > 6,0 МПа  -> Рпо = 1,10*Рн
    """
    assert approx(pr.P_po_gauge(0.2e6)[0], 0.25e6, 1e-12)
    assert approx(pr.P_po_gauge(0.5e6)[0], 0.575e6, 1e-12)
    assert approx(pr.P_po_gauge(3.0e6)[0], 3.45e6, 1e-12)
    assert approx(pr.P_po_gauge(6.0e6)[0], 6.90e6, 1e-12)
    assert approx(pr.P_po_gauge(8.0e6)[0], 8.80e6, 1e-12)


def test_P_po_regression_against_old_bug():
    """
    Регрессия на исходную ошибку: для Рн = 0,5 МПа применялся коэффициент
    1,10 из диапазона (3) вместо 1,15 из диапазона (2).
    """
    correct = pr.P_po_gauge(0.5e6)[0]
    buggy = 1.10 * 0.5e6
    assert correct > buggy
    assert abs(correct / buggy - 1.0) - 0.04545 < 1e-4


def test_pressure_set_definitions():
    """(Д.4.2): Р1 = Рпо + Ратм; Р2 = Рп + Ратм."""
    ps = pr.build_pressures(P_n_g=5e5, P_back_static_g=0.0, P_atm=101320.0)
    assert approx(ps.P1, 1.15 * 5e5 + 101320.0, 1e-12)
    assert approx(ps.P2, 101320.0, 1e-12)
    assert approx(ps.ratio_po_n, 1.15, 1e-12)


def test_dP_in_reduces_P1():
    """(8.2.1): потери на входе учитываются при расчёте пропускной способности."""
    a = pr.build_pressures(P_n_g=5e5, dP_in=0.0)
    b = pr.build_pressures(P_n_g=5e5, dP_in=0.02e5)
    assert b.P1 < a.P1
    assert approx(a.P1 - b.P1, 0.02e5, 1e-9)


def test_check_flags_violations():
    """Проверки должны ловить Рн <= Рр и Рпо > Рав max."""
    ps = pr.build_pressures(P_n_g=3.0e5)
    checks = pr.check_pressures(ps, P_work_g=4.0e5, P_design_g=3.2e5,
                                P_av_max_g=3.3e5)
    fails = [c for c in checks if c.severity == pr.SEVERITY_FAIL]
    assert len(fails) >= 2, [str(c) for c in checks]


# =========================================================================
#  8. Константы перевода единиц: Д.7/Д.9, Е.1.8/Е.1.10
# =========================================================================
def test_unit_constants_derivation():
    """
    G[кг/ч] = 3600*G[кг/с], F[м2] = F[мм2]*1e-6, sqrt(P[Па]) = sqrt(P[ед])*sqrt(коэф)
      МПа: 3600*1e-6*sqrt(1e6) = 3,60   (Д.7, Е.1.8)
      бар: 3600*1e-6*sqrt(1e5) = 1,1384 (Д.9, Е.1.10)
    """
    assert approx(3600 * 1e-6 * math.sqrt(1e6), sz.C_MPA, 1e-12)
    assert abs(3600 * 1e-6 * math.sqrt(1e5) - sz.C_BAR) < 5e-4


def test_dimensional_forms_agree():
    """СИ-расчёт (Д.1) и размерные формы (Д.7)/(Д.9) должны совпадать."""
    alpha, Kc_, Kv_, Kw_ = 0.75, 1.0, 1.0, 1.0
    P1, rho1, Kn = 6.763e5, 23.0, 0.68
    F_mm2 = 78.5
    G_si = sz.capacity(alpha, Kc_, Kv_, Kw_, Kn * math.sqrt(P1 * rho1), F_mm2 * 1e-6)
    G_bar = sz.capacity_kgh_bar(alpha, Kc_, Kv_, Kw_, Kn, F_mm2, P1 / 1e5, rho1)
    G_mpa = sz.capacity_kgh_mpa(alpha, Kc_, Kv_, Kw_, Kn, F_mm2, P1 / 1e6, rho1)
    assert abs(G_bar - G_si * 3600) / (G_si * 3600) < 1e-3
    assert abs(G_mpa - G_si * 3600) / (G_si * 3600) < 1e-6


def test_sizing_roundtrip():
    """(Д.1) и (Д.2) должны быть взаимно обратны."""
    G_ideal, alpha, Kc_, Kv_, Kw_, N = 2739.0, 0.75, 1.0, 1.0, 1.0, 1
    F = 78.5e-6
    G = sz.capacity(alpha, Kc_, Kv_, Kw_, G_ideal, F) * N
    F_back = sz.required_area(G, alpha, Kc_, Kv_, Kw_, G_ideal, N)
    assert abs(F_back - F) / F < 1e-12


def test_catalog_selection_rule():
    """(5.5): выбирается площадь, равная или ближайшая большая."""
    cat = (6, 8, 10, 12, 16, 20)
    F_req = math.pi * 9.0 ** 2 / 4.0 * 1e-6         # между d0 = 8 и d0 = 10
    d0, F = sz.select_from_catalog(F_req, cat)
    assert d0 == 10.0
    assert F >= F_req
    F_big = math.pi * 30.0 ** 2 / 4.0 * 1e-6
    assert sz.select_from_catalog(F_big, cat) == (None, None)


# =========================================================================
#  9. Свойства азота и классификация режима (требуют CoolProp)
# =========================================================================
def _nitrogen():
    return th.Fluid('Nitrogen')


def test_nitrogen_critical_point():
    """Справочные значения критической точки азота (ср. табл. И.1: 33,94 бар, 126,05 К)."""
    f = _nitrogen()
    assert abs(f.P_crit / 1e5 - 33.96) < 0.1
    assert abs(f.T_crit - 126.19) < 0.3
    assert abs(f.R_specific - 296.8) < 1.0        # табл. И.1: Rуд = 298 Дж/(кг·К)


def test_nitrogen_phase_detection():
    """Определение фазы по энтропии (Е.3.3) без «глушения» исключений."""
    f = _nitrogen()
    P = 6.763e5
    T_sat = f.T_sat(P)
    s_gas = f.s_PT(P, T_sat + 30.0)
    s_liq = f.s_PT(P, T_sat - 20.0)
    s_mid = 0.5 * (f.s_liq_sat(P) + f.s_gas_sat(P))
    assert f.phase_Ps(P, s_gas) == th.PHASE_GAS
    assert f.phase_Ps(P, s_liq) == th.PHASE_LIQUID
    assert f.phase_Ps(P, s_mid) == th.PHASE_TWOPHASE
    assert abs(f.quality_Ps(P, s_mid) - 0.5) < 1e-9


def test_T_sat_above_critical_raises():
    """Р1 > Ркр среды должно давать внятную ошибку, а не трейсбек CoolProp."""
    f = _nitrogen()
    try:
        f.T_sat(f.P_crit * 1.2)
    except th.ThermoError as exc:
        assert 'сверхкритическая' in str(exc)
    else:
        raise AssertionError('ожидалась ThermoError')


def test_nitrogen_regime_vs_superheat():
    """
    Азот — «регулярная» среда (3.1.25): при малом перегреве изоэнтропа
    пересекает линию конденсации (режим Г-2Ф, Д.10 п. 4), при большом —
    остаётся в области газа (Г-Г, Д.10 п. 3).
    """
    from pk_gost import regime as rg
    f = _nitrogen()
    P1 = 1.15 * 5e5 + 101320.0
    P2 = 101320.0
    T_sat = f.T_sat(P1)

    for dT in (0.5, 5.0):
        info = rg.classify(f, P1, T_sat + dT, P2)
        assert info.kinematic == rg.G2F, f'dT={dT}: {info.kinematic}'
        sol = integ.solve(f, P1, f.s_PT(P1, T_sat + dT), P2, n_points=8000)
        assert info.P_dew > sol.P0, (
            f'dT={dT}: запирание должно быть НИЖЕ точки росы')

    for dT in (20.0, 40.0):
        info = rg.classify(f, P1, T_sat + dT, P2)
        sol = integ.solve(f, P1, f.s_PT(P1, T_sat + dT), P2, n_points=8000)
        rg.refine_with_solution(info, sol)
        assert rg.GG in info.effective, f'dT={dT}: {info.effective}'


def test_nitrogen_choke_on_phase_boundary():
    """
    При перегреве ~10 К клапан запирается практически на линии конденсации —
    особый случай Е.1.5 (равенство G* = sqrt(n*Ркр*rho_кр) не выполняется,
    выполняется только неравенство).
    """
    f = _nitrogen()
    P1 = 1.10 * 5e5 + 101320.0
    P2 = 101320.0
    s1 = f.s_PT(P1, f.T_sat(P1) + 10.0)
    P_dew = f.dew_pressure(P1, P2, s1)
    sol = integ.solve(f, P1, s1, P2, n_points=20000)
    assert abs(sol.P0 - P_dew) / P1 < 5e-3, (
        f'Ркр = {sol.P0/1e5:.4f}, Pросы = {P_dew/1e5:.4f} бар')
    assert sol.on_phase_boundary
    sn = integ.check_sonic(sol)
    assert not sn['equality']
    assert sn['n_plus'] > sn['n_minus'], 'показатель изоэнтропы должен падать'
    assert sn['ok'], (f'{sn["lower"]:.1f} <= {sn["G_ideal"]:.1f} <= {sn["upper"]:.1f}')


def test_nitrogen_GG_cross_check():
    """
    При устойчивом Г-Г прямое интегрирование (Е.1) и постоянный показатель
    изоэнтропы (Е.2.2) должны сойтись в пределах 1 %.
    """
    from pk_gost import regime as rg, report as rp
    f = _nitrogen()
    P1 = 1.15 * 5e5 + 101320.0
    P2 = 101320.0
    T1 = f.T_sat(P1) + 25.0
    s1 = f.s_PT(P1, T1)
    info = rg.classify(f, P1, T1, P2, s1)
    sol = integ.solve(f, P1, s1, P2, n_points=20000)
    intervals, _ = rp.build_intervals(f, P1, s1, P2, info)
    res = an.solve_chain(intervals, P2)
    dev = abs(res['G_ideal'] - sol.G_ideal) / sol.G_ideal
    assert dev < 0.01, f'расхождение {dev*100:.2f} %'


def test_relief_load_definition():
    """(Г.2): Gав = Q/h_gl."""
    from pk_gost import scenario as scn
    f = _nitrogen()
    P1 = 6.763e5
    T1 = f.T_sat(P1) + 20.0
    ld = scn.relief_load(f, 20_000.0, P1, T1)
    assert approx(ld.G_av, 20_000.0 / f.h_fg(P1), 1e-12)
    assert ld.G_av > ld.G_av_enthalpy, 'вариант по h_gl должен быть консервативнее'


# =========================================================================
#  Запуск без pytest
# =========================================================================
def _main():
    tests = [(n, o) for n, o in sorted(globals().items())
             if n.startswith('test_') and callable(o)]
    passed, failed = 0, []
    for name, fn in tests:
        try:
            fn()
            passed += 1
            print(f'  OK    {name}')
        except Exception as exc:
            failed.append((name, exc))
            print(f'  FAIL  {name}: {exc}')
    print()
    print(f'Пройдено {passed} из {len(tests)}')
    if failed:
        print('Провалено:')
        for name, exc in failed:
            print(f'  {name}: {exc}')
        return 1
    return 0


if __name__ == '__main__':
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.exit(_main())
