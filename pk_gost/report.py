# -*- coding: utf-8 -*-
"""
Оркестровка расчёта и пошаговый протокол.

Последовательность соответствует таблице 2 ГОСТ 12.2.085-2017:
    определение причин повышенного давления
    -> фазовый состав среды на входе
    -> аварийный расход при сбросе через ПК
    -> расчёт массовой скорости и расчёт ПК
    -> расчёт элементов подводящей и сбросной линий
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from . import analytic, coeffs as cf, integrate as integ, pressures as pr
from . import regime as rg, scenario as sc, sizing as sz, thermo as th, walls as wl

LINE = '=' * 78
THIN = '-' * 78


@dataclass
class CaseResult:
    case: object
    fluid: object
    ps: pr.PressureSet
    state: dict
    load: sc.ReliefLoad
    info: rg.RegimeInfo
    sol: integ.FlowSolution
    sonic: dict | None
    cross: dict | None
    alpha: float
    alpha_note: str
    Kc: float
    Kw: float
    Kw_note: str
    sizing: sz.SizingResult
    walls: wl.WallTemperatures | None
    lines: list = field(default_factory=list)


# =========================================================================
#  Построение интервалов для аналитической перекрёстной проверки (Е.2.3)
# =========================================================================
def build_intervals(fluid, P1, s1, P2, info: rg.RegimeInfo):
    """
    Разбиение изоэнтропы на интервалы с уравнениями состояния по табл. Д.2.

    Г-Г   : один интервал, постоянный показатель изоэнтропы (Е.2.2)
    Г-2Ф  : два интервала с разными n для газа и двухфазной области (Е.2.3.2)
    2Ф-2Ф : омега-метод (Е.2.2, омега), параметр омега по Е.3.11
    Ж-Ж   : несжимаемая жидкость (Е.2.1)
    Ж-2Ф  : несжимаемая жидкость + омега-метод (Е.2.3.1)
    """
    rho1 = fluid.rho_Ps(P1, s1)

    if info.kinematic == rg.GG:
        n1 = fluid.n_isentropic(P1, s1, side='center')
        return [analytic.Interval('n', n1, P1, rho1, P2, 'газ')], f'n = {n1:.4f} (Е.3.1)'

    if info.kinematic == rg.G2F:
        Pd = info.P_dew
        n1 = fluid.n_isentropic(P1, s1, side='center')
        n2 = fluid.n_isentropic(Pd, s1, side='low')
        rho_d = fluid.rho_Ps(Pd, s1)
        return ([analytic.Interval('n', n1, P1, rho1, Pd, 'газ'),
                 analytic.Interval('n', n2, Pd, rho_d, P2, 'двухфазная область')],
                f'n1 = {n1:.4f} (газ), n2 = {n2:.4f} (2Ф) — Е.2.3.2')

    if info.kinematic == rg.F2F2:
        om = fluid.omega_two_point(P1, s1)
        return ([analytic.Interval('omega', om, P1, rho1, P2, 'двухфазная область')],
                f'omega = {om:.4f} (Е.3.11, двухточечный метод)')

    if info.kinematic == rg.ZZ:
        return ([analytic.Interval('rho', None, P1, rho1, P2, 'жидкость')],
                'rho = const (Д.27)')

    if info.kinematic == rg.Z2F:
        Pb = info.P_bubble
        rho_b = fluid.rho_Ps(Pb, s1)
        om = fluid.omega_two_point(Pb, s1)
        return ([analytic.Interval('rho', None, P1, rho1, Pb, 'жидкость'),
                 analytic.Interval('omega', om, Pb, rho_b, P2, 'двухфазная область')],
                f'rho = const + omega = {om:.4f} (Е.2.3.1)')

    # сверхкритическая область — постоянный показатель изоэнтропы
    n1 = fluid.n_isentropic(P1, s1, side='center')
    return ([analytic.Interval('n', n1, P1, rho1, P2, 'суперкритическая среда')],
            f'n = {n1:.4f} (Е.3.1)')


# =========================================================================
#  Основной расчёт
# =========================================================================
def run_case(case, n_points=20000, verbose_thermo=True):
    """Полный расчёт одного случая. Возвращает CaseResult."""
    fluid = th.Fluid(case.fluid)

    # --- 1. Давления (5.4, Д.4.2) ---
    ps = pr.build_pressures(
        P_n_g=case.P_n_g, P_back_static_g=case.P_back_static_g,
        P_back_dyn_g=case.P_back_dyn_g, dP_in=case.dP_in,
        P_no_g=case.P_no_g, P_atm=case.P_atm, P_po_g=case.P_po_g)

    # --- 2. Проверки (табл. 3, 8.2.1, 4.1.1/Б.3.4) ---
    pr.check_pressures(ps, P_work_g=case.P_work_g, P_design_g=case.P_design_g,
                       P_av_max_g=case.P_av_max_g, P_allowed_g=case.P_allowed_g,
                       fluid=fluid, balanced=case.balanced)

    # --- 3. Состояние среды перед клапаном ---
    P1, P2, T1 = ps.P1, ps.P2, case.T1
    T_sat = None
    try:
        T_sat = fluid.T_sat(P1)
    except th.ThermoError:
        pass
    state = dict(
        P1=P1, P2=P2, T1=T1, T_sat=T_sat,
        superheat=None if T_sat is None else T1 - T_sat,
        rho1=fluid.rho_PT(P1, T1), s1=fluid.s_PT(P1, T1),
        h1=fluid.h_PT(P1, T1), mu1=fluid.mu_PT(P1, T1),
        Z1=fluid.Z_PT(P1, T1), k1=fluid.k_PT(P1, T1),
        Tr=T1 / fluid.T_crit, Pr=P1 / fluid.P_crit,   # (Д.30), (Д.31)
    )
    s1 = state['s1']

    # --- 4. Аварийный расход (Г.2) ---
    load = sc.relief_load(fluid, case.Q_heat, P1, T1)

    # --- 5. Режим течения (Д.10) и метод (табл. Д.2) ---
    info = rg.classify(fluid, P1, T1, P2, s1)

    # --- 6. Массовая скорость: прямое интегрирование (Е.1) ---
    sol = integ.solve(fluid, P1, s1, P2, n_points=n_points)
    sonic = integ.check_sonic(sol)
    rg.refine_with_solution(info, sol)

    # перекрёстная проверка аналитическим методом табл. Д.2
    cross = None
    try:
        intervals, desc = build_intervals(fluid, P1, s1, P2, info)
        res = analytic.solve_chain(intervals, P2)
        cross = dict(res)
        cross['desc'] = desc
        cross['intervals'] = intervals
        cross['dev_pct'] = abs(res['G_ideal'] - sol.G_ideal) / sol.G_ideal * 100.0
    except Exception as exc:
        cross = dict(error=str(exc))

    # --- 7. Коэффициенты (Д.5, Д.6, Д.8) ---
    alpha, alpha_note = cf.alpha_for_regime(
        sol.regime, case.alpha1, case.alpha2, case.seat_pressure_known)
    Kc_val = cf.Kc(case.membrane_device)
    medium = 'газ' if info.inlet_phase == th.PHASE_GAS else (
        'двухфазная' if info.inlet_phase == th.PHASE_TWOPHASE else 'жидкость')
    Kw_val, Kw_note = cf.Kw(ps.ratio_back_no, ps.ratio_po_n,
                            medium=medium, balanced=case.balanced,
                            regime=sol.regime)

    # --- 8. Площадь седла (Д.2) и подбор по каталогу (5.5) ---
    sizing = sz.size_valve(
        G_av=load.G_av, G_ideal=sol.G_ideal, alpha=alpha, Kc_val=Kc_val,
        Kw_val=Kw_val, mu=state['mu1'], N=case.N,
        catalog_d0_mm=case.catalog_d0_mm)

    # --- 10. Температуры стенок (прил. Ж) ---
    walls_res, walls_error = None, None
    try:
        walls_res = wl.wall_temperatures(fluid, P1, T1, sol.P0, s1=s1,
                                         k_ideal=state.get('k1'))
    except Exception as exc:
        walls_error = str(exc)
    state['walls_error'] = walls_error

    return CaseResult(case=case, fluid=fluid, ps=ps, state=state, load=load,
                      info=info, sol=sol, sonic=sonic, cross=cross,
                      alpha=alpha, alpha_note=alpha_note, Kc=Kc_val,
                      Kw=Kw_val, Kw_note=Kw_note, sizing=sizing, walls=walls_res)


# =========================================================================
#  Печать протокола
# =========================================================================
def _bar(P):
    return P / 1e5


def format_report(r: CaseResult):
    """Пошаговый протокол со ссылками на пункты и формулы ГОСТ."""
    c, ps, st, out = r.case, r.ps, r.state, []
    A = out.append

    A(LINE)
    A(f'РАСЧЁТ ПРЕДОХРАНИТЕЛЬНОГО КЛАПАНА ПО ГОСТ 12.2.085-2017')
    A(f'{c.title}')
    A(LINE)

    # ---------- исходные ----------
    A('')
    A('ИСХОДНЫЕ ДАННЫЕ')
    A(THIN)
    A(f'  Среда                      {r.fluid.name}')
    A(f'    Ткр = {r.fluid.T_crit:8.3f} К,  Ркр = {_bar(r.fluid.P_crit):7.3f} бар абс.,  '
      f'Rуд = {r.fluid.R_specific:7.2f} Дж/(кг·К)')
    A(f'  Т1 (перед клапаном)        {c.T1:8.2f} К{c.mark("T1")}')
    A(f'  Аварийный теплоприток Q    {c.Q_heat:8.0f} Вт{c.mark("Q_heat")}')
    A(f'  Рн  давление настройки     {_bar(c.P_n_g):8.3f} бар изб.{c.mark("P_n_g")}')
    A(f'  Рр  рабочее давление       {_bar(c.P_work_g):8.3f} бар изб.{c.mark("P_work_g")}')
    A(f'  Р   расчётное давление     {_bar(c.P_design_g):8.3f} бар изб.{c.mark("P_design_g")}')
    A(f'  Рав max                    {_bar(c.P_av_max_g):8.3f} бар изб.{c.mark("P_av_max_g")}')
    A(f'  Рп ст / Рп дин             {_bar(c.P_back_static_g):8.3f} / '
      f'{_bar(c.P_back_dyn_g):.3f} бар изб.')
    A(f'  dPвх потери на входе       {_bar(c.dP_in):8.4f} бар')
    A(f'  alpha1 / alpha2            {c.alpha1:8.3f} / {c.alpha2:.3f}'
      f'{c.mark("alpha1")}')
    A(f'  Число клапанов N           {c.N:8d}{c.mark("N")}')
    A(f'  Клапан                     '
      f'{"разгруженный" if c.balanced else "неразгруженный"}{c.mark("balanced")}')
    if c.stubs:
        A('')
        A(f'  ! Помечено как ЗАГЛУШКА: {", ".join(sorted(c.stubs))}')
        A(f'    Заменить данными ЭД клапана (5.11) и паспорта сосуда.')

    # ---------- шаг 1 ----------
    A('')
    A('ШАГ 1. ДАВЛЕНИЯ (п. 5.4, Д.4.2)')
    A(THIN)
    A(f'  Рпо = {_bar(ps.P_po_g):.3f} бар изб.   {ps.formula_po}')
    A(f'      Рпо/Рн = {ps.ratio_po_n:.3f}  — ключ входа в табл. Д.1 для Kw')
    A(f'  Р1  = Рпо + Ратм{" - dPвх" if ps.dP_in else ""} = '
      f'{_bar(ps.P1):.4f} бар абс.')
    A(f'  Р2  = Рп  + Ратм = {_bar(ps.P2):.4f} бар абс.')
    A(f'  beta = Р2/Р1 = {ps.beta:.4f}')

    # ---------- шаг 2 ----------
    A('')
    A('ШАГ 2. ПРОВЕРКА СООТНОШЕНИЯ ДАВЛЕНИЙ (табл. 3, п. 8.2.1)')
    A(THIN)
    for ch in ps.checks:
        A(str(ch))
    bad = [x for x in ps.checks if x.severity == pr.SEVERITY_FAIL]
    if bad:
        A(f'  !! Нарушено требований: {len(bad)} — исходные данные недопустимы.')

    # ---------- шаг 3 ----------
    A('')
    A('ШАГ 3. СОСТОЯНИЕ СРЕДЫ ПЕРЕД КЛАПАНОМ')
    A(THIN)
    if st['T_sat'] is not None:
        A(f'  Тнас(Р1)     = {st["T_sat"]:8.2f} К')
        A(f'  Перегрев     = {st["superheat"]:8.2f} К')
    A(f'  rho1         = {st["rho1"]:8.3f} кг/м3')
    A(f'  s1           = {st["s1"]:8.2f} Дж/(кг·К)')
    A(f'  h1           = {st["h1"]/1e3:8.2f} кДж/кг')
    A(f'  mu1          = {st["mu1"]*1e6:8.3f} мкПа·с')
    A(f'  Z1           = {st["Z1"]:8.4f}')
    A(f'  k = cp/cv    = {st["k1"]:8.4f}   (ср. табл. И.1: для азота 1,40)')
    A(f'  Tr, Pr       = {st["Tr"]:8.4f}, {st["Pr"]:.4f}   (Д.30), (Д.31)')

    # ---------- шаг 4 ----------
    A('')
    A('ШАГ 4. АВАРИЙНЫЙ РАСХОД ОТ ТЕПЛОПРИТОКА (прил. Г.1.2, Г.2)')
    A(THIN)
    ld = r.load
    A(f'  h_gl(Р1)     = {ld.h_fg/1e3:8.2f} кДж/кг   — скрытая теплота парообразования')
    A(f'  Gав = Q/h_gl = {ld.G_av:8.4f} кг/с = {ld.G_av_kgh:.1f} кг/ч')
    A(f'  справочно: h(Р1,Т1) - h_l(Р1) = {ld.dh_total/1e3:.2f} кДж/кг '
      f'-> Gав\' = {ld.G_av_enthalpy*3600:.1f} кг/ч')
    A(f'  ({ld.note})')

    # ---------- шаг 5 ----------
    A('')
    A('ШАГ 5. РЕЖИМ ТЕЧЕНИЯ (п. Д.10) И ВЫБОР МЕТОДА (табл. Д.2)')
    A(THIN)
    inf = r.info
    A(f'  Состояние на входе         {inf.inlet_phase}')
    A(f'  Режим по Д.10 на [Р2, Р1]  {inf.kinematic}')
    if inf.P_dew:
        A(f'  Давление начала конденсации {_bar(inf.P_dew):.4f} бар абс. (Е.3.3.2)')
    if inf.P_bubble:
        A(f'  Давление вскипания          {_bar(inf.P_bubble):.4f} бар абс. (Е.3.3.1)')
    A(f'  Режим до сечения запирания {inf.effective}')
    A(f'  Метод по табл. Д.2         {inf.method}')
    A(f'                             [{inf.method_ref}]')
    A(f'  Основной расчёт            метод прямого интегрирования (Е.1) —')
    A(f'                             универсален для всех режимов (Е.1.1)')
    for note in inf.notes:
        for i, chunk in enumerate(_wrap(note, 72)):
            A(f'  {"* " if i == 0 else "  "}{chunk}')

    # ---------- шаг 6 ----------
    A('')
    A('ШАГ 6. МАССОВАЯ СКОРОСТЬ (прил. Е.1, метод прямого интегрирования)')
    A(THIN)
    s = r.sol
    A(f'  Точек сетки                {s.grid_points} (границы фаз включены — прим. к Е.1.4)')
    A(f'  Режим истечения            {s.regime.upper()}')
    if s.beta_cr:
        A(f'  Ркр = {_bar(s.P0):.4f} бар абс.,  beta_кр = Ркр/Р1 = {s.beta_cr:.4f}')
    A(f'  rho0 (в седле)             {s.rho0:8.3f} кг/м3')
    A(f'  -2*INT dP/rho              {s.integral/1e3:8.3f} кДж/кг')
    A(f'  G*ideal                    {s.G_ideal:8.1f} кг/(м2·с)      (Е.1.2)')
    A(f'  Кп = G*ideal/sqrt(Р1·rho1) {s.Kn:8.4f}                (Д.3)')

    if r.sonic:
        sn = r.sonic
        A('')
        A('  Контроль (Е.1.5):')
        if sn['equality']:
            A(f'    n_кр = {sn["n_plus"]:.4f};  sqrt(n·Ркр·rho_кр) = {sn["upper"]:.1f} '
              f'против G*ideal = {sn["G_ideal"]:.1f} кг/(м2·с)')
            A(f'    расхождение {sn["rel_dev"]*100:.2f} %  '
              f'[{"OK" if sn["ok"] else "ПРОВЕРИТЬ"}]')
        else:
            A(f'    Ркр лежит на границе фазовой диаграммы — показатель изоэнтропы')
            A(f'    и скорость звука терпят скачок, выполняется НЕРАВЕНСТВО:')
            A(f'      n+ (газ, P>Ркр)  = {sn["n_plus"]:.4f}  ->  sqrt(n+·Ркр·rho) = '
              f'{sn["upper"]:.1f}')
            A(f'      n- (2Ф,  P<Ркр)  = {sn["n_minus"]:.4f}  ->  sqrt(n-·Ркр·rho) = '
              f'{sn["lower"]:.1f}')
            A(f'      G*ideal          = {sn["G_ideal"]:.1f} кг/(м2·с)')
            A(f'      {sn["lower"]:.1f} <= {sn["G_ideal"]:.1f} <= {sn["upper"]:.1f}  '
              f'[{"OK" if sn["ok"] else "НАРУШЕНО"}]')

    # перекрёстная проверка
    A('')
    A('  Перекрёстная проверка аналитическим методом табл. Д.2:')
    cx = r.cross
    if cx and 'error' not in cx:
        A(f'    {cx["desc"]}')
        for iv in cx['intervals']:
            A(f'      {iv}')
        A(f'    режим {cx["regime"]}, G*ideal = {cx["G_ideal"]:.1f} кг/(м2·с), '
          f'Кп = {cx["Kn"]:.4f}')
        if cx['beta_cr']:
            A(f'    beta_кр = {cx["beta_cr"]:.4f}')
        if cx['Mach'] and len(cx['Mach']) > 1:
            A(f'    числа Маха на границах интервалов: '
              f'{", ".join(f"{m:.4f}" for m in cx["Mach"])}   (Е.2.16)')
        A(f'    расхождение с прямым интегрированием: {cx["dev_pct"]:.2f} %')
        if cx['note']:
            A(f'    {cx["note"]}')
    else:
        A(f'    не выполнена: {cx.get("error", "нет данных")}')

    # ---------- шаг 7 ----------
    A('')
    A('ШАГ 7. ПОПРАВОЧНЫЕ КОЭФФИЦИЕНТЫ (Д.5-Д.8)')
    A(THIN)
    zg = r.sizing
    A(f'  alpha = {r.alpha:.4f}   {r.alpha_note}')
    A(f'  Kc    = {r.Kc:.4f}   ' +
      ('мембранно-предохранительное устройство установлено (Д.6)'
       if c.membrane_device else 'мембранно-предохранительных устройств нет (Д.6)'))
    A(f'  Kw    = {r.Kw:.4f}   {r.Kw_note}')
    A(f'  Kv    = {zg.Kv:.4f}   Re = {zg.Re:.3e}'
      + (', Re >= 1e5 -> Kv = 1,0 (Д.7)' if zg.Re >= cf.RE_TURBULENT
         else f', итераций {zg.Kv_iterations} (Д.9.1)'))

    # ---------- шаг 8 ----------
    A('')
    A('ШАГ 8. МИНИМАЛЬНАЯ ПЛОЩАДЬ СЕДЛА (Д.2) И ПОДБОР (п. 5.5)')
    A(THIN)
    A(f'  F = Gав/(alpha·Kc·Kv·Kw·G*ideal·N)')
    A(f'    = {zg.G_av:.4f}/({r.alpha:.4f}·{r.Kc:.3f}·{zg.Kv:.4f}·{r.Kw:.4f}·'
      f'{zg.G_ideal:.1f}·{zg.N})')
    A(f'    = {zg.F_required_mm2:.3f} мм2   ->  d0 = {zg.d0_required_mm:.2f} мм')
    if zg.d0_selected_mm:
        A(f'  Каталог (5.5, «равной или ближайшей большей»):')
        A(f'    d0 = {zg.d0_selected_mm:.0f} мм,  F = {zg.F_selected_mm2:.2f} мм2')
    else:
        A(f'  ! В каталоге нет типоразмера с F >= {zg.F_required_mm2:.1f} мм2 — '
          f'увеличить N или пересмотреть каталог')

    # ---------- шаг 9 ----------
    A('')
    A('ШАГ 9. ПОВЕРКА ПРОПУСКНОЙ СПОСОБНОСТИ (Д.1, п. 5.1)')
    A(THIN)
    if zg.G_selected is not None:
        A(f'  G   = alpha·Kc·Kv·Kw·G*ideal·F·N = {zg.G_selected:.4f} кг/с = '
          f'{zg.G_selected*3600:.1f} кг/ч')
        A(f'  Gав = {zg.G_av:.4f} кг/с = {zg.G_av*3600:.1f} кг/ч')
        ok = zg.G_selected >= zg.G_av
        A(f'  G >= Gав : {"ДА" if ok else "НЕТ"},  запас {zg.margin_pct:+.1f} %')
        if zg.margin_pct is not None and zg.margin_pct > 10.0:
            A(f'  ! Запас > 10 %: по п. 6.7.1 рекомендуется либо ограничитель высоты')
            A(f'    подъёма золотника, либо пересмотр выпускной (сбросной) системы.')

        # контроль размерных форм (Д.7/Д.9, Е.1.8/Е.1.10)
        G_bar = sz.capacity_kgh_bar(r.alpha, r.Kc, zg.Kv, r.Kw, r.sol.Kn,
                                    zg.F_selected_mm2, _bar(ps.P1), st['rho1']) * zg.N
        G_mpa = sz.capacity_kgh_mpa(r.alpha, r.Kc, zg.Kv, r.Kw, r.sol.Kn,
                                    zg.F_selected_mm2, ps.P1 / 1e6, st['rho1']) * zg.N
        G_e110 = (sz.C_BAR * r.alpha * r.Kc * zg.Kv * r.Kw * zg.F_selected_mm2
                  * r.sol.rho0 * math.sqrt(r.sol.integral / 1e5)) * zg.N
        A('')
        A(f'  Контроль размерных форм (кг/ч, мм2):')
        A(f'    (Д.9)  константа 1,138, бар : {G_bar:9.1f} кг/ч')
        A(f'    (Д.7)  константа 3,60,  МПа : {G_mpa:9.1f} кг/ч')
        A(f'    (Е.1.10) через rho0 и интеграл: {G_e110:9.1f} кг/ч')
        A(f'    СИ-расчёт                     : {zg.G_selected*3600:9.1f} кг/ч')

    # ---------- шаг 10 ----------
    A('')
    A('ШАГ 10. ТЕМПЕРАТУРЫ СРЕДЫ И СТЕНОК (прил. Ж)')
    A(THIN)
    if r.walls:
        w = r.walls
        A(f'  Т0   в седле                    {w.T0:8.2f} К   (Ж.4, изоэнтропно до седла)')
        A(f'  Тторм температура торможения    {w.T_stag:8.2f} К   (Ж.5)')
        A(f'  r    коэффициент восстановления {w.r:8.3f}      (Ж.8, r = Pr^(1/3))')
        A(f'  Тст  температура стенки         {w.T_wall:8.2f} К   (Ж.7)')
        if w.T_wall_ideal is not None:
            A(f'  контроль по (Ж.9), идеальный газ: {w.T_wall_ideal:6.2f} К   '
              f'(расхождение {abs(w.T_wall_ideal - w.T_wall):.2f} К)')
        A(f'  -> Тст = {w.T_wall - 273.15:.1f} °C. На эту температуру вести выбор материала')
        A(f'     корпуса ПК и трубопроводов системы сброса, а также расчёт')
        A(f'     компенсации температурных удлинений и крепежа (8.1.3).')
        if w.note:
            for chunk in _wrap(w.note, 70):
                A(f'  ! {chunk}')
    else:
        A(f'  ! Расчёт не выполнен: {st.get("walls_error", "нет данных")}')

    # ---------- шаг 11 ----------
    A('')
    A('ШАГ 11. ДОПОЛНИТЕЛЬНЫЕ ТРЕБОВАНИЯ ДЛЯ КРИОГЕННОЙ СРЕДЫ')
    A(THIN)
    A('  п. 5.9  Для оборудования, работающего при криогенных температурах,')
    A('          предусматривают систему из РАБОЧЕГО И РЕЗЕРВНОГО клапанов равной')
    A('          пропускной способности с переключающим устройством, исключающим')
    A('          одновременное закрытие обоих.')
    A('  п. 5.12 Изготовитель должен привести Kt и минимальную температуру')
    A('          применения клапана; Kt — множитель к давлению настройки при')
    A('          стендовых испытаниях (5.13.2, ф. (4)).')
    A('  п. 8.3.1 Динамическое противодавление определяют для всех систем,')
    A('          включая сброс напрямую в атмосферу через короткие трубы (Б.1.3).')
    A('  п. 8.2.1 Уклон подводящего трубопровода в сторону сосуда; внутренний')
    A('          диаметр не менее диаметра входного патрубка ПК.')
    A(LINE)

    return '\n'.join(out)


def _wrap(text, width):
    words, line, out = text.split(), '', []
    for w in words:
        if len(line) + len(w) + 1 > width:
            out.append(line)
            line = w
        else:
            line = f'{line} {w}'.strip()
    if line:
        out.append(line)
    return out


# =========================================================================
#  Таблица чувствительности к перегреву
# =========================================================================
def superheat_table(case, superheats=(0.5, 5.0, 10.0, 20.0, 40.0), n_points=8000):
    """
    Влияние перегрева пара на режим течения и на требуемую площадь седла.

    Показывает, где проходит граница применимости расчёта по режиму Г-Г.
    """
    fluid = th.Fluid(case.fluid)
    ps = pr.build_pressures(P_n_g=case.P_n_g,
                            P_back_static_g=case.P_back_static_g,
                            P_back_dyn_g=case.P_back_dyn_g,
                            dP_in=case.dP_in, P_no_g=case.P_no_g,
                            P_atm=case.P_atm, P_po_g=case.P_po_g)
    P1, P2 = ps.P1, ps.P2
    T_sat = fluid.T_sat(P1)

    rows = []
    for dT in superheats:
        T1 = T_sat + dT
        s1 = fluid.s_PT(P1, T1)
        info = rg.classify(fluid, P1, T1, P2, s1)
        sol = integ.solve(fluid, P1, s1, P2, n_points=n_points)
        rg.refine_with_solution(info, sol)
        load = sc.relief_load(fluid, case.Q_heat, P1, T1)
        alpha, _ = cf.alpha_for_regime(sol.regime, case.alpha1, case.alpha2)
        Kw_val, _ = cf.Kw(ps.ratio_back_no, ps.ratio_po_n, medium='газ',
                          balanced=case.balanced, regime=sol.regime)
        F = sz.required_area(load.G_av, alpha, cf.Kc(case.membrane_device),
                             1.0, Kw_val, sol.G_ideal, case.N)
        # Давление насыщения ПРИ температуре Т1 — приводится только для
        # сравнения с Ркoнд, в расчёте не участвует.
        P_sat_T1 = None
        if T1 < fluid.T_crit:
            try:
                P_sat_T1 = fluid.P_sat(T1)
            except th.ThermoError:
                pass
        rows.append(dict(dT=dT, T1=T1, P_dew=info.P_dew, P_cr=sol.P0,
                         P_sat_T1=P_sat_T1,
                         beta_cr=sol.beta_cr, G_ideal=sol.G_ideal,
                         regime=info.effective, F_mm2=F * 1e6,
                         G_av_kgh=load.G_av_kgh))
    return T_sat, rows


def format_superheat_table(case, T_sat, rows):
    out = []
    A = out.append
    A('')
    A(LINE)
    A('ТАБЛИЦА ЧУВСТВИТЕЛЬНОСТИ К ПЕРЕГРЕВУ ПАРА НА ВХОДЕ')
    A(f'Тнас(Р1) = {T_sat:.2f} К. Азот — «регулярная» среда (3.1.25), при малом')
    A('перегреве изоэнтропа уходит в двухфазную область (режим Г-2Ф по Д.10).')
    A(LINE)
    A(f'{"dT, К":>6} {"Т1, К":>8} {"Ркoнд":>8} {"Ps(Т1)":>8} {"Ркр":>8} '
      f'{"beta_кр":>8} {"G*, кг/(м2с)":>13} {"F, мм2":>9}  режим')
    A(THIN)
    for r_ in rows:
        pd = f'{_bar(r_["P_dew"]):8.3f}' if r_['P_dew'] else '       -'
        pst = f'{_bar(r_["P_sat_T1"]):8.3f}' if r_.get('P_sat_T1') else '       -'
        pc = f'{_bar(r_["P_cr"]):8.3f}' if r_['P_cr'] else '       -'
        bc = f'{r_["beta_cr"]:8.4f}' if r_['beta_cr'] else '       -'
        A(f'{r_["dT"]:6.1f} {r_["T1"]:8.2f} {pd} {pst} {pc} {bc} '
          f'{r_["G_ideal"]:13.1f} {r_["F_mm2"]:9.3f}  {r_["regime"]}')
    A(THIN)
    A('Единицы давления — бар абс.')
    A('')
    A('ВНИМАНИЕ на две РАЗНЫЕ величины — их легко перепутать:')
    A('')
    A('  Ps(Т1) — давление насыщения ПРИ ТЕМПЕРАТУРЕ Т1 (изотермическое сжатие).')
    A('           РАСТЁТ с температурой, как и подсказывает интуиция. В расчёте')
    A('           не используется, приведено только для сравнения.')
    A('')
    A('  Ркoнд  — давление начала конденсации (п. 3.4) НА ИЗОЭНТРОПЕ: давление,')
    A('           при котором изоэнтропа s = s1 из точки (Р1, Т1) пересекает')
    A('           линию конденсации при расширении (Е.3.3.2). Именно оно делит')
    A('           отрезок на интервалы в методе Е.2.3.2. ПАДАЕТ с температурой.')
    A('')
    A('Почему Ркoнд падает, хотя Ps(Т1) растёт: перегрев повышает энтропию s1,')
    A('а энтропия сухого насыщенного пара s_g(P) растёт при ПАДЕНИИ давления.')
    A('Условие пересечения s1 = s_g(Ркoнд) выполняется тем ниже по давлению, чем')
    A('сильнее перегрет газ: перегретому газу нужно расшириться дальше, чтобы')
    A('стать насыщенным. Прочерк означает, что s1 превысила s_g даже при Р2 —')
    A('конденсации на отрезке нет вовсе, режим Г-Г реализуется полностью.')
    A('То же направление даёт формула (Е.3.16): Ркoнд = Р1·(Тнас(Р1)/Т1)^(1/(Лg-Л)):')
    A('с ростом Т1 отношение Тнас(Р1)/Т1 убывает, значит убывает и Ркoнд.')
    return '\n'.join(out)
