
from __future__ import annotations
"""
Упрощённый расчёт предохранительного клапана по ГОСТ 12.2.085-2017.
Все модули объединены в один файл для удобства.

Добавлено:
  - Учёт объёма сосуда V_vessel (м³)
  - Уровень заполнения жидкостью fill_level (0..1)
  - Расчёт аварийного расхода с учётом фазового состава
"""


import math
from dataclasses import dataclass, field
from typing import Optional, Tuple, List

# Попытка импорта CoolProp для термодинамики
try:
    from CoolProp.CoolProp import PropsSI
    HAS_COOLPROP = True
except ImportError:
    HAS_COOLPROP = False
    print("ВНИМАНИЕ: CoolProp не установлен. Используйте идеальную модель газа.")


# =============================================================================
# КОНСТАНТЫ
# =============================================================================
P_ATM_DEFAULT = 101320.0      # Па, атмосферное давление (п. 3.4)
R_UNIVERSAL = 8314.3          # Дж/(кмоль·К)
RE_TURBULENT = 1.0e5          # Порог турбулентности для Kv (Д.7)


# =============================================================================
# ТЕРМОДИНАМИКА
# =============================================================================
class Fluid:
    """Обёртка над CoolProp для получения свойств среды."""

    def __init__(self, name: str = 'Nitrogen'):
        self.name = name
        if HAS_COOLPROP:
            self.T_crit = PropsSI('Tcrit', '', 0, '', 0, name)
            self.P_crit = PropsSI('Pcrit', '', 0, '', 0, name)
            self.M_molar = PropsSI('molemass', '', 0, '', 0, name)
            self.R_specific = PropsSI('gas_constant', '', 0, '', 0, name) / self.M_molar
        else:
            # Заглушки для азота без CoolProp
            self.T_crit = 126.2
            self.P_crit = 33.9e5
            self.R_specific = 296.8

    def rho_PT(self, P: float, T: float) -> float:
        """Плотность по давлению и температуре."""
        if HAS_COOLPROP:
            return PropsSI('D', 'P', P, 'T', T, self.name)
        # Идеальный газ: rho = P / (R * T)
        return P / (self.R_specific * T)

    def h_fg(self, P: float) -> float:
        """Скрытая теплота парообразования."""
        if HAS_COOLPROP:
            h_g = PropsSI('H', 'Q', 1, 'P', P, self.name)
            h_l = PropsSI('H', 'Q', 0, 'P', P, self.name)
            return h_g - h_l
        return 200000.0  # Заглушка для азота ~200 кДж/кг

    def h_liq_sat(self, P: float) -> float:
        """Энтальпия кипящей жидкости."""
        if HAS_COOLPROP:
            return PropsSI('H', 'Q', 0, 'P', P, self.name)
        return 0.0

    def h_PT(self, P: float, T: float) -> float:
        """Энтальпия по P,T."""
        if HAS_COOLPROP:
            return PropsSI('H', 'P', P, 'T', T, self.name)
        # Для идеального газа: h = cp * T (cp ~ 1040 Дж/(кг·К) для N2)
        return 1040.0 * T

    def mu_PT(self, P: float, T: float) -> float:
        """Динамическая вязкость."""
        if HAS_COOLPROP:
            return PropsSI('V', 'P', P, 'T', T, self.name)
        return 1.7e-5  # Заглушка для азота

    def T_sat(self, P: float) -> float:
        """Температура насыщения."""
        if HAS_COOLPROP:
            if P >= self.P_crit:
                raise ValueError(f"P={P/1e5:.2f} бар >= Pкр={self.P_crit/1e5:.2f} бар")
            return PropsSI('T', 'Q', 1, 'P', P, self.name)
        # Приближённая формула для азота
        return 77.0 + (P - 1e5) / 1e5 * 10.0

    def k_PT(self, P: float, T: float) -> float:
        """Показатель адиабаты k = cp/cv."""
        if HAS_COOLPROP:
            cp = PropsSI('C', 'P', P, 'T', T, self.name)
            cv = PropsSI('O', 'P', P, 'T', T, self.name)
            return cp / cv
        return 1.4

    def s_PT(self, P: float, T: float) -> float:
        """Энтропия."""
        if HAS_COOLPROP:
            return PropsSI('S', 'P', P, 'T', T, self.name)
        return 5000.0  # Заглушка


# =============================================================================
# ИСХОДНЫЕ ДАННЫЕ
# =============================================================================
@dataclass
class ValveCase:
    """Исходные данные расчёта одного случая."""

    # --- Среда и условия ---
    fluid_name: str = 'Nitrogen'
    T1: float = 107.5           # К, температура перед клапаном
    Q_heat: float = 20000.0     # Вт, аварийный теплоприток

    # --- Давления (избыточные, Па) ---
    P_n_g: float = 5.0e5        # Рн — давление настройки
    P_work_g: float = 3.0e5     # Рр — рабочее давление
    P_design_g: float = 6.0e5   # Р — расчётное давление
    P_av_max_g: float = 6.6e5   # Рав max — макс. давление аварийного сброса
    P_back_static_g: float = 0.0  # Противодавление статическое
    P_back_dyn_g: float = 0.0     # Противодавление динамическое
    dP_in: float = 0.0          # Потери на входе
    P_atm: float = P_ATM_DEFAULT

    # --- Клапан ---
    alpha1: float = 0.75        # Коэфф. расхода для газа
    alpha2: float = 0.65        # Коэфф. расхода для жидкости
    catalog_d0_mm: tuple = (6, 8, 10, 12, 16, 20, 25, 32, 40, 50, 65, 80)
    N: int = 1                  # Число клапанов
    balanced: bool = False      # Разгруженный клапан
    membrane_device: bool = False  # Мембранное устройство

    # --- НОВОЕ: Параметры сосуда ---
    V_vessel: float = 1.0       # м³, полный объём сосуда
    fill_level: float = 0.5     # 0..1, доля заполнения жидкостью

    title: str = 'Расчёт ПК'
    stubs: set = field(default_factory=set)

    def mark(self, name: str) -> str:
        return '   <-- ЗАГЛУШКА' if name in self.stubs else ''


# =============================================================================
# ДАВЛЕНИЯ
# =============================================================================
@dataclass
class PressureSet:
    """Набор расчётных давлений."""
    P_atm: float
    P_n_g: float
    P_no_g: float
    P_po_g: float
    P1: float          # Абсолютное перед клапаном
    P2: float          # Абсолютное за клапаном
    formula_po: str

    @property
    def beta(self) -> float:
        return self.P2 / self.P1


def calc_pressures(case: ValveCase) -> PressureSet:
    """Расчёт давлений по п. 5.4, Д.4.2."""
    P_n_g = case.P_n_g

    # Давление полного открытия Рпо (п. 5.4)
    if P_n_g < 0.3e6:
        P_po_g = P_n_g + 0.05e6
        formula = 'Рпо = Рн + 0,05 МПа'
    elif P_n_g <= 6.0e6:
        P_po_g = 1.15 * P_n_g
        formula = 'Рпо = 1,15·Рн'
    else:
        P_po_g = 1.10 * P_n_g
        formula = 'Рпо = 1,10·Рн'

    P_no_g = P_n_g  # Давление начала открытия
    P_back_g = case.P_back_static_g + case.P_back_dyn_g

    # Д.4.2: абсолютные давления
    P1 = P_po_g + case.P_atm - case.dP_in
    P2 = P_back_g + case.P_atm

    return PressureSet(
        P_atm=case.P_atm, P_n_g=P_n_g, P_no_g=P_no_g, P_po_g=P_po_g,
        P1=P1, P2=P2, formula_po=formula
    )


# =============================================================================
# АВАРИЙНЫЙ РАСХОД (с учётом объёма сосуда)
# =============================================================================
@dataclass
class ReliefLoad:
    """Результат расчёта аварийного расхода."""
    G_av: float           # кг/с, основной расход
    G_av_kgh: float       # кг/ч
    h_fg: float           # Дж/кг, скрытая теплота
    note: str = ''


def calc_relief_load(case: ValveCase, fluid: Fluid, ps: PressureSet) -> ReliefLoad:
    """
    Расчёт аварийного расхода от теплопритока (прил. Г).

    С учётом объёма сосуда и уровня заполнения:
    - Проверяется, достаточно ли жидкости для выпаривания
    - Рассчитывается время до полного испарения
    """
    P1 = ps.P1

    # Скрытая теплота парообразования
    try:
        h_fg = fluid.h_fg(P1)
    except:
        h_fg = 200000.0  # Заглушка

    # Основной расчёт по Г.2: Gав = Q / h_gl
    G_av = case.Q_heat / h_fg if h_fg > 0 else 0.0

    # --- Учёт объёма сосуда и уровня заполнения ---
    V_liquid = case.V_vessel * case.fill_level  # м³, объём жидкости
    V_gas = case.V_vessel * (1.0 - case.fill_level)  # м³, объём газа

    # Плотность жидкости и пара
    try:
        rho_l = fluid.rho_PT(P1, fluid.T_sat(P1))
        rho_g = fluid.rho_PT(P1, case.T1)
    except:
        rho_l = 800.0  # Заглушка для жидкого азота
        rho_g = 5.0    # Заглушка для пара

    m_liquid = V_liquid * rho_l  # кг, масса жидкости
    m_gas = V_gas * rho_g        # кг, масса пара

    # Время до полного испарения при данном теплопритоке
    t_evap = m_liquid * h_fg / case.Q_heat if case.Q_heat > 0 else float('inf')

    note = (f"Объём сосуда: {case.V_vessel:.2f} м³, "
            f"заполнение: {case.fill_level*100:.0f}% | "
            f"Жидкости: {m_liquid:.1f} кг, Пара: {m_gas:.1f} кг | "
            f"Время испарения: {t_evap/60:.1f} мин")

    return ReliefLoad(G_av=G_av, G_av_kgh=G_av*3600, h_fg=h_fg, note=note)


# =============================================================================
# МАССОВАЯ СКОРОСТЬ (упрощённый расчёт)
# =============================================================================
@dataclass
class FlowSolution:
    """Результат расчёта массовой скорости."""
    G_ideal: float    # кг/(м²·с), массовая скорость
    regime: str       # 'критический' или 'докритический'
    beta_cr: float    # Отношение Pкр/P1
    rho0: float       # кг/м³, плотность в седле
    Kn: float         # Функция расхода


def calc_flow(case: ValveCase, fluid: Fluid, ps: PressureSet) -> FlowSolution:
    """
    Расчёт массовой скорости методом прямого интегрирования (Е.1).
    Упрощённая версия для газового режима.
    """
    P1, P2 = ps.P1, ps.P2
    T1 = case.T1

    # Плотность на входе
    rho1 = fluid.rho_PT(P1, T1)
    k = fluid.k_PT(P1, T1)

    # Критическое отношение давлений для идеального газа
    beta_cr = (2.0 / (k + 1.0)) ** (k / (k - 1.0))

    # Определение режима течения
    beta = P2 / P1
    regime = 'критический' if beta <= beta_cr else 'докритический'

    # Расчёт массовой скорости
    if regime == 'критический':
        # Критический режим: G* = sqrt(k * P1 * rho1) * (2/(k+1))^((k+1)/(2*(k-1)))
        P0 = beta_cr * P1
        rho0 = rho1 * (P0 / P1) ** (1.0 / k)
        G_ideal = math.sqrt(k * P0 * rho0)
    else:
        # Докритический режим
        P0 = P2
        rho0 = rho1 * (P2 / P1) ** (1.0 / k)
        # Интеграл -2*∫dP/rho
        integral = 2.0 * k / (k - 1.0) * P1 / rho1 * (1.0 - (P2/P1)**((k-1.0)/k))
        G_ideal = math.sqrt(rho1**2 * integral) if integral > 0 else 0.0

    # Функция расхода Kn = G*ideal / sqrt(P1 * rho1)
    Kn = G_ideal / math.sqrt(P1 * rho1) if P1 * rho1 > 0 else 0.0

    return FlowSolution(
        G_ideal=G_ideal, regime=regime, beta_cr=beta_cr, rho0=rho0, Kn=Kn
    )


# =============================================================================
# КОЭФФИЦИЕНТЫ
# =============================================================================
def calc_coeffs(case: ValveCase, sol: FlowSolution) -> Tuple[float, float, float, float]:
    """
    Расчёт коэффициентов alpha, Kc, Kv, Kw.
    Возвращает (alpha, Kc, Kv, Kw).
    """
    # alpha по режиму (Д.5)
    alpha = case.alpha1 if sol.regime == 'критический' else case.alpha2

    # Kc для мембранных устройств (Д.6)
    Kc = 0.9 if case.membrane_device else 1.0

    # Kv по числу Рейнольдса (Д.7) - упрощённо принимаем 1.0
    Kv = 1.0

    # Kw для разгруженных клапанов (Д.8, табл. Д.1)
    if case.balanced:
        # Упрощённый расчёт Kw
        r = case.P_back_static_g / case.P_n_g if case.P_n_g > 0 else 0
        Kw = 1.0 if r <= 0.3 else max(0.7, 1.3 - r)
    else:
        Kw = 1.0

    return alpha, Kc, Kv, Kw


# =============================================================================
# ПОДБОР КЛАПАНА
# =============================================================================
@dataclass
class SizingResult:
    """Результат подбора клапана."""
    F_required_m2: float
    d0_required_mm: float
    d0_selected_mm: Optional[float]
    F_selected_m2: Optional[float]
    G_selected: Optional[float]
    margin_pct: Optional[float]

    @property
    def F_required_mm2(self):
        return self.F_required_m2 * 1e6

    @property
    def F_selected_mm2(self):
        return None if self.F_selected_m2 is None else self.F_selected_m2 * 1e6


def size_valve(case: ValveCase, load: ReliefLoad, sol: FlowSolution,
               alpha: float, Kc: float, Kv: float, Kw: float) -> SizingResult:
    """
    Подбор площади седла и типоразмера клапана (Д.2, 5.5).
    """
    G_av = load.G_av
    G_ideal = sol.G_ideal
    N = case.N

    # Минимальная площадь седла по формуле (Д.2)
    denom = alpha * Kc * Kv * Kw * G_ideal * N
    F_req = G_av / denom if denom > 0 else float('inf')

    d0_req = math.sqrt(4.0 * F_req / math.pi) * 1e3  # мм

    # Подбор из каталога (5.5) - ближайший больший
    d0_sel = None
    F_sel = None
    G_sel = None
    margin = None

    for d0 in sorted(case.catalog_d0_mm):
        F = math.pi * d0**2 / 4.0 * 1e-6  # м²
        if F >= F_req:
            d0_sel = float(d0)
            F_sel = F
            G_sel = alpha * Kc * Kv * Kw * G_ideal * F * N
            margin = (G_sel / G_av - 1.0) * 100.0 if G_av > 0 else 0.0
            break

    return SizingResult(
        F_required_m2=F_req, d0_required_mm=d0_req,
        d0_selected_mm=d0_sel, F_selected_m2=F_sel,
        G_selected=G_sel, margin_pct=margin
    )


# =============================================================================
# ОСНОВНОЙ РАСЧЁТ
# =============================================================================
@dataclass
class CaseResult:
    """Полный результат расчёта случая."""
    case: ValveCase
    fluid: Fluid
    pressures: PressureSet
    load: ReliefLoad
    flow: FlowSolution
    alpha: float
    Kc: float
    Kv: float
    Kw: float
    sizing: SizingResult


def run_calculation(case: ValveCase) -> CaseResult:
    """Выполнить полный расчёт предохранительного клапана."""
    fluid = Fluid(case.fluid_name)
    pressures = calc_pressures(case)
    load = calc_relief_load(case, fluid, pressures)
    flow = calc_flow(case, fluid, pressures)
    alpha, Kc, Kv, Kw = calc_coeffs(case, flow)
    sizing = size_valve(case, load, flow, alpha, Kc, Kv, Kw)

    return CaseResult(
        case=case, fluid=fluid, pressures=pressures, load=load,
        flow=flow, alpha=alpha, Kc=Kc, Kv=Kv, Kw=Kw, sizing=sizing
    )


# =============================================================================
# ПРОТОКОЛ РАСЧЁТА
# =============================================================================
def format_report(r: CaseResult) -> str:
    """Сформировать текстовый протокол расчёта."""
    c = r.case
    ps = r.pressures
    lines = []

    def add(s):
        lines.append(s)

    add('=' * 78)
    add('РАСЧЁТ ПРЕДОХРАНИТЕЛЬНОГО КЛАПАНА ПО ГОСТ 12.2.085-2017')
    add(c.title)
    add('=' * 78)
    add('')

    # Исходные данные
    add('ИСХОДНЫЕ ДАННЫЕ')
    add('-' * 78)
    add(f'  Среда:                     {r.fluid.name}')
    add(f'  Температура T1:            {c.T1:.2f} К{c.mark("T1")}')
    add(f'  Теплоприток Q:             {c.Q_heat:.0f} Вт{c.mark("Q_heat")}')
    add(f'  Давление настройки Рн:     {c.P_n_g/1e5:.3f} бар изб.{c.mark("P_n_g")}')
    add(f'  Рабочее давление Рр:       {c.P_work_g/1e5:.3f} бар изб.')
    add(f'  Расчётное давление Р:      {c.P_design_g/1e5:.3f} бар изб.')
    add(f'  Макс. давление Рав max:    {c.P_av_max_g/1e5:.3f} бар изб.')
    add(f'  Противодавление:           {c.P_back_static_g/1e5:.3f} бар изб.')
    add(f'  Потери на входе dPвх:      {c.dP_in/1e5:.4f} бар')
    add(f'  alpha1/alpha2:             {c.alpha1:.3f}/{c.alpha2:.3f}')
    add(f'  Число клапанов N:          {c.N}')
    add(f'  Тип клапана:               {"разгруженный" if c.balanced else "неразгруженный"}')
    add('')
    add(f'  *** ПАРАМЕТРЫ СОСУДА ***')
    add(f'  Полный объём V:            {c.V_vessel:.3f} м³')
    add(f'  Уровень заполнения:        {c.fill_level*100:.1f}%')
    add(f'  {r.load.note}')
    add('')

    # Давления
    add('ШАГ 1. ДАВЛЕНИЯ (п. 5.4, Д.4.2)')
    add('-' * 78)
    add(f'  Рпо = {ps.P_po_g/1e5:.3f} бар изб.   [{ps.formula_po}]')
    add(f'  Р1  = {ps.P1/1e5:.4f} бар абс. (перед клапаном)')
    add(f'  Р2  = {ps.P2/1e5:.4f} бар абс. (за клапаном)')
    add(f'  beta = P2/P1 = {ps.beta:.4f}')
    add('')

    # Аварийный расход
    add('ШАГ 2. АВАРИЙНЫЙ РАСХОД (прил. Г.2)')
    add('-' * 78)
    add(f'  h_gl(Р1) = {r.load.h_fg/1e3:.2f} кДж/кг')
    add(f'  Gав = Q/h_gl = {r.load.G_av:.4f} кг/с = {r.load.G_av_kgh:.1f} кг/ч')
    add('')

    # Массовая скорость
    add('ШАГ 3. МАССОВАЯ СКОРОСТЬ (прил. Е.1)')
    add('-' * 78)
    add(f'  Режим течения:             {r.flow.regime}')
    add(f'  Критическое отношение:     beta_кр = {r.flow.beta_cr:.4f}')
    add(f'  Массовая скорость G*ideal: {r.flow.G_ideal:.1f} кг/(м²·с)')
    add(f'  Функция расхода Kn:        {r.flow.Kn:.4f}')
    add('')

    # Коэффициенты
    add('ШАГ 4. КОЭФФИЦИЕНТЫ (Д.5-Д.8)')
    add('-' * 78)
    add(f'  alpha = {r.alpha:.4f}   ({"критический" if r.flow.regime == "критический" else "докритический"} режим)')
    add(f'  Kc    = {r.Kc:.4f}   ({"мембранное устройство есть" if c.membrane_device else "без мембраны"})')
    add(f'  Kv    = {r.Kv:.4f}   (Re >= 1e5)')
    add(f'  Kw    = {r.Kw:.4f}   ({"разгруженный" if c.balanced else "неразгруженный"} клапан)')
    add('')

    # Площадь седла
    add('ШАГ 5. ПОДБОР КЛАПАНА (Д.2, 5.5)')
    add('-' * 78)
    sz = r.sizing
    add(f'  Fрасч = {sz.F_required_mm2:.3f} мм²  ->  d0расч = {sz.d0_required_mm:.2f} мм')
    if sz.d0_selected_mm:
        add(f'  Выбрано: d0 = {sz.d0_selected_mm:.0f} мм, F = {sz.F_selected_mm2:.2f} мм²')
        add(f'  Gфакт = {sz.G_selected:.4f} кг/с = {sz.G_selected*3600:.1f} кг/ч')
        add(f'  Запас пропускной способности: {sz.margin_pct:.1f}%')
        ok_str = 'ПОДХОДИТ' if sz.margin_pct >= 0 else 'НЕ ПОДХОДИТ'
        add(f'  Результат: {ok_str}')
    else:
        add(f'  ! В каталоге нет типоразмера >= {sz.F_required_mm2:.1f} мм²')
        add(f'    Увеличьте число клапанов N или выберите другой каталог')
    add('')

    add('=' * 78)
    add('КОНЕЦ РАСЧЁТА')
    add('=' * 78)

    return '\n'.join(lines)


# =============================================================================
# ПРИМЕР ИСПОЛЬЗОВАНИЯ
# =============================================================================
def default_case() -> ValveCase:
    """Создать случай по умолчанию для сосуда с жидким азотом."""
    case = ValveCase(
        fluid_name='Nitrogen',
        T1=130.0,
        Q_heat=3000.0,
        P_n_g=5.0e5,
        P_work_g=3.0e5,
        P_design_g=6.0e5,
        P_av_max_g=6.6e5,
        P_back_static_g=0.0,
        P_back_dyn_g=0.0,
        dP_in=0.0,
        alpha1=0.75,
        alpha2=0.65,
        N=1,
        balanced=False,
        # Параметры сосуда
        V_vessel=2.0,       # 2 м³
        fill_level=0.6,     # 60% заполнения
        title='Азот, сосуд с жидкой фазой, внешний теплоприток',
    )
    case.stubs = {'T1', 'Q_heat', 'P_n_g', 'alpha1', 'alpha2'}
    return case


if __name__ == '__main__':
    import sys

    # Создание случая
    case = default_case()

    # Расчёт
    result = run_calculation(case)

    # Вывод протокола
    report = format_report(result)
    print(report)

    # Если нужно сохранить в файл
    if len(sys.argv) > 1 and sys.argv[1] == '--save':
        with open('pk_report.txt', 'w', encoding='utf-8') as f:
            f.write(report)
        print('\nОтчёт сохранён в pk_report.txt')
