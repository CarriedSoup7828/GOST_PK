# -*- coding: utf-8 -*-
"""
Давления и проверки их иерархии.

  5.4    давление полного открытия Рпо (три диапазона по Рн)
  Д.4.2  Р1 = Рпо + Ратм;  Р2 = Рп + Ратм
  табл.3 требования к соотношению давлений в оборудовании и ПК
  8.2.1  потери давления на входе dPвх <= 0,03*Рн

ВНИМАНИЕ. Именно здесь была ошибка исходного скрипта: для Рн = 0,5 МПа
применялся коэффициент 1,10 из диапазона (3), тогда как действует
формула (2) с коэффициентом 1,15. Занижение Р1 -> занижение G на ~4 %.
"""
from __future__ import annotations

from dataclasses import dataclass, field

# Границы диапазонов п. 5.4
P_N_LOW = 0.3e6      # Па, граница «менее 0,3 МПа»
P_N_HIGH = 6.0e6     # Па, граница «свыше 6,0 МПа»
DP_LOW = 0.05e6      # Па, добавка для формулы (1)

SEVERITY_OK = 'OK'
SEVERITY_WARN = 'ПРЕДУПРЕЖДЕНИЕ'
SEVERITY_FAIL = 'НАРУШЕНИЕ'


@dataclass
class Check:
    """Результат одной проверки требований ГОСТ."""
    ok: bool
    severity: str
    ref: str          # пункт стандарта
    text: str

    def __str__(self):
        mark = {SEVERITY_OK: '+', SEVERITY_WARN: '!', SEVERITY_FAIL: 'X'}[self.severity]
        return f'  [{mark}] {self.ref:<12} {self.text}'


def P_po_gauge(P_n_g: float):
    """
    Давление полного открытия Рпо (избыточное) по п. 5.4.

    (1) Рн < 0,3 МПа           : Рпо = Рн + 0,05 МПа
    (2) 0,3 МПа <= Рн <= 6,0 МПа: Рпо = 1,15*Рн
    (3) Рн > 6,0 МПа           : Рпо = 1,10*Рн

    Возвращает (Рпо, формула-строка).
    """
    if P_n_g <= 0.0:
        raise ValueError('Давление настройки Рн должно быть положительным (избыточное)')
    if P_n_g < P_N_LOW:
        return P_n_g + DP_LOW, 'ф.(1) Рпо = Рн + 0,05 МПа'
    if P_n_g <= P_N_HIGH:
        return 1.15 * P_n_g, 'ф.(2) Рпо = 1,15·Рн'
    return 1.10 * P_n_g, 'ф.(3) Рпо = 1,10·Рн'


@dataclass
class PressureSet:
    """Полный набор давлений расчёта (избыточные — с суффиксом _g, абсолютные — без)."""
    P_atm: float
    P_n_g: float                 # давление настройки Рн
    P_no_g: float                # давление начала открытия Рно
    P_po_g: float                # давление полного открытия Рпо
    P_back_g: float              # противодавление Рп = Рп ст + Рп дин
    P_back_static_g: float
    P_back_dyn_g: float
    dP_in: float                 # потери на входе dPвх
    P1: float                    # абс. давление до клапана (Д.4.2)
    P2: float                    # абс. давление за клапаном (Д.4.2)
    formula_po: str
    ratio_po_n: float            # Рпо/Рн — ключ входа в табл. Д.1
    ratio_back_no: float         # Рп/Рно — аргумент Kw
    checks: list = field(default_factory=list)

    @property
    def beta(self):
        """Отношение абсолютных давлений beta = P2/P1 (п. 3.4)."""
        return self.P2 / self.P1


def build_pressures(P_n_g, P_back_static_g=0.0, P_back_dyn_g=0.0,
                    dP_in=0.0, P_no_g=None, P_atm=101320.0,
                    P_po_g=None, formula_po=None):
    """
    Собрать набор давлений по 5.4 и Д.4.2.

    P_no_g   — давление начала открытия; если не задано, принимается равным Рн
               (п. 5.11 допускает указывать в ЭД Рн и/или Рно).
    P_po_g   — можно задать явно, если изготовитель указал иное соотношение
               (5.4 допускает «другие соотношения при соблюдении требований 5.1»).
    dP_in    — потери давления на входе (3.1.13); при dPвх > 0,03*Рн п. 8.2.1
               требует учитывать их в расчёте пропускной способности, поэтому
               они вычитаются из Р1.
    """
    if P_po_g is None:
        P_po_g, formula_po = P_po_gauge(P_n_g)
    elif formula_po is None:
        formula_po = 'задано изготовителем (5.4, «другие соотношения»)'

    if P_no_g is None:
        P_no_g = P_n_g

    P_back_g = P_back_static_g + P_back_dyn_g

    # Д.4.2: за давление перед клапаном принимают абсолютное давление полного открытия.
    # 8.2.1: потери на входе снижают давление, доходящее до клапана.
    P1 = P_po_g + P_atm - dP_in
    P2 = P_back_g + P_atm

    return PressureSet(
        P_atm=P_atm, P_n_g=P_n_g, P_no_g=P_no_g, P_po_g=P_po_g,
        P_back_g=P_back_g, P_back_static_g=P_back_static_g,
        P_back_dyn_g=P_back_dyn_g, dP_in=dP_in,
        P1=P1, P2=P2, formula_po=formula_po,
        ratio_po_n=P_po_g / P_n_g,
        ratio_back_no=P_back_g / P_no_g if P_no_g > 0 else 0.0,
    )


def check_pressures(ps: PressureSet, P_work_g=None, P_design_g=None,
                    P_av_max_g=None, P_allowed_g=None, fluid=None,
                    balanced=False, medium='газ'):
    """
    Проверки табл. 3 (примечание 4), 8.2.1, 4.1.1/4.1.3 и Б.3.4.

    P_work_g   — рабочее давление Рр
    P_design_g — расчётное давление Р
    P_av_max_g — максимально допустимое давление аварийного сброса Рав max
    P_allowed_g— разрешённое давление Рраз (для систем в эксплуатации)
    balanced   — разгруженный клапан
    """
    ch = []

    def add(ok, ref, text, warn_only=False):
        sev = SEVERITY_OK if ok else (SEVERITY_WARN if warn_only else SEVERITY_FAIL)
        ch.append(Check(ok, sev, ref, text))

    # --- табл. 3, прим. 4: Рн > Рр ---
    if P_work_g is not None:
        add(ps.P_n_g > P_work_g, 'табл.3 п.4',
            f'Рн > Рр: {ps.P_n_g/1e5:.3f} > {P_work_g/1e5:.3f} бар изб. '
            f'— протекание процесса без недопустимой утечки через ПК')

    # --- 5.3 / табл. 3: Рр (или Рраз) < Рно <= Р ---
    P_lim = P_design_g if P_design_g is not None else P_allowed_g
    if P_lim is not None:
        ok = ps.P_no_g <= P_lim
        note = ''
        if not ok and P_work_g is not None and abs(P_lim - P_work_g) < 1e-9:
            # 5.3: допускается Рно до 1,1*Р, если Р = Рр
            ok = ps.P_no_g <= 1.1 * P_lim
            note = ' (по 5.3 допущено Рно <= 1,1·Р, т.к. Р = Рр)'
        add(ok, '5.3',
            f'Рно <= Р: {ps.P_no_g/1e5:.3f} <= {P_lim/1e5:.3f} бар изб.{note}')

    # --- табл. 3: Рпо <= Рав max ---
    if P_av_max_g is not None:
        add(ps.P_po_g <= P_av_max_g, 'табл.3 п.4',
            f'Рпо <= Рав max: {ps.P_po_g/1e5:.3f} <= {P_av_max_g/1e5:.3f} бар изб. '
            f'— полное открытие до достижения предельного давления сброса')
    if P_design_g is not None and P_av_max_g is not None:
        add(P_av_max_g <= 1.1 * P_design_g, 'табл.3 п.2',
            f'Рав max <= 1,1·Р (аккумулирование по 6.2 [6]): '
            f'{P_av_max_g/1e5:.3f} <= {1.1*P_design_g/1e5:.3f} бар изб.', warn_only=True)

    # --- 8.2.1: потери на входе ---
    lim = 0.03 * ps.P_n_g
    add(ps.dP_in <= lim, '8.2.1',
        f'dPвх <= 0,03·Рн: {ps.dP_in/1e5:.4f} <= {lim/1e5:.4f} бар'
        + ('' if ps.dP_in <= lim else
           ' — требуется проверка динамической устойчивости ПК; '
           'dPвх учтён в Р1'),
        warn_only=True)

    # --- Р1 в пределах применимости уравнения состояния ---
    if fluid is not None:
        add(ps.P1 < fluid.P_crit, 'табл.Д.2',
            f'Р1 < Ркр среды: {ps.P1/1e5:.3f} < {fluid.P_crit/1e5:.3f} бар абс. '
            f'— среда не сверхкритическая')

    # --- противодавление: 4.1.1 (неразгруженные) / 4.1.3, Б.3.4 (разгруженные) ---
    r = ps.ratio_back_no
    if balanced:
        add(r <= 0.50, 'Б.3.4',
            f'Рп/Рно <= 0,50 для разгруженного ПК: {r:.3f}', warn_only=True)
    else:
        # 4.1.1: не рекомендуется при Рп >= 0,10*Рн (Рпо = 1,1Рн) / 0,15*Рн (Рпо = 1,15Рн)
        thr = 0.15 if ps.ratio_po_n >= 1.15 else 0.10
        add(ps.P_back_g < thr * ps.P_n_g, '4.1.1',
            f'Рп < {thr:.2f}·Рн для неразгруженного двухпозиционного ПК: '
            f'{ps.P_back_g/1e5:.3f} < {thr*ps.P_n_g/1e5:.3f} бар изб.', warn_only=True)
        # табл. Б.1: динамическое противодавление под ЗЭл, критический режим
        thr_dyn = 0.15 if ps.ratio_po_n > 1.10 else 0.10
        add(ps.P_back_dyn_g <= thr_dyn * ps.P_n_g, 'табл.Б.1',
            f'Рп дин <= {thr_dyn:.2f}·Рн: {ps.P_back_dyn_g/1e5:.3f} <= '
            f'{thr_dyn*ps.P_n_g/1e5:.3f} бар изб.', warn_only=True)

    ps.checks = ch
    return ch
