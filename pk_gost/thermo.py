# -*- coding: utf-8 -*-
"""
Термодинамика: единственная точка входа в CoolProp.

Весь остальной пакет обращается к свойствам среды только через класс Fluid.
Это позволяет в тестах подменить его идеально-газовой моделью (IdealGas)
и проверять формулы ГОСТ независимо от термодинамической библиотеки
(п. Е.1.1 допускает любые «термодинамические компьютерные библиотеки или
соответствующие термодинамические таблицы»).

Единицы — СИ (Па, К, кг/м3, Дж/кг, Дж/(кг·К), Па·с).
"""
from __future__ import annotations

import numpy as np

try:
    from CoolProp.CoolProp import PropsSI as _PropsSI
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        'Требуется CoolProp: pip install CoolProp\n'
        'В проекте: d:\\Studies\\Diplom\\env\\Scripts\\pip.exe install CoolProp'
    ) from exc


class ThermoError(RuntimeError):
    """Ошибка расчёта свойств среды (в отличие от «молча вернуть False»)."""


# --------------------------------------------------------------------------
#  Фазовые состояния (терминология Д.10)
# --------------------------------------------------------------------------
PHASE_LIQUID = 'жидкость'
PHASE_TWOPHASE = 'двухфазная смесь'
PHASE_GAS = 'газ/пар'
PHASE_SUPERCRIT = 'сверхкритическая среда'


class Fluid:
    """Обёртка над CoolProp для однокомпонентной среды."""

    def __init__(self, name: str = 'Nitrogen'):
        self.name = name
        self.T_crit = _PropsSI('Tcrit', '', 0, '', 0, name)
        self.P_crit = _PropsSI('Pcrit', '', 0, '', 0, name)
        self.T_triple = _PropsSI('Ttriple', '', 0, '', 0, name)
        self.P_triple = _PropsSI('ptriple', '', 0, '', 0, name)
        self.M_molar = _PropsSI('molemass', '', 0, '', 0, name)          # кг/моль
        self.R_specific = _PropsSI('gas_constant', '', 0, '', 0, name) / self.M_molar

    # ---------------- базовые свойства ----------------
    def _props(self, out, n1, v1, n2, v2):
        try:
            return _PropsSI(out, n1, v1, n2, v2, self.name)
        except Exception as exc:
            raise ThermoError(
                f'CoolProp не смог посчитать {out} при {n1}={v1}, {n2}={v2} '
                f'для среды {self.name}: {exc}'
            ) from exc

    def rho_PT(self, P, T):
        return self._props('D', 'P', P, 'T', T)

    def s_PT(self, P, T):
        return self._props('S', 'P', P, 'T', T)

    def h_PT(self, P, T):
        return self._props('H', 'P', P, 'T', T)

    def mu_PT(self, P, T):
        return self._props('V', 'P', P, 'T', T)          # 'V' = динамическая вязкость

    def cp_PT(self, P, T):
        return self._props('C', 'P', P, 'T', T)

    def Pr_PT(self, P, T):
        """Число Прандтля (Ж.8): Pr = cp*mu/lambda."""
        return self._props('PRANDTL', 'P', P, 'T', T)

    def Z_PT(self, P, T):
        return self._props('Z', 'P', P, 'T', T)

    def k_PT(self, P, T):
        """Показатель адиабаты k = cp/cv (п. 3.4; ср. табл. И.1)."""
        return self._props('C', 'P', P, 'T', T) / self._props('O', 'P', P, 'T', T)

    # ---------------- изоэнтропа ----------------
    def rho_Ps(self, P, s):
        """Плотность на изоэнтропе. Принимает скаляр или массив давлений."""
        if np.isscalar(P):
            return self._props('D', 'P', float(P), 'S', s)
        try:
            return np.asarray(_PropsSI('D', 'P', list(np.asarray(P, float)),
                                       'S', s, self.name), dtype=float)
        except Exception as exc:
            raise ThermoError(
                f'CoolProp не смог посчитать плотность на изоэнтропе '
                f's={s:.4f} Дж/(кг·К) для среды {self.name}: {exc}'
            ) from exc

    def T_Ps(self, P, s):
        return self._props('T', 'P', P, 'S', s)

    def h_Ps(self, P, s):
        return self._props('H', 'P', P, 'S', s)

    # ---------------- линия насыщения ----------------
    def T_sat(self, P):
        """Температура насыщения при давлении P (нужна для контроля перегрева)."""
        if P >= self.P_crit:
            raise ThermoError(
                f'P = {P/1e5:.3f} бар >= Pкр = {self.P_crit/1e5:.3f} бар '
                f'для {self.name}: линии насыщения нет, среда сверхкритическая. '
                f'См. табл. Д.2 — ветвь для суперкритической области.'
            )
        return self._props('T', 'Q', 1, 'P', P)

    def P_sat(self, T):
        """
        Давление насыщения при температуре T (обратная к T_sat).

        ВНИМАНИЕ: это НЕ давление начала конденсации Ркoнд на изоэнтропе.
        Ps(T) растёт с температурой, а Ркoнд при расширении из состояния
        (Р1, Т1) с ростом Т1 ПАДАЕТ — см. Е.3.3.2 и (Е.3.16). Величины разные:
        Ps(T) отвечает изотермическому сжатию, Ркoнд — изоэнтропному расширению.
        """
        if T >= self.T_crit:
            raise ThermoError(
                f'T = {T:.3f} К >= Ткр = {self.T_crit:.3f} К для {self.name}: '
                f'давления насыщения не существует.')
        return self._props('P', 'Q', 1, 'T', T)

    def s_gas_sat(self, P):
        """Энтропия сухого насыщенного пара s_g(P) — граница конденсации."""
        return self._props('S', 'Q', 1, 'P', P)

    def s_liq_sat(self, P):
        """Энтропия кипящей жидкости s_l(P) — граница кипения."""
        return self._props('S', 'Q', 0, 'P', P)

    def rho_gas_sat(self, P):
        return self._props('D', 'Q', 1, 'P', P)

    def rho_liq_sat(self, P):
        return self._props('D', 'Q', 0, 'P', P)

    def h_fg(self, P):
        """Скрытая теплота парообразования h_gl = h_g - h_l (обозначение п. 3.4)."""
        return (self._props('H', 'Q', 1, 'P', P) - self._props('H', 'Q', 0, 'P', P))

    def h_liq_sat(self, P):
        return self._props('H', 'Q', 0, 'P', P)

    # ---------------- фазовое состояние на изоэнтропе ----------------
    def phase_Ps(self, P, s):
        """
        Фазовое состояние точки (P, s).

        Определяется прямым сравнением с энтропиями на линиях насыщения
        (п. Е.3.3), а не через паросодержание из P,S-флеша: последний
        в однофазной области возвращает Q = -1 и требует «глушить» исключения.
        """
        if P >= self.P_crit:
            return PHASE_SUPERCRIT
        if P <= self.P_triple:
            # ниже тройной точки двухфазной области «жидкость-пар» нет
            return PHASE_GAS
        if s >= self.s_gas_sat(P):
            return PHASE_GAS
        if s <= self.s_liq_sat(P):
            return PHASE_LIQUID
        return PHASE_TWOPHASE

    def quality_Ps(self, P, s):
        """Массовое паросодержание x на изоэнтропе (0..1) либо None вне купола."""
        if self.phase_Ps(P, s) != PHASE_TWOPHASE:
            return None
        s_l, s_g = self.s_liq_sat(P), self.s_gas_sat(P)
        return (s - s_l) / (s_g - s_l)

    # ---------------- пересечение изоэнтропы с границей фаз (Е.3.3) ----------------
    def _cross(self, P_hi, P_lo, s, boundary):
        """
        Бисекция по давлению для уравнения s = s_boundary(P) на [P_lo, P_hi].

        boundary: 'dew' (линия конденсации, Е.3.3.2) или 'bubble' (линия кипения, Е.3.3.1).
        Возвращает давление пересечения либо None, если пересечения нет.
        """
        s_bnd = self.s_gas_sat if boundary == 'dew' else self.s_liq_sat
        P_lo = max(P_lo, self.P_triple * 1.001)
        P_hi = min(P_hi, self.P_crit * 0.999)
        if P_lo >= P_hi:
            return None

        # f(P) = s - s_bnd(P); ищем смену знака между P_lo и P_hi
        f_lo = s - s_bnd(P_lo)
        f_hi = s - s_bnd(P_hi)
        if f_lo * f_hi > 0.0:
            return None                      # изоэнтропа не пересекает границу
        for _ in range(200):
            P_mid = 0.5 * (P_lo + P_hi)
            if (s - s_bnd(P_mid)) * f_lo > 0.0:
                P_lo = P_mid
            else:
                P_hi = P_mid
            if P_hi - P_lo < 1e-6 * max(P_hi, 1.0):
                break
        return 0.5 * (P_lo + P_hi)

    def dew_pressure(self, P1, P2, s):
        """Давление начала конденсации на изоэнтропе s при расширении P1 -> P2 (Е.3.3.2)."""
        return self._cross(P1, P2, s, 'dew')

    def bubble_pressure(self, P1, P2, s):
        """Давление вскипания на изоэнтропе s при расширении P1 -> P2 (Е.3.3.1)."""
        return self._cross(P1, P2, s, 'bubble')

    def phase_crossings(self, P1, P2, s):
        """
        Все точки пересечения изоэнтропы с границей двухфазной области
        на отрезке [P2, P1], по убыванию давления.

        Примечание к Е.1.4: эти точки обязательно включают в сетку
        интегрирования, «учитывая, что при сбросе сред с фазовыми
        превращениями критическое давление часто находится на границе
        фазовой диаграммы».
        """
        pts = []
        for boundary in ('dew', 'bubble'):
            P = self._cross(P1, P2, s, boundary)
            if P is not None and P2 < P < P1:
                pts.append(P)
        return sorted(set(pts), reverse=True)

    # ---------------- показатель изоэнтропы ----------------
    def n_isentropic(self, P, s, side='center', rel_step=2e-4):
        """
        Показатель изоэнтропы n = (dlnP/dlnrho)_s  (Д.28, Е.3.1).

        side:
            'center' — центральная разность (только внутри одной фазовой области);
            'gas'    — односторонняя со стороны БОЛЬШИХ давлений (n+ по Е.1.5);
            'low'    — односторонняя со стороны МЕНЬШИХ давлений (n- по Е.1.5).

        На границе фазовой диаграммы n терпит разрыв (Е.1.5), поэтому
        центральную разность через границу брать нельзя — она даёт смесь n+ и n-.
        """
        d = P * rel_step
        if side == 'center':
            Pa, Pb = P + d, P - d
        elif side == 'gas':                 # P < Pa < Pb, обе точки выше P
            Pa, Pb = P + d, P + 2.0 * d
        elif side == 'low':                 # обе точки ниже P
            Pa, Pb = P - 2.0 * d, P - d
        else:
            raise ValueError(f'side должно быть center/gas/low, получено {side!r}')
        rho_a, rho_b = self.rho_Ps(Pa, s), self.rho_Ps(Pb, s)
        return float(np.log(Pa / Pb) / np.log(rho_a / rho_b))

    def omega_two_point(self, P, s):
        """
        Параметр омега двухточечным методом (Е.3.11):
            omega = 9*(rho* / rho_0,9 - 1),
        где rho_0,9 — плотность при изоэнтропном расширении до 0,9 от базового давления.
        """
        rho_star = self.rho_Ps(P, s)
        rho_09 = self.rho_Ps(0.9 * P, s)
        return float(9.0 * (rho_star / rho_09 - 1.0))


class IdealGas:
    """
    Идеальный газ с постоянным показателем изоэнтропы — эталон для тестов.

    Реализует ту часть интерфейса Fluid, которая нужна интегратору (Е.1),
    что позволяет проверить численный интегратор против аналитических
    формул табл. Е.1 (Е.2.4, Е.2.6) без обращения к CoolProp.
    """

    def __init__(self, n=1.4, P1=1.0, rho1=1.0):
        self.n, self.P1, self.rho1 = float(n), float(P1), float(rho1)
        self.name = f'IdealGas(n={n})'
        self.P_crit = float('inf')
        self.P_triple = 0.0

    def rho_Ps(self, P, s=None):
        """rho = rho1*(P/P1)^(1/n)  — уравнение состояния (Д.28)."""
        return self.rho1 * np.power(np.asarray(P, float) / self.P1, 1.0 / self.n)

    def phase_Ps(self, P, s=None):
        return PHASE_GAS

    def phase_crossings(self, P1, P2, s=None):
        return []

    def n_isentropic(self, P, s=None, side='center', rel_step=2e-4):
        return self.n
