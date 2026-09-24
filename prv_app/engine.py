# -*- coding: utf-8 -*-
"""
Движок расчёта динамики предохранительного клапана (PRV).

Вся математика полностью взята из файла Another_PRessure_Radau.py
(метод интегрирования Radau, термодинамика CoolProp HEOS 'Nitrogen',
кубические сплайны характеристик клапана).
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import CubicSpline
import CoolProp.CoolProp as CP

# ================= Константы =================
R = 8.314
M_molar = 0.0280134
k = 1.4
g = 9.81
P_atm = 101325.0

# ---------- Кубические сплайны (гладкие производные) ----------
hdc_nodes = [0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40]
alpha_nodes = [0, 0.28, 0.42, 0.52, 0.60, 0.66, 0.68, 0.70]
alpha_c = CubicSpline(hdc_nodes, alpha_nodes, bc_type='clamped')

hdc_rho = [0, 0.05, 0.10, 0.20, 0.30, 0.40]
rho_nodes = [1.0, 1.1, 1.3, 1.8, 2.2, 2.5]
rho_flow = CubicSpline(hdc_rho, rho_nodes, bc_type='clamped')


class Params:
    """Известные данные, которые задаёт пользователь."""

    # Единицы ввода -> СИ
    BAR = 1e5
    MM2M = 1e-3

    def __init__(self):
        # ---- Резервуар / газ ----
        self.V_total = 1.30            # м^3, полный объём резервуара
        self.fill_fraction = 0.18      # доля заполнения жидкостью [-]
        self.Q_dot = 5000.0            # Вт, теплоприток
        self.t_end = 500.0             # с, время моделирования

        # ---- Давления (ввод в бар) ----
        self.P_init_bar = 4.0          # начальное давление
        self.P_open_bar = 4.22         # давление начала открытия
        self.P_full_bar = 4.34         # давление полного открытия

        # ---- Клапан ----
        self.d_c_mm = 32.0             # диаметр седла, мм
        self.m = 0.5                   # масса золотника, кг
        self.eta = 30.0                # коэффициент демпфирования, Н*с/м
        self.h_max_frac = 0.25         # макс. подъём (доля d_c) [не влияет на ODE, как в оригинале]
        self.h_stop_frac = 0.30        # ход ограничителя (доля d_c)
        self.c_spring = 1000.0         # жёсткость пружины, Н/м

    # -------- Приведение к СИ --------
    @property
    def P_init(self):
        return self.P_init_bar * self.BAR

    @property
    def P_open(self):
        return self.P_open_bar * self.BAR

    @property
    def P_full(self):
        return self.P_full_bar * self.BAR

    @property
    def d_c(self):
        return self.d_c_mm * self.MM2M

    @property
    def F_c(self):
        return np.pi * self.d_c ** 2 / 4

    @property
    def h_max(self):
        return self.h_max_frac * self.d_c

    @property
    def h_stop(self):
        return self.h_stop_frac * self.d_c

    # Преднатяг пружины автоматически рассчитывается так, чтобы клапан
    # открылся ровно при P_open (формула из оригинального файла)
    @property
    def h_0(self):
        return (self.F_c * (self.P_open - P_atm) * float(rho_flow(0)) - self.m * g) / self.c_spring

    def validate(self):
        """Возвращает список ошибок ввода (пустой список = всё корректно)."""
        errs = []
        if self.V_total <= 0:
            errs.append("Объём V_total должен быть > 0.")
        if not (0 < self.fill_fraction < 1):
            errs.append("Доля заполнения должна быть в интервале (0; 1).")
        if self.d_c <= 0:
            errs.append("Диаметр седла должен быть > 0.")
        if self.m <= 0:
            errs.append("Масса золотника должна быть > 0.")
        if self.eta < 0:
            errs.append("Демпфирование не может быть отрицательным.")
        if self.c_spring <= 0:
            errs.append("Жёсткость пружины должна быть > 0.")
        if self.t_end <= 0:
            errs.append("Время моделирования должно быть > 0.")
        if self.P_init <= P_atm:
            errs.append("Начальное давление должно быть выше атмосферного.")
        if self.P_open <= self.P_init:
            errs.append("Давление открытия должно быть выше начального.")
        if self.P_full < self.P_open:
            errs.append("Давление полного открытия должно быть >= давления открытия.")
        # Ограничение по давлению насыщения азота (критическое давление ~33.96 бар)
        P_CRIT_N2 = 33.958e5
        for name, val in (("P_init", self.P_init), ("P_open", self.P_open),
                          ("P_full", self.P_full)):
            if val >= P_CRIT_N2:
                errs.append(f"{name} = {val/1e5:.1f} бар ≥ критического давления азота "
                            f"(33.96 бар): двухфазная модель неприменима, уменьшите давления.")
        return errs


def G_valve(P, h, F_c, d_c, rho_g):
    """Расход через клапан (из оригинального файла)."""
    if h <= 0 or P <= P_atm:
        return 0.0
    s = max(P_atm / P, (2 / (k + 1)) ** (k / (k - 1)))
    eps = np.sqrt((k / (k - 1)) * (s ** (2 / k) - s ** ((k + 1) / k)))
    return float(alpha_c(max(h / d_c, 0.0))) * F_c * eps * np.sqrt(2 * P * rho_g)


class PRV_System:
    """Класс системы для solve_ivp (полностью из оригинального файла)."""

    def __init__(self, prm: Params, fluid):
        self.prm = prm
        self.k_stop = 1e7
        self.c_stop = 10000
        self.fluid = fluid

    def thermo(self, rho_avg, u_avg, prm):
        """Термодинамика: прямое вычисление P из текущих rho и u (как в оригинале)."""
        try:
            self.fluid.update(CP.DmassUmass_INPUTS, rho_avg, u_avg)
            return self.fluid.p(), self.fluid.Q()
        except ValueError:
            return prm.P_init, 1.0

    def deriv(self, t, y):
        prm = self.prm
        M_tot, U_tot, h, v = y
        M_tot = max(M_tot, 1e-3)

        rho_avg = M_tot / prm.V_total
        u_avg = U_tot / M_tot

        # 1. Термодинамика: прямое вычисление P из текущей массы и энергии
        P, Q_vap = self.thermo(rho_avg, u_avg, prm)

        # 2. Свойства выходящего газа
        if 0 <= Q_vap < 1.0:
            rho_g_vent = self.fluid.saturated_vapor_keyed_output(CP.iDmass)
            h_g_vent = self.fluid.saturated_vapor_keyed_output(CP.iHmass)
        else:
            rho_g_vent = self.fluid.rhomass()
            h_g_vent = self.fluid.hmass()

        # 3. Массо- и энергообмен
        Gv = G_valve(P, h, prm.F_c, prm.d_c, rho_g_vent)
        dM = -Gv
        dU = prm.Q_dot - Gv * h_g_vent

        # 4. Динамика золотника
        F_p = prm.F_c * (P - P_atm) * float(rho_flow(max(h, 0) / prm.d_c))
        F_spring = prm.c_spring * (h + prm.h_0)
        F_gravity = prm.m * g
        F_damp = prm.eta * v

        F_stop = 0.0
        if h < 0:
            F_stop = -self.k_stop * h - self.c_stop * v
        elif h > prm.h_stop:
            F_stop = -self.k_stop * (h - prm.h_stop) - self.c_stop * v

        a = (F_p - F_spring - F_damp - F_gravity + F_stop) / prm.m

        return [dM, dU, v, a]


def initial_state(prm: Params):
    """Начальные условия (строгая термодинамика, как в оригинале)."""
    fluid = CP.AbstractState('HEOS', 'Nitrogen')

    # Свойства жидкости на линии насыщения при P_init
    fluid.update(CP.PQ_INPUTS, prm.P_init, 0.0)
    rho_l0 = fluid.rhomass()
    u_l0 = fluid.umass()
    T_init = fluid.T()

    # Свойства пара на линии насыщения при P_init
    fluid.update(CP.PQ_INPUTS, prm.P_init, 1.0)
    rho_g0 = fluid.rhomass()
    u_g0 = fluid.umass()

    V_liq0 = prm.V_total * prm.fill_fraction
    V_gas0 = prm.V_total - V_liq0

    M_tot0 = V_liq0 * rho_l0 + V_gas0 * rho_g0
    U_tot0 = (V_liq0 * rho_l0) * u_l0 + (V_gas0 * rho_g0) * u_g0

    y0 = [M_tot0, U_tot0, 0.0, 0.0]  # [M_tot, U_tot, h, v]
    info = dict(T_init=T_init, rho_l0=rho_l0, rho_g0=rho_g0,
                M_tot0=M_tot0, U_tot0=U_tot0,
                V_liq0=V_liq0, V_gas0=V_gas0)
    return fluid, y0, info


def compute(prm: Params, progress_cb=None):
    """Полный расчёт. Возвращает словарь результатов + текстовый отчёт."""
    errors = prm.validate()
    if errors:
        raise ValueError("\n".join(errors))

    fluid, y0, init_info = initial_state(prm)

    system = PRV_System(prm, fluid)

    sol = solve_ivp(
        system.deriv,
        [0, prm.t_end],
        y0,
        method='Radau',
        t_eval=np.linspace(0, prm.t_end, 2000),
        rtol=1e-4,
        atol=1e-6,
    )

    if not sol.success:
        raise RuntimeError(f"Интегрирование не завершилось успешно: {sol.message}")

    time = sol.t
    M_res = sol.y[0]
    U_res = sol.y[1]
    lift = sol.y[2]
    vel = sol.y[3]

    # Пересчёт давления, температуры, сухого пара, масс фаз для графиков
    press = np.zeros_like(time)
    temp = np.zeros_like(time)
    qvap = np.zeros_like(time)
    M_gas = np.zeros_like(time)     # масса газовой части
    M_liq = np.zeros_like(time)     # масса жидкой части
    calc_fluid = CP.AbstractState('HEOS', 'Nitrogen')

    def state_from(Mi, Ui):
        """P, Q, T из (M, U) — та же схема с обработкой ошибок, что и в deriv оригинала."""
        rho_avg = Mi / prm.V_total
        u_avg = Ui / Mi
        try:
            calc_fluid.update(CP.DmassUmass_INPUTS, rho_avg, u_avg)
            return calc_fluid.p(), calc_fluid.Q(), calc_fluid.T()
        except ValueError:
            return prm.P_init, 1.0, 77.36  # резервное значение T (азот, 1 атм)

    for i in range(len(time)):
        Mi = max(M_res[i], 1e-3)
        rho_avg = Mi / prm.V_total
        u_avg = U_res[i] / Mi
        P_i, q_i, T_i = state_from(Mi, U_res[i])
        press[i] = P_i
        temp[i] = T_i
        qvap[i] = q_i
        # Плотности насыщенных фаз при текущем давлении (для разделения масс)
        try:
            calc_fluid.update(CP.PQ_INPUTS, max(P_i, 5e3), 1.0)
            rho_v_sat = calc_fluid.rhomass()
            calc_fluid.update(CP.PQ_INPUTS, max(P_i, 5e3), 0.0)
            rho_l_sat = calc_fluid.rhomass()
        except ValueError:
            rho_v_sat, rho_l_sat = None, None
        # Масса газовой части: пар в двухфазной смеси + весь газ в однофазной
        if 0 <= q_i < 1.0 and rho_v_sat and rho_l_sat:
            v_mix = 1.0 / rho_avg
            v_g = 1.0 / rho_v_sat
            v_l = 1.0 / rho_l_sat
            mass_frac_gas = (v_mix - v_l) / (v_g - v_l)
            M_gas[i] = M_res[i] * mass_frac_gas
            M_liq[i] = M_res[i] - M_gas[i]
        elif q_i >= 1.0:
            M_gas[i] = M_res[i]
            M_liq[i] = 0.0
        else:
            M_gas[i] = 0.0
            M_liq[i] = M_res[i]

    # Расход через клапан во времени
    flow = np.zeros_like(time)
    for i in range(len(time)):
        Mi = max(M_res[i], 1e-3)
        rho_avg = Mi / prm.V_total
        u_avg = U_res[i] / Mi
        try:
            calc_fluid.update(CP.DmassUmass_INPUTS, rho_avg, u_avg)
            q = calc_fluid.Q()
            if 0 <= q < 1.0:
                rg = calc_fluid.saturated_vapor_keyed_output(CP.iDmass)
            else:
                rg = calc_fluid.rhomass()
        except ValueError:
            rg = 5.0
        flow[i] = G_valve(press[i], lift[i], prm.F_c, prm.d_c, rg)

    p_max = np.max(press) / 1e5
    mask = time > 5
    p_min_after = np.min(press[mask]) / 1e5 if np.any(mask) else press[0] / 1e5

    total_vented = init_info['M_tot0'] - M_res[-1]
    report = (
        f"Начальное равновесное состояние при {prm.P_init/1e5:.1f} бар: T = {init_info['T_init']:.2f} К\n"
        f"c пружины = {prm.c_spring:.0f} Н/м, Преднатяг h0 = {prm.h_0*1000:.2f} мм\n"
        f"P макс = {p_max:.3f} бар, P мин после открытия = {p_min_after:.3f} бар\n"
        f"Начальная полная масса = {init_info['M_tot0']:.2f} кг, конечная = {M_res[-1]:.2f} кг\n"
        f"Макс. подъём золотника = {np.max(lift)*1000:.2f} мм\n"
        f"Сброшено через клапан за модель: {total_vented:.2f} кг"
    )

    results = dict(
        time=time,
        press=press,          # Па
        lift=lift,            # м
        vel=vel,              # м/с
        M_tot=M_res,          # кг
        M_gas=M_gas,          # кг (газовая часть)
        M_liq=M_liq,          # кг
        temp=temp,            # К
        qvap=qvap,            # %
        flow=flow,            # кг/с
        p_max=p_max,
        p_min_after=p_min_after,
        init_info=init_info,
        report=report,
    )
    return results
