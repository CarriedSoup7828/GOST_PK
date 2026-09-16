import numpy as np
from scipy.optimize import root_scalar
import matplotlib.pyplot as plt

# === Константы и свойства азота ===
R = 8.314  # Дж/(моль·К)
M = 0.0280134  # кг/моль

def P_sat(T):
    """Давление насыщенного пара азота (Па)"""
    # Уравнение Антуана для N2 (T в K, P в Pa)
    T = np.clip(T, 63.15, 126.2)
    log10_P_mmHg = 6.49457 - 255.68 / (T - 6.0)
    P_mmHg = 10 ** log10_P_mmHg
    return P_mmHg * 133.322

def T_sat(P):
    """Температура насыщения при заданном давлении (K)"""
    P_mmHg = P / 133.322
    log10_P = np.log10(P_mmHg)
    return 255.68 / (6.49457 - log10_P) + 6.0

def Lv(T):
    """Удельная теплота парообразования (Дж/моль)"""
    Tc = 126.2
    Lv0 = 5580  # при T=77.36 K
    return Lv0 * ((1 - T / Tc) / (1 - 77.36 / Tc)) ** 0.38

def rho_liquid(T):
    """Плотность жидкого азота (кг/м³)"""
    return 808.0 - 1.5 * (T - 77.36)

def rho_gas(P, T):
    """Плотность газа (кг/м³)"""
    return P * M / (R * T)

def alpha_c(h_dc):
    """Коэффициент расхода от относительного подъема (из рис. 17)"""
    # Аппроксимация для полноподъемного клапана
    if h_dc <= 0.05:
        return 0.3 + 4 * h_dc  # линейный рост
    elif h_dc <= 0.25:
        return 0.5 + 0.5 * (h_dc - 0.05) / 0.2  # переходная зона
    else:
        return 0.65  # насыщение для полноподъемного режима

def rho_flow(h_dc):
    """Коэффициент давления потока (из рис. 29-34)"""
    # Аппроксимация для полноподъемного клапана с буртом
    if h_dc <= 0.1:
        return 1.0 + 2 * h_dc
    elif h_dc <= 0.25:
        return 1.2 + 0.8 * (h_dc - 0.1) / 0.15
    else:
        return 2.0  # насыщение

def solve_T_for_equilibrium(n_g, V_total, n_l, T_guess=77.0):
    """Решает уравнение равновесия: P_sat(T) = n_g * R * T / V_gas(T)"""
    def func(T):
        V_liq = (n_l * M) / rho_liquid(T)
        V_gas = V_total - V_liq
        if V_gas <= 1e-6:
            return 1e10
        P_eq = n_g * R * T / V_gas
        P_sat_val = P_sat(T)
        return P_eq - P_sat_val

    try:
        sol = root_scalar(func, bracket=[63.2, 125.0], method='brentq')
        return sol.root if sol.converged else T_guess
    except ValueError:
        return T_guess

# === Параметры задачи ===
V_total = 1.30  # м³
fill_fraction = 0.18
Q_dot = 3000  # Вт
t_end = 100  # секунд
dt = 0.0001  # шаг по времени

# === Параметры клапана ===
d_c = 0.025  # диаметр седла (м)
F_c = np.pi * d_c**2 / 4  # площадь седла (м²)
m_spool = 0.5  # масса золотника (кг)
c_spring = 950  # жесткость пружины (Н/м)
h_0 = 0.005  # предварительный натяг (м)
eta_damp = 10  # коэффициент демпфирования (Н·с/м)
P_open = 1.2 * 101325  # давление начала открытия (Па)
P_full = 1.1 * P_open  # давление полного открытия (Па)

# === Начальные условия ===
T0 = 77.36
P0 = P_sat(T0)
V_liq0 = V_total * fill_fraction
rho0 = rho_liquid(T0)
n_l0 = (V_liq0 * rho0) / M
V_gas0 = V_total - V_liq0
n_g0 = P0 * V_gas0 / (R * T0)

# Массивы для сохранения истории
time = [0.0]
pressure = [P0]
spool_height = [0.0]
spool_velocity = [0.0]
mass_flow = [0.0]

n_g = n_g0
n_l = n_l0
T = T0
h = 0.0  # высота подъема золотника
v = 0.0  # скорость золотника

# === Цикл моделирования (RK4) ===
for t in np.arange(dt, t_end + dt, dt):
    # Функция для вычисления правых частей ОДУ
    def derivatives(n_g, h, v):
        # 1. Вычисляем температуру. 
        # T_guess берется из внешней переменной T, результат сохраняем в T_new
        T_new = solve_T_for_equilibrium(n_g, V_total, n_l, T_guess=T)
        P = P_sat(T_new)
        
        # 2. Вычисляем относительный подъем
        h_dc = h / d_c
        
        # 3. Коэффициенты из учебника
        alpha = alpha_c(h_dc)
        rho_coeff = rho_flow(h_dc)
        
        # 4. Массовый расход через клапан (используем T_new)
        if h > 0:
            G_valve = 1.59 * F_c * alpha * np.sqrt(P * rho_gas(P, T_new))
        else:
            G_valve = 0.0
        
        # 5. Силы на золотник
        F_pressure = F_c * (P - 101325) * rho_coeff
        F_spring = c_spring * (h + h_0)
        F_damping = eta_damp * v
        
        # 6. Уравнение движения золотника
        a = (F_pressure - F_spring - F_damping) / m_spool
        
        # 7. Баланс массы в резервуаре (используем T_new)
        Lv_val = Lv(T_new)
        dn_g_dt = Q_dot / Lv_val - G_valve / M
        
        return dn_g_dt, v, a
    
    # Метод Рунге-Кутты 4-го порядка
    k1_n, k1_h, k1_v = derivatives(n_g, h, v)
    k2_n, k2_h, k2_v = derivatives(n_g + 0.5*dt*k1_n, h + 0.5*dt*k1_h, v + 0.5*dt*k1_v)
    k3_n, k3_h, k3_v = derivatives(n_g + 0.5*dt*k2_n, h + 0.5*dt*k2_h, v + 0.5*dt*k2_v)
    k4_n, k4_h, k4_v = derivatives(n_g + dt*k3_n, h + dt*k3_h, v + dt*k3_v)
    
    n_g += (dt / 6) * (k1_n + 2*k2_n + 2*k3_n + k4_n)
    h += (dt / 6) * (k1_h + 2*k2_h + 2*k3_h + k4_h)
    v += (dt / 6) * (k1_v + 2*k2_v + 2*k3_v + k4_v)
    
    # Ограничения
    if h < 0:
        h = 0
        v = 0
    
    # Обновляем количество жидкости (упрощенно)
    n_l = n_l0 - (n_g - n_g0) 
    if n_l <= 0:
        print(f"Вся жидкость испарилась при t = {t:.2f} с")
        break
    
    # ВАЖНО: Обновляем внешнюю переменную T для следующего шага по времени
    T = solve_T_for_equilibrium(n_g, V_total, n_l, T_guess=T)
    P = P_sat(T)
    
    # Вычисляем массовый расход
    h_dc = h / d_c
    alpha = alpha_c(h_dc)
    G_valve = 1.59 * F_c * alpha * np.sqrt(P * rho_gas(P, T)) if h > 0 else 0.0
    
    # Сохраняем результаты
    time.append(t)
    pressure.append(P)
    spool_height.append(h)
    spool_velocity.append(v)
    mass_flow.append(G_valve)

print(f"Конечное давление: {pressure[-1]/1e5:.2f} бар")
print(f"Максимальный подъем золотника: {max(spool_height)*1000:.2f} мм")

# === Построение графиков ===
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

# График 1: Давление от времени
ax1.plot(time, np.array(pressure)/1e5, 'b-', linewidth=2)
ax1.set_xlabel('Время (с)')
ax1.set_ylabel('Давление (бар)')
ax1.set_title('Зависимость давления от времени')
ax1.grid(True)
ax1.axhline(y=P_open/1e5, color='r', linestyle='--', label=f'P_нач = {P_open/1e5:.2f} бар')
ax1.axhline(y=P_full/1e5, color='g', linestyle='--', label=f'P_полн = {P_full/1e5:.2f} бар')
ax1.legend()

# График 2: Подъем золотника от времени
ax2.plot(time, np.array(spool_height)*1000, 'r-', linewidth=2)
ax2.set_xlabel('Время (с)')
ax2.set_ylabel('Подъем золотника (мм)')
ax2.set_title('Динамика подъема золотника')
ax2.grid(True)

plt.tight_layout()
plt.show()