import numpy as np
from scipy.optimize import root_scalar
import matplotlib.pyplot as plt

# Константы для азота
R = 8.314  # J/(mol·K)
M = 0.0280134  # kg/mol

# Исправленные термодинамические свойства
def P_sat(T):
    """Давление насыщенного пара азота (Pa)"""
    # Уравнение Антуана для N2 (T в K, P в Pa)
    # Коэффициенты подобраны для диапазона 63-126 K
    # log10(P_mmHg) = 6.49457 - 255.68/(T - 6.0)
    if T < 63.15 or T > 126.2:
        raise ValueError(f"T={T} outside valid range for N2 saturation")
    log10_P_mmHg = 6.49457 - 255.68 / (T - 6.0)
    P_mmHg = 10 ** log10_P_mmHg
    return P_mmHg * 133.322  # конвертация в Pa

def T_sat(P):
    """Температура насыщения при заданном давлении (K)"""
    P_mmHg = P / 133.322
    log10_P = np.log10(P_mmHg)
    return 255.68 / (6.49457 - log10_P) + 6.0

def Lv(T):
    """Удельная теплота парообразования (J/mol)"""
    Tc = 126.2  # критическая температура
    Lv0 = 5580  # при T=77.36 K
    return Lv0 * ((1 - T / Tc) / (1 - 77.36 / Tc)) ** 0.38

def rho_liquid(T):
    """Плотность жидкого азота (kg/m³)"""
    # Более точная аппроксимация
    return 808.0 - 1.5 * (T - 77.36)

def solve_T_for_equilibrium(n_g, V_total, n_l, T_guess=77.0):
    """Решает уравнение равновесия: P_sat(T) = n_g * R * T / V_gas(T)"""
    def func(T):
        V_liq = (n_l * M) / rho_liquid(T)
        V_gas = V_total - V_liq
        if V_gas <= 0:
            return 1e10
        P_eq = n_g * R * T / V_gas
        P_sat_val = P_sat(T)
        return P_eq - P_sat_val

    try:
        sol = root_scalar(func, bracket=[63.2, 125.0], method='brentq')
        return sol.root if sol.converged else T_guess
    except ValueError:
        # Если bracket не подходит, используем начальное приближение
        return T_guess

# Параметры задачи
V_total = 1.30  # m³
fill_fraction = 0.18  # начальная доля заполнения жидкостью
Q_dot = 33000  # W (Дж/с)
t_end = 3600  # секунд
dt = 1.0

# Начальные условия
T0 = 64  # K (начальная температура)
P0 = P_sat(T0)  # начальное давление (~101325 Pa)
V_liq0 = V_total * fill_fraction
rho0 = rho_liquid(T0)
n_l0 = (V_liq0 * rho0) / M
V_gas0 = V_total - V_liq0
n_g0 = P0 * V_gas0 / (R * T0)  # ПРАВИЛЬНОЕ начальное количество газа!

print(f"Начальные условия:")
print(f"T0 = {T0:.2f} K")
print(f"P0 = {P0:.0f} Pa")
print(f"n_l0 = {n_l0:.2f} mol")
print(f"n_g0 = {n_g0:.4f} mol")

# История
time = [0.0]
pressure = [P0]
gas_density = [n_g0 * M / V_gas0] # плотность газовой фазы, кг/м³
temperature = [T0]

n_g = n_g0
n_l = n_l0
T = T0

for t in np.arange(dt, t_end + dt, dt):
    # Решаем равновесие для текущих n_g, n_l
    T = solve_T_for_equilibrium(n_g, V_total, n_l, T_guess=T)
    P = P_sat(T)
    
    V_liq = (n_l * M) / rho_liquid(T)
    V_gas = max(V_total - V_liq, 1e-6)
    rho_g = (n_g * M) / V_gas
    
    pressure.append(P)
    gas_density.append(rho_g)
    temperature.append(T)
    time.append(t)

    # Обновляем количество газа
    Lv_val = Lv(T)
    dn_g = (Q_dot / Lv_val) * dt
    n_g += dn_g
    n_l -= dn_g

    # Проверка: закончилась жидкость?
    if n_l <= 0:
        print(f"Вся жидкость испарилась при t={t:.0f}s")
        break

print(f"\nРезультат:")
print(f"Конечное давление: {pressure[-1]:.0f} Pa")
print(f"Конечная температура: {temperature[-1]:.2f} K")
print(f"Время моделирования: {time[-1]:.0f} s")

# --- Построение графиков ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
# График 1: Давление от времени
ax1.plot(np.array(time)/60, np.array(pressure)/1e5, 'b-', linewidth=2)
ax1.set_xlabel('Время (мин)')
ax1.set_ylabel('Давление (бар)')
ax1.set_title('Рост давления во времени')
ax1.grid(True)

# График 2: Давление от плотности газовой фазы
ax2.plot(gas_density, np.array(pressure)/1e5, 'r-', linewidth=2)
ax2.set_xlabel('Плотность газовой фазы (кг/м³)')
ax2.set_ylabel('Давление (бар)')
ax2.set_title('Зависимость давления от плотности газа')
ax2.grid(True)

plt.tight_layout()
plt.show()