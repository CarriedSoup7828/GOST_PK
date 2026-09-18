import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.interpolate import CubicSpline
import CoolProp.CoolProp as CP

# ================= Константы =================
R = 8.314; M_molar = 0.0280134; k = 1.4; g = 9.81
P_atm = 101325.0

# ---------- Инициализация CoolProp (HEOS для скорости) ----------
fluid = CP.AbstractState('HEOS', 'Nitrogen')

# ---------- Кубические сплайны (гладкие производные) ----------
hdc_nodes = [0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40]
alpha_nodes = [0, 0.28, 0.42, 0.52, 0.60, 0.66, 0.68, 0.70]
alpha_c = CubicSpline(hdc_nodes, alpha_nodes, bc_type='clamped')

hdc_rho = [0, 0.05, 0.10, 0.20, 0.30, 0.40]
rho_nodes = [1.0, 1.1, 1.3, 1.8, 2.2, 2.5]
rho_flow = CubicSpline(hdc_rho, rho_nodes, bc_type='clamped')

# ---------- Расход через клапан ----------
def G_valve(P, h, F_c, d_c, rho_g):
    if h <= 0 or P <= P_atm:
        return 0.0
    s = max(P_atm/P, (2/(k+1))**(k/(k-1)))
    eps = np.sqrt((k/(k-1))*(s**(2/k) - s**((k+1)/k)))
    return float(alpha_c(max(h/d_c, 0.0))) * F_c * eps * np.sqrt(2 * P * rho_g)

# ================= Параметры системы =================
V_total = 1.30; fill_fraction = 0.18; Q_dot = 5000.0 # Увеличен теплоприток для наглядности
t_end = 500.0

# ================= НАСТРОЙКА ДАВЛЕНИЙ =================
P_init = 4.0 * 1e5    # Начальное давление в резервуаре (5 бар)
P_open = 4.22 * 1e5    # Давление открытия клапана (6 бар)
P_full = 4.34 * 1e5    # Давление полного открытия (обычно +10% от P_open)

# ================= Клапан =================
d_c = 0.032; F_c = np.pi*d_c**2/4
m = 0.5; eta = 30.0
h_max = 0.25*d_c; h_stop = 0.30*d_c

# Пружина должна быть более жесткой, чтобы сдерживать 6 бар на площади F_c
c_spring = 1000.0 
# Преднатяг пружины автоматически рассчитывается так, чтобы клапан открылся ровно при P_open
h_0 = (F_c*(P_open - P_atm)*rho_flow(0) - m*g)/c_spring

# ================= Начальные условия (Строгая термодинамика) =================
# Инициализация равновесной смеси (жидкость + пар) по давлению P_init

# Свойства жидкости на линии насыщения при P_init
fluid.update(CP.PQ_INPUTS, P_init, 0.0)
rho_l0 = fluid.rhomass()
u_l0 = fluid.umass()
T_init = fluid.T() # Температура теперь вычисляется автоматически

# Свойства пара на линии насыщения при P_init
fluid.update(CP.PQ_INPUTS, P_init, 1.0)
rho_g0 = fluid.rhomass()
u_g0 = fluid.umass()

V_liq0 = V_total * fill_fraction
V_gas0 = V_total - V_liq0

# Масса и энергия системы в начальный момент времени
M_tot0 = V_liq0 * rho_l0 + V_gas0 * rho_g0
U_tot0 = (V_liq0 * rho_l0) * u_l0 + (V_gas0 * rho_g0) * u_g0

y0 = [M_tot0, U_tot0, 0.0, 0.0]   # Вектор состояния: [M_tot, U_tot, h, v]

print(f"Начальное равновесное состояние при {P_init/1e5:.1f} бар: T = {T_init:.2f} К")

# ================= Класс системы для solve_ivp =================
class PRV_System:
    def __init__(self):
        self.k_stop = 1e7  
        self.c_stop = 10000 
        self.fluid = CP.AbstractState('HEOS', 'Nitrogen')

    def deriv(self, t, y):
        M_tot, U_tot, h, v = y
        M_tot = max(M_tot, 1e-3)
        
        rho_avg = M_tot / V_total
        u_avg = U_tot / M_tot
        
        # 1. Термодинамика: прямое вычисление P из текущей массы и энергии
        try:
            self.fluid.update(CP.DmassUmass_INPUTS, rho_avg, u_avg)
            P = self.fluid.p()
            Q_vap = self.fluid.Q() 
        except ValueError:
            P = P_init
            Q_vap = 1.0
            
        # 2. Свойства выходящего газа
        if 0 <= Q_vap < 1.0:
            rho_g_vent = self.fluid.saturated_vapor_keyed_output(CP.iDmass)
            h_g_vent = self.fluid.saturated_vapor_keyed_output(CP.iHmass)
        else:
            rho_g_vent = self.fluid.rhomass()
            h_g_vent = self.fluid.hmass()
            
        # 3. Массо- и энергообмен
        Gv = G_valve(P, h, F_c, d_c, rho_g_vent)
        dM = -Gv
        dU = Q_dot - Gv * h_g_vent 
        
        # 4. Динамика золотника
        F_p = F_c * (P - P_atm) * float(rho_flow(max(h, 0) / d_c))
        F_spring = c_spring * (h + h_0)
        F_gravity = m * g
        F_damp = eta * v
        
        F_stop = 0.0
        if h < 0:
            F_stop = -self.k_stop * h - self.c_stop * v
        elif h > h_stop:
            F_stop = -self.k_stop * (h - h_stop) - self.c_stop * v
            
        a = (F_p - F_spring - F_damp - F_gravity + F_stop) / m
        
        return [dM, dU, v, a]

# ================= Интегрирование =================
system = PRV_System()

sol = solve_ivp(
    system.deriv, 
    [0, t_end], 
    y0, 
    method='Radau', 
    t_eval=np.linspace(0, t_end, 2000), 
    rtol=1e-4, 
    atol=1e-6
)

# Извлечение результатов
time = sol.t
M_res = sol.y[0]
U_res = sol.y[1]
lift = sol.y[2]

# Быстрый пересчет давления для графиков
press = np.zeros_like(time)
calc_fluid = CP.AbstractState('HEOS', 'Nitrogen')

for i in range(len(time)):
    rho_avg = M_res[i] / V_total
    u_avg = U_res[i] / M_res[i]
    calc_fluid.update(CP.DmassUmass_INPUTS, rho_avg, u_avg)
    press[i] = calc_fluid.p()
    
print(f"c пружины = {c_spring:.0f} Н/м, Преднатяг h0 = {h_0*1000:.2f} мм")
print(f"P макс = {np.max(press)/1e5:.3f} бар, P мин после открытия = {np.min(press[time > 5])/1e5:.3f} бар")

# ================= Графики =================
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8), sharex=True)

ax1.plot(time, press/1e5, 'b-', linewidth=1.5)
ax1.axhline(P_open/1e5, color='r', ls='--', label='P_open ')
ax1.axhline(P_init/1e5, color='k', ls=':', label='P_init')
ax1.set_ylabel('Давление (бар)'); ax1.grid(); ax1.legend(loc='upper right')
# ax1.set_title('Динамика клапана: старт с 5 бар, открытие на 6 бар')

ax2.plot(time, lift*1000, 'r-', linewidth=1.5)
ax2.set_ylabel('Подъем золотника (мм)'); ax2.set_xlabel('Время (с)'); ax2.grid()

plt.tight_layout()
plt.show()