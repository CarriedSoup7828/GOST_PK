import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# ================= Константы =================
R = 8.314; M = 0.0280134; k = 1.4; g = 9.81
P_atm = 101325.0

# ---------- Свойства азота ----------
def P_sat(T):
    T = np.clip(T, 63.15, 126.2)
    return 10**(6.49457 - 255.68/(T - 6.0)) * 133.322

def Lv(T):
    Tc = 126.2
    return 5580*((1 - T/Tc)/(1 - 77.36/Tc))**0.38

def rho_l(T):
    return 808.0 - 1.5*(T - 77.36)

# ---------- alpha_c(h/dc) по рис. 17 ----------
def alpha_c(hdc):
    return np.interp(hdc, [0,.05,.10,.15,.20,.25,.30,.40],
                          [0,.28,.42,.52,.60,.66,.68,.70])

# ---------- rho(h/dc): диск с буртом, рис. 31, 34, 35 ----------
def rho_flow(hdc):
    h = [0, .05, .10, .20, .30, .40]
    r = [1.0, 1.1, 1.3, 1.8, 2.2, 2.5]
    return np.interp(hdc, h, r)

# ---------- Расход через клапан ----------
def G_valve(P, T, h, F_c, d_c):
    if h <= 0 or P <= P_atm:
        return 0.0
    rho_g = P*M/(R*T)
    s = max(P_atm/P, (2/(k+1))**(k/(k-1)))
    eps = np.sqrt((k/(k-1))*(s**(2/k) - s**((k+1)/k)))
    return alpha_c(h/d_c)*F_c*eps*np.sqrt(2*P*rho_g)

# ---------- Условие насыщения ----------
def solve_T(n_g, n_l, T_guess):
    T = T_guess
    for _ in range(30):
        Vg  = max(V_total - n_l*M/rho_l(T), 1e-6)
        f   = n_g*R*T/Vg - P_sat(T)
        dT  = 1e-3
        Vg2 = max(V_total - n_l*M/rho_l(T+dT), 1e-6)
        f2  = n_g*R*(T+dT)/Vg2 - P_sat(T+dT)
        df  = (f2-f)/dT
        if abs(df) < 1e-12: break
        T -= np.clip(f/df, -2.0, 2.0)
        T = min(max(T, 63.2), 125.0)
        if abs(f) < 1.0: break
    return T

# ================= Параметры системы =================
V_total = 1.30; fill_fraction = 0.18; Q_dot = 30000.0
t_end = 40.0

# ================= Клапан =================
d_c = 0.032; F_c = np.pi*d_c**2/4
m = 0.5; eta = 30.0
P_open = 1.22e5; P_full = 1.34e5
h_max = 0.25*d_c; h_stop = 0.30*d_c

c_spring = 1500.0
h_0 = (F_c*(P_open - P_atm)*rho_flow(0) - m*g)/c_spring

# ================= Начальные условия =================
T_init = 77.36
V_liq0 = V_total*fill_fraction
n_l0 = V_liq0*rho_l(T_init)/M
n_g0 = P_sat(T_init)*(V_total - V_liq0)/(R*T_init)

y0 = [n_g0, n_l0, 0.0, 0.0]   # [ng, nl, h, v]

# ================= Класс системы для solve_ivp =================
class PRV_System:
    def __init__(self, T_guess):
        self.T_guess = T_guess
        # Штрафные коэффициенты для имитации жестких ограничителей (седла и упора)
        # Непрерывные решатели не любят жесткие "if h<0: h=0", поэтому мы добавляем 
        # виртуальную сверхжесткую пружину и демпфер в моменты удара.
        self.k_stop = 1e6  # Жесткость "виртуальной" пружины удара, Н/м
        self.c_stop = 5000 # Коэффициент демпфирования удара, Н*с/м

    def deriv(self, t, y):
        ng, nl, h, v = y
        
        # Защита от ухода в отрицательные значения на промежуточных шагах неявного решателя
        ng = max(ng, 1e-9)
        nl = max(nl, 1e-9)
        
        # 1. Термодинамика (теперь считается ВНУТРИ каждого вызова функции!)
        T = solve_T(ng, nl, self.T_guess)
        self.T_guess = T  # Сохраняем для следующего шага, чтобы ускорить сходимость
        P = P_sat(T)
        
        # 2. Массообмен
        Gv = G_valve(P, T, h, F_c, d_c)
        evap_rate = Q_dot / Lv(T)
        dng = evap_rate - Gv / M
        dnl = -evap_rate
        
        # 3. Силы, действующие на золотник
        F_p = F_c * (P - P_atm) * rho_flow(max(h, 0) / d_c)
        F_spring = c_spring * (h + h_0)
        F_gravity = m * g
        F_damp = eta * v
        
        # Штрафная сила (имитация удара об седло или ограничитель)
        F_stop = 0.0
        if h < 0:
            F_stop = -self.k_stop * h - self.c_stop * v
        elif h > h_stop:
            F_stop = -self.k_stop * (h - h_stop) - self.c_stop * v
            
        a = (F_p - F_spring - F_damp - F_gravity + F_stop) / m
        
        return [dng, dnl, v, a]

# ================= Интегрирование =================
system = PRV_System(T_init)

# Используем неявный метод Radau для жестких систем (stiff ODE)
# rtol и atol задают точность. Для жестких систем лучше использовать векторные допуски.
sol = solve_ivp(
    system.deriv, 
    [0, t_end], 
    y0, 
    method='Radau', 
    t_eval=np.linspace(0, t_end, 2000), # Вывод в 2000 точек для гладких графиков
    rtol=1e-5, 
    atol=1e-7
)

# Извлечение результатов
time = sol.t
n_g_res = sol.y[0]
n_l_res = sol.y[1]
lift = sol.y[2]

# Пересчет давления и расхода по сохраненным точкам для графиков
press = []
flow = []
for i in range(len(time)):
    T = solve_T(n_g_res[i], n_l_res[i], T_init)
    P = P_sat(T)
    press.append(P)
    flow.append(G_valve(P, T, lift[i], F_c, d_c))

press = np.array(press)
flow = np.array(flow)

print(f"c пружины = {c_spring:.0f} Н/м, h0 = {h_0*1000:.2f} мм")
print(f"P макс = {max(press)/1e5:.3f} бар, P мин после открытия = {min(press[time > 5])/1e5:.3f} бар")
print(f"Макс. подъем = {max(lift)*1000:.2f} мм")

# ================= Графики =================
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8), sharex=True)

ax1.plot(time, press/1e5, 'b-', linewidth=1.5)
ax1.axhline(P_open/1e5, color='r', ls='--', label='P_нач')
ax1.axhline(P_full/1e5, color='g', ls='--', label='P_полн')
ax1.set_ylabel('Давление (бар)'); ax1.grid(); ax1.legend()
ax1.set_title('Давление от времени (Q = 33 кВт) - Метод Radau (Жесткая система)')

ax2.plot(time, lift*1000, 'r-', linewidth=1.5)
ax2.set_ylabel('Подъем золотника (мм)'); ax2.set_xlabel('Время (с)'); ax2.grid()

plt.tight_layout()
plt.savefig('ln2_valve_pop_radau.png', dpi=150)
plt.show()