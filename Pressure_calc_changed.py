import numpy as np
import matplotlib.pyplot as plt

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
    h = [0, .05, .10, .15, .20, .25, .30, .40]
    a = [0, .28, .42, .52, .60, .66, .68, .70]
    return np.interp(hdc, h, a)

# ---------- rho(h/dc) по рис. 29-34 (плоский диск с юбкой) ----------
def rho_flow(hdc):
    h = [0, .05, .10, .20, .30, .40]
    r = [1.0, 1.1, 1.3, 1.8, 2.2, 2.5]
    return np.interp(hdc, h, r)

# ---------- Расход через клапан, (II.16)-(II.17) в СИ ----------
def G_valve(P, T, h, F_c, d_c):
    if h <= 0 or P <= P_atm:
        return 0.0
    rho_g = P*M/(R*T)
    sigma = P_atm/P
    s_cr = (2/(k+1))**(k/(k-1))          # ~0.528
    s = max(sigma, s_cr)
    eps = np.sqrt((k/(k-1))*(s**(2/k) - s**((k+1)/k)))   # (II.16)
    return alpha_c(h/d_c)*F_c*eps*np.sqrt(2*P*rho_g)    # (II.17)

# ---------- Условие насыщения: P_sat(T)*Vg = ng*R*T ----------
def solve_T(n_g, n_l, T_guess):
    T = T_guess
    for _ in range(30):
        Vg = max(V_total - n_l*M/rho_l(T), 1e-6)
        f = n_g*R*T/Vg - P_sat(T)
        dT = 1e-3
        Vg2 = max(V_total - n_l*M/rho_l(T+dT), 1e-6)
        f2 = n_g*R*(T+dT)/Vg2 - P_sat(T+dT)
        df = (f2 - f)/dT
        if abs(df) < 1e-12: break
        T -= np.clip(f/df, -2.0, 2.0)
        T = min(max(T, 63.2), 125.0)
        if abs(f) < 1.0: break
    return T

# ================= Параметры системы =================
V_total = 1.30; fill_fraction = 0.18; Q_dot = 20000.0
t_end = 40.0; dt = 1e-3

# ================= Параметры клапана =================
d_c = 0.032; F_c = np.pi*d_c**2/4
m = 0.5; eta = 30.0
P_open = 1.22e5; P_full = 1.34e5
h_max = 0.25*d_c; h_stop = 0.30*d_c

# Настройка пружины по балансу сил (III.41)-(III.44):
# при h=0:  Fc*(P_open-P_atm)*rho(0) = c*h0 + m*g
# при hmax: Fc*(P_full-P_atm)*rho(hmax) = c*(h0+hmax) + m*g
c_spring = 2200
# c_spring = F_c*((P_full-P_atm)*rho_flow(0.25) - (P_open-P_atm)*rho_flow(0))/h_max
h_0 = (F_c*(P_open-P_atm)*rho_flow(0) - m*g)/c_spring

# ================= Начальные условия =================
T = 77.36
V_liq0 = V_total*fill_fraction
n_l = V_liq0*rho_l(T)/M
n_g = P_sat(T)*(V_total-V_liq0)/(R*T)
h = 0.0; v = 0.0

time=[0]; press=[P_sat(T)]; lift=[0]; flow=[0]

# ================= Правые части (III.56) + баланс масс =================
def deriv(y, T_, P_):
    ng, nl, hh, vv = y
    Lv_ = Lv(T_)
    Gv = G_valve(P_, T_, hh, F_c, d_c)
    dng = Q_dot/Lv_ - Gv/M          # испарение - сброс
    dnl = -Q_dot/Lv_                # жидкость уходит ТОЛЬКО на испарение
    F_p = F_c*(P_-P_atm)*rho_flow(max(hh,0)/d_c)
    a = (F_p - c_spring*(hh+h_0) - eta*vv - m*g)/m
    return np.array([dng, dnl, vv, a])

# ================= Цикл RK4 =================
y = np.array([n_g, n_l, h, v])
n_steps = int(t_end/dt)
for i in range(n_steps):
    ng, nl, hh, vv = y
    T = solve_T(ng, nl, T)
    P = P_sat(T)

    k1 = deriv(y, T, P)
    k2 = deriv(y+0.5*dt*k1, T, P)
    k3 = deriv(y+0.5*dt*k2, T, P)
    k4 = deriv(y+dt*k3, T, P)
    y = y + dt/6*(k1+2*k2+2*k3+k4)

    if y[2] < 0: y[2]=0; y[3]=max(y[3],0)          # седло
    if y[2] > h_stop: y[2]=h_stop; y[3]=0          # ограничитель
    if y[1] <= 0:
        print(f"Жидкость испарилась при t={i*dt:.1f} с"); break

    time.append(i*dt); press.append(P); lift.append(y[2]); flow.append(G_valve(P,T,y[2],F_c,d_c))

print(f"P_нач открытия (факт): {P_open/1e5:.2f} бар, P_полн: {P_full/1e5:.2f} бар")
print(f"c пружины = {c_spring:.0f} Н/м, h0 = {h_0*1000:.2f} мм")
print(f"Конечное P = {press[-1]/1e5:.3f} бар, макс. подъем = {max(lift)*1000:.2f} мм")

# ================= Графики =================
fig,(ax1,ax2)=plt.subplots(2,1,figsize=(10,8),sharex=True)
ax1.plot(time,np.array(press)/1e5,'b-')
ax1.axhline(P_open/1e5,color='r',ls='--',label='P_нач')
ax1.axhline(P_full/1e5,color='g',ls='--',label='P_полн')
ax1.set_ylabel('Давление (бар)'); ax1.grid(); ax1.legend()
ax1.set_title('Давление от времени')
ax2.plot(time,np.array(lift)*1000,'r-')
ax2.set_ylabel('Подъем золотника'); ax2.set_xlabel('Время (с)'); ax2.grid()
plt.tight_layout(); plt.savefig('ln2_valve_fixed.png',dpi=150); plt.show()