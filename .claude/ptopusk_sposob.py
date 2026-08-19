# -*- coding: utf-8 -*-
"""
Пропускная способность предохранительного клапана по ГОСТ 12.2.085-2017.
Метод прямого интегрирования (прил. Е.1), среда — азот, перегретый пар.

Установки: pip install CoolProp numpy
"""
import numpy as np
from CoolProp.CoolProp import PropsSI

# ==================== ВХОДНЫЕ ДАННЫЕ ====================
FLUID        = 'Nitrogen'
P_ATM        = 101320.0        # Па (3.4 ГОСТ)
P_N_GAUGE    = 5.0e5           # давление настройки Рн, Па изб. (пример: 5 бар)
SUPERHEAT    = 10.0            # перегрев пара над Тнас, К
F_MM2        = 78.5            # площадь седла F, мм² (пример: d0 = 10 мм)
ALPHA        = 0.75            # коэффициент расхода α1 (задан)
KC           = 1.0             # без мембранно-предохранительных устройств (Д.6)
KW           = 1.0             # неразгруженный клапан (табл. Д.1, прим. 2)
P_BACK_GAUGE = 0.0             # статич. противодавление, Па изб. (сброс в атмосферу)

# ==================== ДАВЛЕНИЯ ПО ГОСТ ====================
# (2): Рпо = 1,1·Рн  (для Рн от 0,3 до 6,0 МПа);  Д.4.2: Р1 = Рпо + Ратм;  Р2 = Рп + Ратм
P_PO = 1.1 * P_N_GAUGE
P1   = P_PO + P_ATM
P2   = P_BACK_GAUGE + P_ATM

# ---------- вспомогательные функции ----------
def is_two_phase(P, s):
    """Признак двухфазной области на изоэнтропе s = const."""
    try:
        Q = PropsSI('Q', 'P', P, 'S', s, FLUID)
    except Exception:
        return False
    return 0.0 <= Q <= 1.0

def dew_point_on_isentrope(P1, P2, s):
    """Е.3.3: точка пересечения изоэнтропы с линией конденсации (биcекция)."""
    if not is_two_phase(P2, s):
        return None                      # изоэнтропа не уходит в 2Ф-область (режим Г-Г)
    lo, hi = P2, P1                      # lo — двухфазная, hi — газ
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if is_two_phase(mid, s): lo = mid
        else:                    hi = mid
    return 0.5 * (lo + hi)

def mass_flux_curve(P, rho):
    """Е.1.4: G*(P_j) = rho_j·[Σ (1/rho_{i-1}+1/rho_i)·(P_{i-1}-P_i)]^1/2 (трапеции)."""
    terms = (1.0/rho[:-1] + 1.0/rho[1:]) * (P[:-1] - P[1:])
    S = np.concatenate(([0.0], np.cumsum(terms)))   # -2·∫ dP/rho  (Па·м³/кг)
    return rho * np.sqrt(S), S

def first_maximum(g):
    """Е.1.2: первый максимум G*(P0) → критический режим."""
    idx = np.where(np.diff(g) < 0.0)[0]
    return idx[0] if len(idx) else None

def kv_from_re(re):
    """Кв по (Д.12),(Д.13); при Re > 100 000 Kv = 1,0 (Д.7)."""
    if re > 1e5:  return 1.0
    if re > 1e3:  return 0.9935 + 2.8780/re + 342.75/re**1.5   # сверить с текстом при Re<1e5
    return 0.975/re + 0.980

# ==================== ШАГ 1: состояние перед клапаном ====================
T_SAT = PropsSI('T', 'Q', 1, 'P', P1, FLUID)     # температура насыщения при Р1
T1    = T_SAT + SUPERHEAT                        # перегретый пар
RHO1  = PropsSI('D', 'P', P1, 'T', T1, FLUID)
S1    = PropsSI('S', 'P', P1, 'T', T1, FLUID)
MU1   = PropsSI('V', 'P', P1, 'T', T1, FLUID)    # динамическая вязкость для Re

# ==================== ШАГ 2-4: изоэнтропа и сетка ====================
P_DEW = dew_point_on_isentrope(P1, P2, S1)
if P_DEW is not None:
    # Е.1.4 (прим.): точка пересечения изоэнтропы с границей фаз включается в сетку
    P_grid = np.concatenate([np.linspace(P1, P_DEW, 1500),
                             np.linspace(P_DEW, P2, 2500)[1:]])
else:
    P_grid = np.linspace(P1, P2, 4000)
RHO_grid = np.array(PropsSI('D', 'P', P_grid.tolist(), 'S', S1, FLUID))

# ==================== ШАГ 5-6: режим течения ====================
G_STAR, S_INT = mass_flux_curve(P_grid, RHO_grid)
j = first_maximum(G_STAR)
if j is not None:
    REGIME   = 'КРИТИЧЕСКИЙ'
    P_CRIT   = P_grid[j]
    G_IDEAL  = G_STAR[j]                 # Кп кр·sqrt(P1·rho1), (Д.3)
    RHO0, S0 = RHO_grid[j], S_INT[j]     # плотность и интеграл в горле (Е.1.6)
else:
    REGIME   = 'ДОКРИТИЧЕСКИЙ'
    P_CRIT   = None
    G_IDEAL  = G_STAR[-1]
    RHO0, S0 = RHO_grid[-1], S_INT[-1]

# ==================== ШАГ 7-8: Kv итерационно (Д.9) ====================
F_M2 = F_MM2 * 1e-6
D0   = np.sqrt(4 * F_M2 / np.pi)
Kv = 1.0
for _ in range(20):
    G = ALPHA * KC * Kv * KW * F_M2 * G_IDEAL          # (Д.1), СИ: кг/с
    Re = 4 * G / (np.pi * D0 * MU1)                    # (Д.14)
    Kv_new = kv_from_re(Re)
    if abs(Kv_new - Kv) < 0.005 * Kv:                  # Д.9: ΔKv < 0,5 %
        Kv = Kv_new
        break
    Kv = Kv_new
G = ALPHA * KC * Kv * KW * F_M2 * G_IDEAL

# ==================== ВЫВОД РЕЗУЛЬТАТОВ ====================
print(f'Р1 = {P1/1e5:.3f} бар абс., Т1 = {T1:.2f} К, rho1 = {RHO1:.3f} кг/м3')
print(f'Точка росы на изоэнтропе: '
      f'{P_DEW/1e5:.3f} бар' if P_DEW else 'изоэнтропа в области газа (Г-Г)')
print(f'Режим течения: {REGIME}' +
      (f', Ркр = {P_CRIT/1e5:.3f} бар (бета_кр = {P_CRIT/P1:.3f})' if P_CRIT else ''))
print(f'Kv = {Kv:.4f}, Re = {4*G/(np.pi*D0*MU1):.3e}')
print(f'G  = {G:.4f} кг/с = {G*3600:.1f} кг/ч')

# ---- контроль (Е.1.5): G*кр ≈ sqrt(n_кр·P_кр·rho_кр) ----
if P_CRIT:
    dP = 1e-4 * P_CRIT
    rp = PropsSI('D', 'P', P_CRIT + dP, 'S', S1, FLUID)
    rm = PropsSI('D', 'P', P_CRIT - dP, 'S', S1, FLUID)
    n_cr = np.log((P_CRIT + dP)/(P_CRIT - dP)) / np.log(rp/rm)
    print(f'Контроль (Е.1.5): sqrt(n·P·rho) = {np.sqrt(n_cr*P_CRIT*RHO0):.1f} '
          f'против G* = {G_IDEAL:.1f} кг/(м2·с)')

# ---- контроль размерной формы (Е.1.10): 1,138 (кг/ч, мм2, бар) ----
G_1138 = 1.138 * ALPHA * KC * Kv * KW * F_MM2 * RHO0 * np.sqrt(S0 / 1e5)
print(f'Контроль (Е.1.10): G = {G_1138:.1f} кг/ч (должно совпасть с {G*3600:.1f})')

# ==================== ШАГ 9: перекрёстная проверка (Е.2.2) ====================
dP = 1e-4 * P1
rp = PropsSI('D', 'P', P1 + dP, 'S', S1, FLUID)
rm = PropsSI('D', 'P', P1 - dP, 'S', S1, FLUID)
N_ISO = np.log((P1 + dP)/(P1 - dP)) / np.log(rp/rm)      # n = (d lnP/d lnrho)s, (Е.3.1)
BETA_CR = (2/(N_ISO + 1)) ** (N_ISO/(N_ISO - 1))          # (Е.2.4)
KP_CR   = np.sqrt(N_ISO * (2/(N_ISO + 1)) ** ((N_ISO + 1)/(N_ISO - 1)))  # (Е.2.6)
BETA = P2 / P1
KP = KP_CR if BETA <= BETA_CR else \
     np.sqrt((2*N_ISO/(N_ISO - 1)) * (BETA**(2/N_ISO) - BETA**((N_ISO + 1)/N_ISO)))  # (Е.2.3)
G_ANAL = ALPHA * KC * Kv * KW * KP * F_M2 * np.sqrt(P1 * RHO1)   # (Д.5)
print(f'Аналитика (Е.2.2): n = {N_ISO:.4f}, beta_кр = {BETA_CR:.3f}, '
      f'G = {G_ANAL*3600:.1f} кг/ч (расхождение {(abs(G_ANAL-G)/G*100):.2f} %)')

# ==================== ШАГ 10: верификация интегратора (идеальный газ) ====================
K = 1.40                                   # табл. И.1: азот
P_v  = np.linspace(1.0, 0.30, 4000)        # нормировано: P1 = 1, rho1 = 1
R_v  = P_v ** (1.0/K)                      # изоэнтропа идеального газа
g_v, _ = mass_flux_curve(P_v, R_v)
jv = first_maximum(g_v)
print(f'Верификация: beta_кр = {P_v[jv]:.3f} (норма 0.528), '
      f'Кп кр = {g_v[jv]:.4f} (норма 0.6847)')