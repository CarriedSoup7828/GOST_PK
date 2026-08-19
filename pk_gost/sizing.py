# -*- coding: utf-8 -*-
"""
Площадь седла и подбор типоразмера.

  Д.1  G = alpha*Kc*Kv*Kw*G*ideal*F               — пропускная способность
  Д.2  F = Gав / (alpha*Kc*Kv*Kw*G*ideal*N)       — минимальная площадь седла
  Д.7-Д.8, Д.9-Д.10 — те же формулы в кг/ч, мм2, МПа/бар
  5.5  «Площадь седла ПК, выбранного из каталога, должна быть равной или
       ближайшей большей к расчётной минимальной площади седла»
  6.7.1 если расчётная пропускная способность превышает заданный аварийный
       расход более чем на 10 %, рекомендуется изменить сбросную систему
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from . import coeffs as cf

# Константы перевода единиц (проверка вывода — в тестах)
#   G[кг/ч] = 3600*G[кг/с]; F[м2] = F[мм2]*1e-6; sqrt(P[Па]) = sqrt(P[МПа])*sqrt(1e6)
#   => 3600*1e-6*sqrt(1e6) = 3,60      (Д.7, Е.1.8)
#      3600*1e-6*sqrt(1e5) = 1,1384    (Д.9, Е.1.10)
C_MPA = 3.60
C_BAR = 1.138


@dataclass
class SizingResult:
    G_av: float              # требуемый аварийный расход, кг/с
    G_ideal: float           # массовая скорость, кг/(м2*с)
    alpha: float
    Kc: float
    Kv: float
    Kw: float
    N: int
    F_required_m2: float     # минимальная площадь седла по Д.2
    d0_required_mm: float
    d0_selected_mm: float | None
    F_selected_m2: float | None
    G_selected: float | None  # фактическая пропускная способность выбранного ПК
    margin_pct: float | None  # запас над Gав, %
    Re: float
    Kv_iterations: int

    @property
    def F_required_mm2(self):
        return self.F_required_m2 * 1e6

    @property
    def F_selected_mm2(self):
        return None if self.F_selected_m2 is None else self.F_selected_m2 * 1e6


def capacity(alpha, Kc, Kv, Kw, G_ideal, F_m2):
    """(Д.1) Реальная пропускная способность ПК, кг/с."""
    return alpha * Kc * Kv * Kw * G_ideal * F_m2


def required_area(G_av, alpha, Kc, Kv, Kw, G_ideal, N=1):
    """(Д.2) Минимальная площадь седла клапана, м2."""
    return G_av / (alpha * Kc * Kv * Kw * G_ideal * N)


def capacity_kgh_bar(alpha, Kc, Kv, Kw, Kn, F_mm2, P1_bar, rho1):
    """
    (Д.9) G = 1,138*alpha*Kc*Kv*Kw*Кп*F*sqrt(P1*rho1)
    для G в кг/ч, F в мм2, P в бар.
    """
    return C_BAR * alpha * Kc * Kv * Kw * Kn * F_mm2 * math.sqrt(P1_bar * rho1)


def capacity_kgh_mpa(alpha, Kc, Kv, Kw, Kn, F_mm2, P1_mpa, rho1):
    """(Д.7) То же для P в МПа с константой 3,60."""
    return C_MPA * alpha * Kc * Kv * Kw * Kn * F_mm2 * math.sqrt(P1_mpa * rho1)


def select_from_catalog(F_required_m2, catalog_d0_mm):
    """
    (5.5) Подбор из каталога площади «равной или ближайшей большей».

    Возвращает (d0_мм, F_м2) либо (None, None), если каталог исчерпан.
    """
    F_req_mm2 = F_required_m2 * 1e6
    for d0 in sorted(catalog_d0_mm):
        F = math.pi * d0 ** 2 / 4.0
        if F >= F_req_mm2:
            return float(d0), F * 1e-6
    return None, None


def size_valve(G_av, G_ideal, alpha, Kc_val, Kw_val, mu, N=1,
               catalog_d0_mm=(), Kv_func=cf.Kv_from_Re):
    """
    Полный подбор клапана с итерационным уточнением Kv (Д.9.1).

    Алгоритм Д.9.1:
      «по уравнению (Д.17) или (Д.18) рассчитать начальное приближённое число
       Рейнольдса Re0 при Kv = 1,0; если Re0 >= 100000, то Kv = 1,0 и итераций
       не требуется; если Re0 < 100000, то Kv определить из уравнения
       Re = sqrt(Kv(Re)*Re0)... ; с учётом найденного значения Kv определить
       минимальную площадь сечения клапана (по уравнению (Д.2)).»

    Число Рейнольдса для предварительного расчёта площади — по (Д.19):
        Re = (1/mu)*sqrt( 4*alpha*Kc*Kv*Kw*G*ideal*Gав*N / pi )
    """
    def F_of_Kv(Kv):
        return required_area(G_av, alpha, Kc_val, Kv, Kw_val, G_ideal, N)

    def Re_of_Kv(Kv):
        # (Д.19): Re = (1/mu)*sqrt(4*alpha*Kc*Kv*Kw*G*ideal*Gав*N/pi)
        return (1.0 / mu) * math.sqrt(
            4.0 * alpha * Kc_val * Kv * Kw_val * G_ideal * G_av * N / math.pi)

    Kv = 1.0
    Re = Re_of_Kv(Kv)
    iters = 0
    if Re < cf.RE_TURBULENT:
        for iters in range(1, 51):
            Kv_new = Kv_func(Re)
            done = abs(Kv_new - Kv) < 0.005 * Kv
            Kv = Kv_new
            Re = Re_of_Kv(Kv)
            if done:
                break

    F_req = F_of_Kv(Kv)
    d0_req = math.sqrt(4.0 * F_req / math.pi) * 1e3     # мм

    d0_sel, F_sel = select_from_catalog(F_req, catalog_d0_mm) if catalog_d0_mm else (None, None)

    G_sel = margin = None
    if F_sel is not None:
        G_sel = capacity(alpha, Kc_val, Kv, Kw_val, G_ideal, F_sel) * N
        margin = (G_sel / G_av - 1.0) * 100.0

    return SizingResult(
        G_av=G_av, G_ideal=G_ideal, alpha=alpha, Kc=Kc_val, Kv=Kv, Kw=Kw_val, N=N,
        F_required_m2=F_req, d0_required_mm=d0_req,
        d0_selected_mm=d0_sel, F_selected_m2=F_sel,
        G_selected=G_sel, margin_pct=margin, Re=Re, Kv_iterations=iters)
