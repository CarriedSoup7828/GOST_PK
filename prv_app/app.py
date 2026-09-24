# -*- coding: utf-8 -*-
"""
GUI-приложение расчёта динамики предохранительного клапана.

Слева — поля ввода известных данных (как в начале Another_PRessure_Radau.py),
справа — область с графиками: динамика давления, подъём золотника,
изменение массы газовой части (и дополнительные графики для полноты
информативности оригинальных построений).

Расчёт выполняется в фоновом потоке (scipy Radau + CoolProp).
"""

import sys
import os

import numpy as np

import matplotlib
matplotlib.use('QtAgg')
# Только DejaVu Sans: гарантированно ставится с matplotlib и содержит кириллицу.
matplotlib.rcParams['font.family'] = ['DejaVu Sans']

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QGridLayout, QGroupBox,
    QLabel, QLineEdit, QPushButton, QTextEdit, QVBoxLayout, QHBoxLayout,
    QMessageBox, QSplitter, QFrame, QStatusBar,
)

from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar

try:
    from prv_app.engine import Params, compute
except ImportError:  # запуск скриптом из каталога prv_app
    from engine import Params, compute


# ----------------------------------------------------------------------------- 
class CalcThread(QThread):
    """Фоновый поток интегрирования, чтобы GUI не «зависал»."""
    finished_ok = Signal(dict)
    failed = Signal(str)

    def __init__(self, prm: Params, parent=None):
        super().__init__(parent)
        self.prm = prm

    def run(self):
        try:
            res = compute(self.prm)
            self.finished_ok.emit(res)
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(str(exc))


# -----------------------------------------------------------------------------
def make_line_edit(value, suffix=''):
    w = QWidget()
    lay = QHBoxLayout(w)
    lay.setContentsMargins(0, 0, 0, 0)
    edit = QLineEdit(f"{value:g}")
    lab = QLabel(suffix)
    lab.setStyleSheet("color: #555;")
    lay.addWidget(edit)
    lay.addWidget(lab)
    w.edit = edit
    return w


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Расчёт динамики PRV (Radau + CoolProp)")
        self.resize(1500, 900)
        self.calc_thread = None

        central = QWidget()
        root = QHBoxLayout(central)
        splitter = QSplitter(Qt.Horizontal)
        root.addWidget(splitter)
        self.setCentralWidget(central)

        # ================= Левая панель: ввод данных =================
        left = QWidget()
        left_w = QVBoxLayout(left)

        grid_host = QWidget()
        gw = QVBoxLayout(grid_host)
        gw.setSpacing(10)

        # ---- Группа: резервуар и процесс ----
        gb1 = QGroupBox("Резервуар и нагрев")
        g1 = QGridLayout(gb1)
        self.ed_V = make_line_edit(1.30, "м³")
        self.ed_fill = make_line_edit(0.18, "-")
        self.ed_Q = make_line_edit(5000.0, "Вт")
        self.ed_tend = make_line_edit(500.0, "с")
        for row, (lab, w) in enumerate([
            ("Полный объём V_total", self.ed_V),
            ("Доля заполнения жидкостью", self.ed_fill),
            ("Теплоприток Q̇", self.ed_Q),
            ("Время моделирования t_end", self.ed_tend),
        ]):
            g1.addWidget(QLabel(lab), row, 0)
            g1.addWidget(w, row, 1)
        gw.addWidget(gb1)

        # ---- Группа: давления ----
        gb2 = QGroupBox("Давления")
        g2 = QGridLayout(gb2)
        self.ed_Pinit = make_line_edit(4.0, "бар")
        self.ed_Popen = make_line_edit(4.22, "бар")
        self.ed_Pfull = make_line_edit(4.34, "бар")
        for row, (lab, w) in enumerate([
            ("Начальное давление P_init", self.ed_Pinit),
            ("Давление открытия P_open", self.ed_Popen),
            ("Давление полного открытия P_full", self.ed_Pfull),
        ]):
            g2.addWidget(QLabel(lab), row, 0)
            g2.addWidget(w, row, 1)
        gw.addWidget(gb2)

        # ---- Группа: клапан ----
        gb3 = QGroupBox("Клапан")
        g3 = QGridLayout(gb3)
        self.ed_dc = make_line_edit(32.0, "мм")
        self.ed_m = make_line_edit(0.5, "кг")
        self.ed_eta = make_line_edit(30.0, "Н·с/м")
        self.ed_c = make_line_edit(1000.0, "Н/м")
        self.ed_hstop = make_line_edit(0.30, "×d_c")
        for row, (lab, w) in enumerate([
            ("Диаметр седла d_c", self.ed_dc),
            ("Масса золотника m", self.ed_m),
            ("Демпфирование η", self.ed_eta),
            ("Жёсткость пружины c", self.ed_c),
            ("Ход ограничителя h_stop", self.ed_hstop),
        ]):
            g3.addWidget(QLabel(lab), row, 0)
            g3.addWidget(w, row, 1)
        gw.addWidget(gb3)

        # ---- Кнопки ----
        btn_row = QHBoxLayout()
        self.btn_calc = QPushButton("РАССЧИТАТЬ")
        self.btn_calc.setStyleSheet(
            "QPushButton{background:#1565c0;color:white;padding:10px;"
            "font-weight:bold;border-radius:6px;font-size:13pt;}"
            "QPushButton:disabled{background:#90a4ae;}")
        self.btn_reset = QPushButton("Сброс")
        self.btn_save = QPushButton("Сохранить PNG")
        for b in (self.btn_reset, self.btn_save):
            b.setStyleSheet("padding:8px;")
        btn_row.addWidget(self.btn_calc)
        btn_row.addWidget(self.btn_reset)
        btn_row.addWidget(self.btn_save)
        gw.addLayout(btn_row)

        self.btn_calc.clicked.connect(self.on_calculate)
        self.btn_reset.clicked.connect(self.on_reset)
        self.btn_save.clicked.connect(self.on_save_png)

        # ---- Отчёт ----
        gb4 = QGroupBox("Результаты расчёта")
        g4l = QVBoxLayout(gb4)
        self.txt_report = QTextEdit()
        self.txt_report.setReadOnly(True)
        self.txt_report.setStyleSheet("font-family: monospace;")
        g4l.addWidget(self.txt_report)
        gw.addWidget(gb4, 1)

        left_w.addWidget(grid_host)

        # ================= Правая панель: графики =================
        right = QWidget()
        rw = QVBoxLayout(right)
        rw.setContentsMargins(2, 2, 2, 2)

        self.fig = Figure(figsize=(9, 8), dpi=100, facecolor='white')
        self.canvas = FigureCanvas(self.fig)
        self.toolbar = NavigationToolbar(self.canvas, right)
        rw.addWidget(self.toolbar)
        rw.addWidget(self.canvas, 1)

        self.axP = self.fig.add_subplot(4, 1, 1)
        self.axL = self.fig.add_subplot(4, 1, 2, sharex=self.axP)
        self.axM = self.fig.add_subplot(4, 1, 3, sharex=self.axP)
        self.axF = self.fig.add_subplot(4, 1, 4, sharex=self.axP)
        self._draw_placeholder()

        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([420, 1080])

        sb = QStatusBar()
        sb.showMessage("Введите известные данные и нажмите «Рассчитать».")
        self.setStatusBar(sb)

    # -------------------------------------------------------------
    def _draw_placeholder(self):
        for ax in (self.axP, self.axL, self.axM, self.axF):
            ax.clear()
            ax.grid(True, alpha=0.3)
        self.axP.set_ylabel('Давление (бар)')
        self.axL.set_ylabel('Подъём золотника (мм)')
        self.axM.set_ylabel('Масса (кг)')
        self.axF.set_ylabel('Расход (кг/с)')
        self.axF.set_xlabel('Время (с)')
        self.axP.text(0.5, 0.5, 'Область под графики — выполните расчёт',
                      ha='center', va='center', color='#999',
                      transform=self.axP.transAxes)
        self.fig.tight_layout()
        self.canvas.draw_idle()

    # -------------------------------------------------------------
    def collect_params(self) -> Params:
        prm = Params()
        prm.V_total = float(self.ed_V.edit.text().replace(',', '.'))
        prm.fill_fraction = float(self.ed_fill.edit.text().replace(',', '.'))
        prm.Q_dot = float(self.ed_Q.edit.text().replace(',', '.'))
        prm.t_end = float(self.ed_tend.edit.text().replace(',', '.'))
        prm.P_init_bar = float(self.ed_Pinit.edit.text().replace(',', '.'))
        prm.P_open_bar = float(self.ed_Popen.edit.text().replace(',', '.'))
        prm.P_full_bar = float(self.ed_Pfull.edit.text().replace(',', '.'))
        prm.d_c_mm = float(self.ed_dc.edit.text().replace(',', '.'))
        prm.m = float(self.ed_m.edit.text().replace(',', '.'))
        prm.eta = float(self.ed_eta.edit.text().replace(',', '.'))
        prm.c_spring = float(self.ed_c.edit.text().replace(',', '.'))
        prm.h_stop_frac = float(self.ed_hstop.edit.text().replace(',', '.'))
        return prm

    def on_reset(self):
        defaults = Params()
        pairs = [
            (self.ed_V, defaults.V_total), (self.ed_fill, defaults.fill_fraction),
            (self.ed_Q, defaults.Q_dot), (self.ed_tend, defaults.t_end),
            (self.ed_Pinit, defaults.P_init_bar), (self.ed_Popen, defaults.P_open_bar),
            (self.ed_Pfull, defaults.P_full_bar), (self.ed_dc, defaults.d_c_mm),
            (self.ed_m, defaults.m), (self.ed_eta, defaults.eta),
            (self.ed_c, defaults.c_spring), (self.ed_hstop, defaults.h_stop_frac),
        ]
        for w, val in pairs:
            w.edit.setText(f"{val:g}")
        self.txt_report.clear()
        self._draw_placeholder()
        self.statusBar().showMessage("Значения сброшены по умолчанию.")

    def on_calculate(self):
        if self.calc_thread is not None and self.calc_thread.isRunning():
            return
        try:
            prm = self.collect_params()
        except ValueError:
            QMessageBox.critical(self, "Ошибка ввода",
                                 "Все поля должны содержать корректные числа.")
            return
        errs = prm.validate()
        if errs:
            QMessageBox.critical(self, "Ошибка ввода", "\n".join(errs))
            return

        self.btn_calc.setEnabled(False)
        self.statusBar().showMessage("Идёт расчёт (Radau)… это может занять несколько секунд.")
        QApplication.setOverrideCursor(Qt.BusyCursor)

        self.calc_thread = CalcThread(prm, self)
        self.calc_thread.finished_ok.connect(self.on_finished)
        self.calc_thread.failed.connect(self.on_failed)
        self.calc_thread.start()

    def on_failed(self, msg: str):
        QApplication.restoreOverrideCursor()
        self.btn_calc.setEnabled(True)
        self.statusBar().showMessage("Расчёт завершён с ошибкой.")
        QMessageBox.critical(self, "Ошибка расчёта", msg)

    def on_finished(self, res: dict):
        QApplication.restoreOverrideCursor()
        self.btn_calc.setEnabled(True)
        self.statusBar().showMessage("Расчёт завершён.")
        self.last_res = res
        self.txt_report.setPlainText(res['report'])
        self._plot(res)

    # -------------------------------------------------------------
    def _plot(self, res: dict):
        prm = self.prm_used if hasattr(self, 'prm_used') else None
        time = res['time']
        press = res['press'] / 1e5      # бар
        lift = res['lift'] * 1000       # мм
        M_gas = res['M_gas']
        M_liq = res['M_liq']
        M_tot = res['M_tot']
        flow = res['flow']

        # значения давлений для линий (берём из отчёта/полей ввода)
        try:
            p_init_b = float(self.ed_Pinit.edit.text().replace(',', '.'))
            p_open_b = float(self.ed_Popen.edit.text().replace(',', '.'))
            p_full_b = float(self.ed_Pfull.edit.text().replace(',', '.'))
        except ValueError:
            p_init_b = p_open_b = p_full_b = None

        for ax in (self.axP, self.axL, self.axM, self.axF):
            ax.clear()

        # --- График 1: давление (аналогично ax1 оригинала + доп. линии) ---
        self.axP.plot(time, press, 'b-', linewidth=1.5, label='Давление в резервуаре')
        if p_open_b is not None:
            self.axP.axhline(p_open_b, color='r', ls='--', label='P_open')
            self.axP.axhline(p_full_b, color='orange', ls='--', label='P_full')
            self.axP.axhline(p_init_b, color='k', ls=':', label='P_init')
        self.axP.set_ylabel('Давление (бар)')
        self.axP.set_title('Динамика давления в резервуаре')
        self.axP.grid(True)
        self.axP.legend(loc='upper right', fontsize=8)

        # --- График 2: подъём золотника (аналог ax2 оригинала) ---
        self.axL.plot(time, lift, 'r-', linewidth=1.5, label='Подъём h(t)')
        self.axL.set_ylabel('Подъём золотника (мм)')
        self.axL.set_title('Подъём золотника')
        self.axL.grid(True)
        self.axL.legend(loc='upper right', fontsize=8)

        # --- График 3: массы (газовая часть — запрошена пользователем) ---
        self.axM.plot(time, M_gas, 'g-', linewidth=1.5, label='Масса газовой части')
        self.axM.plot(time, M_liq, 'c--', linewidth=1.2, label='Масса жидкой части')
        self.axM.plot(time, M_tot, 'k:', linewidth=1.2, label='Полная масса')
        self.axM.set_ylabel('Масса (кг)')
        self.axM.set_title('Изменение массы газовой части (и полной массы системы)')
        self.axM.grid(True)
        self.axM.legend(loc='best', fontsize=8)

        # --- График 4: расход через клапан ---
        self.axF.plot(time, flow, 'm-', linewidth=1.5, label='Расход G(t)')
        self.axF.set_ylabel('Расход (кг/с)')
        self.axF.set_xlabel('Время (с)')
        self.axF.set_title('Массовый расход через клапан')
        self.axF.grid(True)
        self.axF.legend(loc='upper right', fontsize=8)

        self.fig.tight_layout()
        self.canvas.draw_idle()

    def on_save_png(self):
        if not hasattr(self, 'last_res'):
            QMessageBox.information(self, "Нет данных", "Сначала выполните расчёт.")
            return
        path = os.path.join(os.getcwd(), "prv_results.png")
        self.fig.savefig(path, dpi=150)
        self.statusBar().showMessage(f"Графики сохранены: {path}")


# -----------------------------------------------------------------------------
def main():
    app = QApplication(sys.argv)
    app.setFont(QFont("DejaVu Sans", 10))
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
