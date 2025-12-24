# app.py

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import os
import numpy as np

# === ИМПОРТ ВАШИХ ФУНКЦИЙ ===
from encode_full_dataset import main as encode_dataset
from generate_missing_blocks import generate_all_missing_versions
from fill_missing import fill_all_datasets
from calculate_stats import calculate_stats
from calculate_errors import calculate_errors_for_all
from isodata import run_isodata_4vars

# === ГЛОБАЛЫ ===
original_df = None
encoded_df = None
current_missing_df = None
current_filled_df = None
isodata_df = None

# === СТИЛИЗОВАННЫЙ TREEVIEW ДЛЯ ТАБЛИЦ ===
def style_treeview(tree, df):
    """Очищает и заполняет Treeview с горизонтальным скроллом."""
    for item in tree.get_children():
        tree.delete(item)
    tree["columns"] = list(df.columns)
    tree["show"] = "headings"
    for col in df.columns:
        tree.heading(col, text=col)
        tree.column(col, width=min(120, max(80, len(str(col)) * 10)), anchor="center")
    for _, row in df.head(30).iterrows():
        tree.insert("", "end", values=list(row))

# === ОСНОВНОЙ КЛАСС ===
class MedicalDataApp:
    def __init__(self, root):
        self.root = root
        self.root.title(" Анализ медицинских данных")
        self.root.geometry("1100x700")
        self.root.configure(bg="#f5f7fa")

        # Стиль
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TNotebook", background="#f5f7fa")
        style.configure("TNotebook.Tab", padding=[12, 6], font=("Segoe UI", 10, "bold"))
        style.map("TNotebook.Tab", background=[("selected", "#4a6fa5")], foreground=[("selected", "white")])

        # Вкладки
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill="both", expand=True, padx=10, pady=10)

        # Создаём вкладки
        self.tabs = {}
        for name, title in [
            ("load", "1. Загрузка"),
            ("encode", "2. Оцифровка"),
            ("missing", "3. Пропуски"),
            ("fill", "4. Восстановление"),
            ("stats", "5. Статистика"),
            ("cluster", "6. Кластеризация")
        ]:
            frame = ttk.Frame(self.notebook)
            self.notebook.add(frame, text=title)
            self.tabs[name] = frame

        self.setup_tabs()

    def setup_tabs(self):
        self.create_tab_load()
        self.create_tab_encode()
        self.create_tab_missing()
        self.create_tab_fill()
        self.create_tab_stats()
        self.create_tab_cluster()

    # === ВКЛАДКА 1: ЗАГРУЗКА ===
    def create_tab_load(self):
        frame = ttk.Frame(self.tabs["load"])
        frame.pack(fill="both", expand=True, padx=15, pady=15)

        btn = ttk.Button(frame, text=" Выбрать Excel-файл", command=self.load_file)
        btn.pack(pady=10)

        self.tree_load = ttk.Treeview(frame)
        vsb = ttk.Scrollbar(frame, orient="vertical", command=self.tree_load.yview)
        hsb = ttk.Scrollbar(frame, orient="horizontal", command=self.tree_load.xview)
        self.tree_load.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)

        self.tree_load.pack(side="left", fill="both", expand=True)
        vsb.pack(side="right", fill="y")
        hsb.pack(side="bottom", fill="x")

    def load_file(self):
        global original_df
        file_path = filedialog.askopenfilename(filetypes=[("Excel files", "*.xlsx")])
        if not file_path:
            return
        try:
            original_df = pd.read_excel(file_path)
            style_treeview(self.tree_load, original_df)
            messagebox.showinfo(" Успех", "Файл загружен!")
        except Exception as e:
            messagebox.showerror(" Ошибка", f"Не удалось загрузить:\n{e}")

    # === ВКЛАДКА 2: ОЦИФРОВКА ===
    def create_tab_encode(self):
        frame = ttk.Frame(self.tabs["encode"])
        frame.pack(fill="both", expand=True, padx=15, pady=15)

        btn = ttk.Button(frame, text=" Оцифровать данные", command=self.encode_data)
        btn.pack(pady=10)

        self.tree_encode = ttk.Treeview(frame)
        vsb = ttk.Scrollbar(frame, orient="vertical", command=self.tree_encode.yview)
        hsb = ttk.Scrollbar(frame, orient="horizontal", command=self.tree_encode.xview)
        self.tree_encode.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)

        self.tree_encode.pack(side="left", fill="both", expand=True)
        vsb.pack(side="right", fill="y")
        hsb.pack(side="bottom", fill="x")

    def encode_data(self):
        global original_df, encoded_df
        if original_df is None:
            messagebox.showwarning(" Внимание", "Сначала загрузите файл!")
            return
        try:
            os.makedirs('working_data', exist_ok=True)
            original_df.to_csv('working_data/original_full.csv', index=False, encoding='utf-8-sig', sep=';')
            encode_dataset()
            encoded_df = pd.read_csv('working_data/encoded_full.csv', sep=';', encoding='utf-8-sig')
            style_treeview(self.tree_encode, encoded_df)
            messagebox.showinfo(" Успех", "Данные оцифрованы!")
        except Exception as e:
            messagebox.showerror(" Ошибка", f"Оцифровка не удалась:\n{e}")

    # === ВКЛАДКА 3: ПРОПУСКИ ===
    def create_tab_missing(self):
        frame = ttk.Frame(self.tabs["missing"])
        frame.pack(fill="both", expand=True, padx=15, pady=15)

        ttk.Label(frame, text="Выберите процент пропусков:", font=("Segoe UI", 10)).pack()
        self.missing_var = tk.StringVar(value="5")
        for p in ["5", "10", "20", "30"]:
            ttk.Radiobutton(frame, text=f"{p}%", variable=self.missing_var, value=p).pack()

        btn = ttk.Button(frame, text=" Сгенерировать пропуски", command=self.generate_missing)
        btn.pack(pady=10)

        self.tree_missing = ttk.Treeview(frame)
        vsb = ttk.Scrollbar(frame, orient="vertical", command=self.tree_missing.yview)
        hsb = ttk.Scrollbar(frame, orient="horizontal", command=self.tree_missing.xview)
        self.tree_missing.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)

        self.tree_missing.pack(side="left", fill="both", expand=True)
        vsb.pack(side="right", fill="y")
        hsb.pack(side="bottom", fill="x")

    def generate_missing(self):
        global encoded_df, current_missing_df
        if encoded_df is None:
            messagebox.showwarning(" Внимание", "Сначала оцифровка!")
            return
        try:
            p = int(self.missing_var.get()) / 100
            os.makedirs('missing_datasets', exist_ok=True)
            generate_all_missing_versions(
                original_path='working_data/encoded_full.csv',
                output_dir='missing_datasets',
                percents=[p],
                random_seed=42
            )
            current_missing_df = pd.read_csv(f'missing_datasets/missing_{int(p*100)}pct.csv', sep=';', encoding='utf-8-sig')
            style_treeview(self.tree_missing, current_missing_df)
            messagebox.showinfo(" Успех", f"Пропуски {int(p*100)}% готовы!")
        except Exception as e:
            messagebox.showerror(" Ошибка", f"Не удалось сгенерировать пропуски:\n{e}")

    # === ВКЛАДКА 4: ВОССТАНОВЛЕНИЕ ===
    def create_tab_fill(self):
        frame = ttk.Frame(self.tabs["fill"])
        frame.pack(fill="both", expand=True, padx=15, pady=15)

        ttk.Label(frame, text="Выберите метод восстановления:", font=("Segoe UI", 10)).pack()
        self.fill_var = tk.StringVar(value="median")
        ttk.Radiobutton(frame, text=" Медиана", variable=self.fill_var, value="median").pack()
        ttk.Radiobutton(frame, text=" Регрессия", variable=self.fill_var, value="regression").pack()

        btn = ttk.Button(frame, text=" Восстановить", command=self.fill_missing)
        btn.pack(pady=10)

        self.tree_fill = ttk.Treeview(frame)
        vsb = ttk.Scrollbar(frame, orient="vertical", command=self.tree_fill.yview)
        hsb = ttk.Scrollbar(frame, orient="horizontal", command=self.tree_fill.xview)
        self.tree_fill.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)

        self.tree_fill.pack(side="left", fill="both", expand=True)
        vsb.pack(side="right", fill="y")
        hsb.pack(side="bottom", fill="x")

    def fill_missing(self):
        global current_missing_df, current_filled_df
        if current_missing_df is None:
            messagebox.showwarning(" Внимание", "Сначала сгенерируйте пропуски!")
            return
        try:
            p = int(self.missing_var.get()) if hasattr(self, 'missing_var') else 5
            current_missing_df.to_csv(f'missing_datasets/missing_{p}pct.csv', sep=';', encoding='utf-8-sig', index=False)
            fill_all_datasets()
            method = self.fill_var.get()
            current_filled_df = pd.read_csv(f'filled_datasets/filled_{method}_{p}pct.csv', sep=';', encoding='utf-8-sig')
            style_treeview(self.tree_fill, current_filled_df)
            messagebox.showinfo(" Успех", f"Восстановление методом '{method}' завершено!")
        except Exception as e:
            messagebox.showerror(" Ошибка", f"Не удалось восстановить:\n{e}")

    # === ВКЛАДКА 5: СТАТИСТИКА ===
    def create_tab_stats(self):
        frame = ttk.Frame(self.tabs["stats"])
        frame.pack(fill="both", expand=True, padx=15, pady=15)

        btn = ttk.Button(frame, text=" Рассчитать всю статистику", command=self.calculate_all_stats)
        btn.pack(pady=10)

        self.text_stats = tk.Text(frame, wrap="word", bg="#f9fbfd", font=("Consolas", 10))
        scroll = ttk.Scrollbar(frame, orient="vertical", command=self.text_stats.yview)
        self.text_stats.configure(yscrollcommand=scroll.set)

        self.text_stats.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")

    def calculate_all_stats(self):
        try:
            # Генерим всё, если ещё не
            if not os.path.exists('missing_datasets/missing_5pct.csv'):
                generate_all_missing_versions(
                    original_path='working_data/encoded_full.csv',
                    output_dir='missing_datasets',
                    percents=[0.05, 0.10, 0.20, 0.30],
                    random_seed=42
                )
            fill_all_datasets()
            from main import save_stats_for_missing_datasets, save_stats_for_filled_datasets
            save_stats_for_missing_datasets()
            save_stats_for_filled_datasets()
            calculate_errors_for_all()

            result = (
                " Статистика рассчитана:\n"
                " original_stats.json\n"
                " missing_stats/\n"
                " filled_stats/\n"
                " error_analysis/error_summary.csv\n\n"
                "Откройте папки, чтобы увидеть JSON-файлы и CSV."
            )
            self.text_stats.delete(1.0, tk.END)
            self.text_stats.insert(tk.END, result)
            messagebox.showinfo(" Успех", "Статистика готова!")
        except Exception as e:
            messagebox.showerror(" Ошибка", f"Ошибка в статистике:\n{e}")

    # === ВКЛАДКА 6: КЛАСТЕРИЗАЦИЯ ===
    def create_tab_cluster(self):
        frame = ttk.Frame(self.tabs["cluster"])
        frame.pack(fill="both", expand=True, padx=15, pady=15)

        btn = ttk.Button(frame, text=" Запустить кластеризацию ISODATA", command=self.run_clustering)
        btn.pack(pady=10)

        self.tree_cluster = ttk.Treeview(frame)
        vsb = ttk.Scrollbar(frame, orient="vertical", command=self.tree_cluster.yview)
        hsb = ttk.Scrollbar(frame, orient="horizontal", command=self.tree_cluster.xview)
        self.tree_cluster.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)

        self.tree_cluster.pack(side="left", fill="both", expand=True)
        vsb.pack(side="right", fill="y")
        hsb.pack(side="bottom", fill="x")

    def run_clustering(self):
        global isodata_df
        try:
            run_isodata_4vars('working_data/encoded_full.csv')
            isodata_df = pd.read_csv('clustered_datasets/isodata_clustered.csv', sep=';', encoding='utf-8-sig')

            # Добавляем колонку с распределением
            summary = isodata_df.groupby('cluster').size().reset_index(name='patients')
            msg = "\n=== Распределение кластеров ===\n"
            for _, row in summary.iterrows():
                msg += f"Кластер {row['cluster']}: {row['patients']} пациентов\n"

            style_treeview(self.tree_cluster, isodata_df)
            messagebox.showinfo(" Успех", "Кластеризация завершена!")

            # Доп. окно с описанием
            desc = tk.Toplevel(self.root)
            desc.title(" Описание кластеров")
            desc.geometry("600x500")
            text = tk.Text(desc, wrap="word", font=("Segoe UI", 10))
            text.pack(fill="both", expand=True, padx=10, pady=10)

            clusters_info = """
Кластер 0: Онкология, неврология (9000-18000 руб)
Кластер 1: Лёгкие симптомы - ОАК, терапевт (600-800 руб)
Кластер 2: Стандартный пакет - фиксированная цена (1400 руб)
Кластер 3: Контрольный осмотр - повторный приём
Кластер 4: Сложные случаи - индивидуальные обследования (до 20500 руб)
Кластер 5: Простые обращения к узким специалистам
            """
            text.insert(tk.END, clusters_info)
            text.configure(state="disabled")

        except Exception as e:
            messagebox.showerror(" Ошибка", f"Ошибка в кластеризации:\n{e}")


# === ЗАПУСК ===
if __name__ == "__main__":
    root = tk.Tk()
    app = MedicalDataApp(root)
    root.mainloop()