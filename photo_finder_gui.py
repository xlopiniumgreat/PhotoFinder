"""PhotoFinder desktop UI. Start with python photo_finder_gui.py."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
import traceback
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import photo_finder as core
from gui_jobs import prepare_bundled_models, run_job, validate


class PhotoPicker(ttk.LabelFrame):
    def __init__(self, parent, title, hint, changed):
        super().__init__(parent, text=title, padding=10)
        self.paths = []
        self.images = []
        self.changed = changed
        self.columnconfigure(0, weight=1)
        self.info = ttk.Label(self, text=hint)
        self.info.grid(row=0, column=0, sticky='w')
        self.choose = ttk.Button(self, text='Выбрать фото…', command=self.select)
        self.choose.grid(row=0, column=1, padx=(12, 0))
        self.clear = ttk.Button(self, text='Сбросить', command=self.reset)
        self.clear.grid(row=0, column=2, padx=(6, 0))
        self.preview = ttk.Frame(self)
        self.preview.grid(row=1, column=0, columnspan=3, sticky='w', pady=(7, 0))
        self.hint = hint

    def select(self):
        paths = filedialog.askopenfilenames(title='Выберите фотографии', filetypes=[
            ('Фотографии', '*.jpg *.jpeg *.png *.webp *.heic *.heif'), ('Все файлы', '*.*')])
        if paths:
            self.set_paths(paths)

    def set_paths(self, paths):
        self.paths = list(dict.fromkeys(str(Path(p).resolve()) for p in paths))
        self.info.configure(text=f'Выбрано: {len(self.paths)}. {self.hint}')
        self.images.clear()
        for widget in self.preview.winfo_children():
            widget.destroy()
        # Only a few thumbnails; no recognition on the Tk thread.
        from PIL import Image, ImageOps, ImageTk
        from pillow_heif import register_heif_opener
        register_heif_opener()
        for file in self.paths[:6]:
            frame = ttk.Frame(self.preview)
            frame.pack(side='left', padx=(0, 10))
            try:
                with Image.open(core.disk_path(file)) as image:
                    thumb = ImageOps.exif_transpose(image).convert('RGB')
                    thumb.thumbnail((64, 48))
                    rendered = ImageTk.PhotoImage(thumb)
                self.images.append(rendered)
                ttk.Label(frame, image=rendered).pack()
            except Exception:
                ttk.Label(frame, text='Нет превью').pack()
            name = Path(file).name
            ttk.Label(frame, text=name[:15] + ('…' if len(name) > 15 else ''), font=('Segoe UI', 8)).pack()
        if len(self.paths) > 6:
            ttk.Label(self.preview, text=f'+ {len(self.paths) - 6} фото').pack(side='left')
        self.changed()

    def reset(self):
        self.set_paths([])

    def enable(self, enabled):
        for button in (self.choose, self.clear):
            button.configure(state='normal' if enabled else 'disabled')


class App:
    def __init__(self, root, data_dir):
        self.root = root
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.events = queue.Queue()
        self.stop = threading.Event()
        self.busy = False
        self.gpu_ok = False
        self.output_dir = None
        self.worker = None
        self.device = tk.StringVar(value='cpu')
        self.photos = tk.StringVar()
        self.output = tk.StringVar(value=str(Path.home() / 'Pictures' / 'PhotoFinder Results'))
        self.low = tk.StringVar(value='0.38')
        self.high = tk.StringVar(value='0.50')
        self.copy = tk.BooleanVar(value=True)
        self.status = tk.StringVar(value='Выберите фотографии. Исходные файлы останутся нетронутыми.')
        self.sample_status = tk.StringVar(value='Добавьте минимум 3 фото с человеком и 3 без него. Используйте отдельные снимки, не эталоны.')
        self.counts = tk.StringVar(value='Найдено: 0    Проверить: 0    Нет совпадений: 0    Ошибки: 0')
        root.title('PhotoFinder — поиск ваших фотографий')
        root.geometry('1000x880')
        root.minsize(800, 650)
        style = ttk.Style()
        if 'vista' in style.theme_names():
            style.theme_use('vista')
        style.configure('TLabel', font=('Segoe UI', 10))
        style.configure('TButton', font=('Segoe UI', 10), padding=5)
        style.configure('TLabelframe.Label', font=('Segoe UI', 10, 'bold'))
        # Scrollable body also supports smaller laptop displays.
        canvas = tk.Canvas(root, highlightthickness=0)
        bar = ttk.Scrollbar(root, orient='vertical', command=canvas.yview)
        canvas.configure(yscrollcommand=bar.set)
        bar.pack(side='right', fill='y')
        canvas.pack(fill='both', expand=True)
        body = ttk.Frame(canvas, padding=20)
        item = canvas.create_window((0, 0), window=body, anchor='nw')
        canvas.bind('<Configure>', lambda e: canvas.itemconfigure(item, width=e.width))
        body.bind('<Configure>', lambda e: canvas.configure(scrollregion=canvas.bbox('all')))
        root.bind('<MouseWheel>', lambda e: canvas.yview_scroll(-int(e.delta / 120), 'units'))
        body.columnconfigure(0, weight=1)
        ttk.Label(body, text='PhotoFinder', font=('Segoe UI', 23, 'bold')).grid(row=0, column=0, sticky='w')
        ttk.Label(body, text='Найдите фотографии с нужным человеком. Всё обрабатывается на этом компьютере.').grid(row=1, column=0, sticky='w', pady=(0, 12))
        self.pickers = {}
        for row, (key, title, hint) in enumerate([
            ('references', '1. Кого ищем — эталонные фотографии', 'Рекомендуем 15–30 сольных фото.'),
            ('positive', '2. Проверка — человек есть на фото', 'Минимум 3 отдельных снимка.'),
            ('negative', '3. Проверка — человека нет на фото', 'Минимум 3 отдельных снимка.')], 2):
            picker = PhotoPicker(body, title, hint, self.inputs_changed)
            picker.grid(row=row, column=0, sticky='ew', pady=4)
            self.pickers[key] = picker
        folders = ttk.LabelFrame(body, text='4. Папки', padding=10)
        folders.grid(row=5, column=0, sticky='ew', pady=6)
        folders.columnconfigure(1, weight=1)
        self.inputs = []
        for row, (label, var) in enumerate([('Где искать', self.photos), ('Куда сохранить', self.output)]):
            ttk.Label(folders, text=label).grid(row=row, column=0, sticky='w', padx=(0, 10))
            entry = ttk.Entry(folders, textvariable=var)
            entry.grid(row=row, column=1, sticky='ew', pady=3)
            button = ttk.Button(folders, text='Выбрать…', command=lambda v=var: self.select_dir(v))
            button.grid(row=row, column=2, padx=(8, 0))
            self.inputs.extend([entry, button])
        devices = ttk.LabelFrame(body, text='5. Устройство', padding=10)
        devices.grid(row=6, column=0, sticky='ew', pady=6)
        self.cpu = ttk.Radiobutton(devices, text='CPU · процессор', value='cpu', variable=self.device)
        self.cpu.grid(row=0, column=0, sticky='w')
        self.gpu = ttk.Radiobutton(devices, text='GPU · NVIDIA', value='cuda', variable=self.device, state='disabled')
        self.gpu.grid(row=0, column=1, padx=20)
        self.probe_button = ttk.Button(devices, text='Проверить GPU', command=self.probe_gpu)
        self.probe_button.grid(row=0, column=2)
        self.gpu_label = ttk.Label(devices, text='CPU выбран по умолчанию. Проверка GPU не меняет выбор.', wraplength=850)
        self.gpu_label.grid(row=1, column=0, columnspan=3, sticky='w', pady=(8, 0))
        self.advanced_button = ttk.Button(body, text='Дополнительные настройки ▸', command=self.toggle_advanced)
        self.advanced_button.grid(row=7, column=0, sticky='w', pady=4)
        self.advanced = ttk.Frame(body)
        ttk.Label(self.advanced, text='Нижний порог').pack(side='left')
        self.low_entry = ttk.Entry(self.advanced, textvariable=self.low, width=7)
        self.low_entry.pack(side='left', padx=8)
        ttk.Label(self.advanced, text='Верхний порог').pack(side='left')
        self.high_entry = ttk.Entry(self.advanced, textvariable=self.high, width=7)
        self.high_entry.pack(side='left', padx=8)
        self.copy_check = ttk.Checkbutton(self.advanced, text='Копировать фотографии в результат', variable=self.copy)
        self.copy_check.pack(side='left', padx=12)
        self.inputs.extend([self.low_entry, self.high_entry, self.copy_check])
        ttk.Label(body, textvariable=self.sample_status, wraplength=900).grid(row=9, column=0, sticky='w', pady=8)
        actions = ttk.Frame(body)
        actions.grid(row=10, column=0, sticky='ew', pady=6)
        self.check_button = ttk.Button(actions, text='Проверить примеры', command=lambda: self.start('check'))
        self.check_button.pack(side='left')
        self.start_button = ttk.Button(actions, text='Начать поиск', command=lambda: self.start('scan'))
        self.start_button.pack(side='left', padx=8)
        self.stop_button = ttk.Button(actions, text='Остановить', command=self.cancel, state='disabled')
        self.stop_button.pack(side='left')
        self.open_button = ttk.Button(actions, text='Открыть результаты', command=self.open_output, state='disabled')
        self.open_button.pack(side='right')
        self.progress = ttk.Progressbar(body, maximum=100)
        self.progress.grid(row=11, column=0, sticky='ew', pady=8)
        ttk.Label(body, textvariable=self.status, wraplength=900).grid(row=12, column=0, sticky='w')
        ttk.Label(body, textvariable=self.counts).grid(row=13, column=0, sticky='w', pady=8)
        ttk.Label(body, text='Для первого запуска модели нужен интернет. Проверка примеров не меняет пороги автоматически.', wraplength=900, foreground='#555555').grid(row=14, column=0, sticky='w')
        root.protocol('WM_DELETE_WINDOW', self.close)
        root.after(100, self.poll)

    def inputs_changed(self):
        self.sample_status.set('Примеры будут проверены с текущими эталонами и порогами перед поиском.')

    def select_dir(self, var):
        directory = filedialog.askdirectory(title='Выберите папку')
        if directory:
            var.set(directory)

    def toggle_advanced(self):
        if self.advanced.winfo_ismapped():
            self.advanced.grid_remove()
        else:
            self.advanced.grid(row=8, column=0, sticky='ew', pady=8)

    def config(self, device=None):
        return dict(core.DEFAULTS, t_low=float(self.low.get().replace(',', '.')),
                    t_high=float(self.high.get().replace(',', '.')),
                    device=device or self.device.get(), model_dir=str(self.data_dir / 'models'),
                    output_mode='copy' if self.copy.get() else 'report')

    def set_busy(self, value):
        self.busy = value
        for picker in self.pickers.values():
            picker.enable(not value)
        for widget in self.inputs + [self.cpu, self.check_button, self.start_button, self.probe_button]:
            widget.configure(state='disabled' if value else 'normal')
        self.gpu.configure(state='normal' if self.gpu_ok and not value else 'disabled')
        self.stop_button.configure(state='normal' if value else 'disabled')
        if not value:
            self.progress.stop()
            self.progress.configure(mode='determinate')

    def emit(self, kind, value=None):
        self.events.put((kind, value))

    def start(self, mode):
        try:
            if not self.output.get().strip():
                raise ValueError('Выберите папку результатов.')
            if mode == 'scan' and not self.photos.get().strip():
                raise ValueError('Выберите папку для поиска.')
            job = {key: list(p.paths) for key, p in self.pickers.items()}
            job.update(mode=mode, config=self.config(), photos=self.photos.get(), output=self.output.get())
            validate(job)
        except (ValueError, OSError) as error:
            messagebox.showerror('Проверьте выбор', str(error))
            return
        self.stop.clear()
        self.output_dir = None
        self.open_button.configure(state='disabled')
        self.counts.set('Найдено: 0    Проверить: 0    Нет совпадений: 0    Ошибки: 0')
        self.set_busy(True)
        def worker():
            try:
                run_job(job, self.emit, self.stop)
                self.emit('done', mode)
            except core.Cancelled:
                self.emit('cancelled')
            except Exception as error:
                traceback.print_exc()
                self.emit('error', str(error))
        self.worker = threading.Thread(target=worker, daemon=True)
        self.worker.start()

    def probe_gpu(self):
        self.set_busy(True)
        self.stop.clear()
        self.gpu_ok = False
        self.gpu_label.configure(text='Проверяем NVIDIA, библиотеки CUDA и выполнение моделей…')
        config = dict(core.DEFAULTS, device='cuda', model_dir=str(self.data_dir / 'models'))
        def worker():
            try:
                import onnxruntime as ort
                if 'CUDAExecutionProvider' not in ort.get_available_providers():
                    raise RuntimeError('Эта сборка содержит только CPU. Нужна GPU-сборка PhotoFinder.')
                try:
                    result = subprocess.run(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],
                        capture_output=True, text=True, timeout=10,
                        creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
                    name = result.stdout.strip().splitlines()[0] if result.returncode == 0 and result.stdout.strip() else 'NVIDIA GPU'
                except (OSError, subprocess.TimeoutExpired):
                    name = 'NVIDIA GPU'
                core.check_cancel(self.stop)
                prepare_bundled_models(config)
                core.Engine(config)  # Warm-up includes both real models; does not analyze user images.
                core.check_cancel(self.stop)
                self.emit('gpu_result', (True, name + ' — проверка пройдена. Можно выбрать GPU.'))
            except core.Cancelled:
                self.emit('gpu_result', (False, 'Проверка отменена. Доступен CPU.'))
            except Exception as error:
                traceback.print_exc()
                self.emit('gpu_result', (False, 'GPU недоступен: ' + str(error)[:280]))
        self.worker = threading.Thread(target=worker, daemon=True)
        self.worker.start()

    def cancel(self):
        self.stop.set()
        self.status.set('Остановка после текущего файла или загрузки модели…')
        self.stop_button.configure(state='disabled')

    def poll(self):
        try:
            while True:
                kind, value = self.events.get_nowait()
                if kind == 'stage':
                    self.status.set(value)
                    self.progress.configure(mode='indeterminate')
                    self.progress.start(15)
                elif kind in ('progress', 'scan_progress'):
                    self.progress.stop()
                    self.progress.configure(mode='determinate')
                    n, total, detail = value
                    self.progress['value'] = 100 * n / max(1, total)
                    if kind == 'scan_progress':
                        self.status.set(f'Обработано {n} из {total} · {"NVIDIA GPU" if self.device.get() == "cuda" else "CPU"}')
                        self.show_counts(detail)
                    else:
                        self.status.set(detail)
                elif kind == 'output':
                    self.output_dir = value
                    self.open_button.configure(state='normal')
                elif kind == 'reference_count':
                    self.sample_status.set(f'Принято эталонов: {value[0]} из {value[1]}. Причины пропуска — references.json.')
                elif kind == 'sample_result':
                    p, n = value['positive'], value['negative']
                    self.sample_status.set(f'С человеком: найдено {p.get("me", 0)}, сомнительно {p.get("uncertain", 0)}, пропущено {p.get("not_me", 0)}. '
                        f'Без человека: нет совпадений {n.get("not_me", 0)}, сомнительно {n.get("uncertain", 0)}, ложных совпадений {n.get("me", 0)}. Технических ошибок: {value["errors"]}.')
                elif kind == 'summary':
                    self.show_counts(value)
                elif kind == 'sample_warning':
                    self.sample_status.set(self.sample_status.get() + '\n' + value)
                elif kind == 'gpu_result':
                    self.gpu_ok = value[0]
                    self.gpu_label.configure(text=value[1])
                    if not self.gpu_ok:
                        self.device.set('cpu')
                    self.set_busy(False)
                elif kind in ('done', 'cancelled', 'error'):
                    self.set_busy(False)
                    if kind == 'done':
                        self.status.set('Проверка примеров завершена. Нажмите «Начать поиск» для обработки основной папки.' if value == 'check' else
                                        'Поиск завершён. Откройте результаты; сомнительные фотографии нужно проверить вручную.')
                    elif kind == 'cancelled':
                        self.status.set('Остановлено. Результаты уже обработанных файлов сохранены.')
                    elif kind == 'error':
                        self.status.set('Не удалось завершить обработку. Подробности — в журнале.')
                        messagebox.showerror('Ошибка', value)
        except queue.Empty:
            pass
        self.root.after(100, self.poll)

    def show_counts(self, counts):
        self.counts.set(f'Найдено: {counts.get("me", 0)}    Проверить: {counts.get("uncertain", 0)}    '
                       f'Нет совпадений: {counts.get("not_me", 0)}    Ошибки: {counts.get("errors", 0)}')

    def open_output(self):
        if self.output_dir:
            try:
                os.startfile(self.output_dir)
            except OSError as error:
                messagebox.showerror('Не удалось открыть папку', str(error))

    def close(self):
        if self.busy:
            self.cancel()
            messagebox.showinfo('Завершаем обработку', 'Дождитесь остановки текущей операции, затем закройте окно ещё раз.')
            return
        self.root.destroy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=Path, default=Path(os.environ.get('LOCALAPPDATA', Path.home())) / 'PhotoFinder')
    parser.add_argument('--smoke-test', action='store_true')
    parser.add_argument('--verify-runtime', choices=['cpu', 'cuda'], help='Run model and HEIF self-test without opening a window')
    args = parser.parse_args()
    args.data_dir.mkdir(parents=True, exist_ok=True)
    # Windowed executables have no stdout/stderr; some ML dependencies assume they do.
    log = open(args.data_dir / 'application.log', 'a', encoding='utf-8', buffering=1)
    sys.stdout = log
    sys.stderr = log
    if args.verify_runtime:
        import numpy as np
        from PIL import Image
        from pillow_heif import register_heif_opener
        config = dict(core.DEFAULTS, device=args.verify_runtime, model_dir=str(args.data_dir / 'models'))
        try:
            prepare_bundled_models(config)
            engine = core.Engine(config)
            for name, model in engine.app.models.items():
                size = config['det_size'] if name == 'detection' else 112
                result = model.session.run(None, {model.session.get_inputs()[0].name: np.zeros((1, 3, size, size), np.float32)})
                if not all(np.isfinite(array).all() for array in result):
                    raise RuntimeError('Non-finite model output')
            register_heif_opener()
            heic = args.data_dir / 'self-test.heic'
            Image.new('RGB', (32, 24), (100, 150, 200)).save(heic, format='HEIF')
            assert core.load_image(heic).shape == (24, 32, 3)
            core.write_json(args.data_dir / 'verification.json', {'ok': True, 'providers': engine.providers, 'heif': True})
            return
        except Exception as error:
            traceback.print_exc()
            core.write_json(args.data_dir / 'verification.json', {'ok': False, 'error': str(error)})
            raise SystemExit(1)
    root = tk.Tk()
    app = App(root, args.data_dir)
    if args.smoke_test:
        assert app.device.get() == 'cpu'
        assert str(app.gpu['state']) == 'disabled'
        root.after(500, root.destroy)
    root.mainloop()


if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    main()
