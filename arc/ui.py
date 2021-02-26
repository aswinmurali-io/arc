#!/usr/bin/env python3
# The GUI for the image reconstruction model.

import sys
import threading
import tkinter as tk
import tkinter.ttk as ttk

import psutil


class ArcUI(tk.Frame):
    @staticmethod
    def cpu_percent(val: str) -> str:
        return f'CPU {val} %'

    @staticmethod
    def cpu_cores(val: str) -> str:
        return f'CPU Cores {val}'
    
    @staticmethod
    def ram_usage(val: str) -> str:
        return f'RAM Usage {val} %'

    def __init__(self, parent, *args, **kwargs) -> None:
        tk.Frame.__init__(self, parent, *args, **kwargs)
        self.parent = parent

        self.cpu_percent = tk.StringVar(value=ArcUI.cpu_percent('...'))
        self.cpu_percent_label = tk.Label(self.parent, textvariable=self.cpu_percent)
        self.cpu_percent_label.pack()

        self.cpu_core_count = tk.StringVar(value=ArcUI.cpu_cores('...'))
        self.cpu_core_count_label = tk.Label(self.parent, textvariable=self.cpu_core_count)
        self.cpu_core_count_label.pack()

        self.ram_count = tk.StringVar(value=ArcUI.ram_usage('...'))
        self.ram_count_label = tk.Label(self.parent, textvariable=self.ram_count)
        self.ram_count_label.pack()

        self.train_button = ttk.Button(self.parent, text="Train", command=None)
        self.train_button.pack()

        # Launch the shared thread to get the hardware stat
        shared_thread = threading.Thread(target=self.shared_thread)
        shared_thread.daemon = True
        shared_thread.start()

    def shared_thread(self) -> None:
        """This function is used to make the logic execution async in nature.
        """
        self.refresh_hwstat()
        self.cpu_core_count.set(ArcUI.cpu_cores(psutil.cpu_count()))

    def refresh_hwstat(self) -> None:
        """This function is used to periodically get the hardware infos.
        """
        self.cpu_percent.set(ArcUI.cpu_percent(psutil.cpu_percent()))
        self.ram_count.set(ArcUI.ram_usage((psutil.virtual_memory().percent)))

        thread = threading.Timer(1, self.refresh_hwstat)
        thread.daemon = True
        thread.start()


if __name__ == "__main__":
    root = tk.Tk()
    root.title('ARC Reconstruction Model')

    # DPI Awareness
    if sys.platform == 'win32':
        from ctypes import windll
        windll.shcore.SetProcessDpiAwareness(1)

    ArcUI(root).pack(side=tk.TOP, fill=tk.BOTH, expand=True)
    root.mainloop()
