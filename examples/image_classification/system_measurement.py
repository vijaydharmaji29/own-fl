import psutil
import time
import pandas as pd

class SystemMeasurement:
    def __init__(self, name):
        self.cpu_utilisation = []
        self.ram_utilisation = []
        self.name = name

    def start_measurement(self):
        while True:
            cpu_percent = psutil.cpu_percent(interval=1)
            self.cpu_utilisation.append(cpu_percent)
            ram_percent = psutil.virtual_memory().percent
            self.ram_utilisation.append(ram_percent)
            time.sleep(1)

    def write_analysis(self):
        df_cpu = pd.DataFrame({'CPU Utilisation:': self.cpu_utilisation})
        df_ram = pd.DataFrame({'RAM Utilisation:': self.ram_utilisation})

        df_cpu.to_csv("./test_files/cpu_utilisation_" + self.name + ".csv")
        df_ram.to_csv("./test_files/ram_utilisation_" + self.name + ".csv")

        print("SYSTEM ANALYSIS WRITTEN")