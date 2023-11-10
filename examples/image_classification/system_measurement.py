import psutil
import time
import pandas as pd

class SystemMeasurement:
    def __init__(self, name):
        self.cpu_utilisation = [[]]
        self.cpu_average_utilisation = []
        self.ram_utilisation = [[]]
        self.ram_average_utilisation = []
        self.name = name

    def start_measurement(self):
        while True:
            cpu_percent = psutil.cpu_percent(interval=1)
            ram_percent = psutil.virtual_memory().percent
            
            if len(self.cpu_utilisation) != len(self.cpu_average_utilisation) or len(self.ram_utilisation) != len(self.ram_average_utilisation):
                self.cpu_utilisation[-1].append(cpu_percent)
                self.ram_utilisation[-1].append(ram_percent)
            
            time.sleep(1)

    def write_analysis(self):
        df_cpu = pd.DataFrame({'CPU Utilisation:': self.cpu_utilisation})
        df_ram = pd.DataFrame({'RAM Utilisation:': self.ram_utilisation})

        df_cpu.to_csv("./test_files/cpu_utilisation_" + self.name + ".csv")
        df_ram.to_csv("./test_files/ram_utilisation_" + self.name + ".csv")

        print("SYSTEM ANALYSIS WRITTEN")

    def start_round(self):
        self.cpu_utilisation.append([])
        self.ram_utilisation.append([])

    def end_round(self):
        cpu_av = 0
        for i in self.cpu_utilisation[-1]:
            cpu_av += i

        if len(self.cpu_utilisation[-1]) > 0:
            cpu_av = cpu_av/len(self.cpu_utilisation[-1])

        self.cpu_average_utilisation.append(cpu_av)

        ram_av = 0
        for i in self.ram_utilisation[-1]:
            ram_av += i
        
        if len(self.ram_utilisation[-1]) > 0:
            ram_av = ram_av/len(self.ram_utilisation[-1])

        self.ram_average_utilisation.append(ram_av)