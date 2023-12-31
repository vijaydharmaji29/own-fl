import psutil
import time
import pandas as pd
import numpy as np

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

    def write_analysis(self, folder_name):
        df_cpu = pd.DataFrame({'CPU Utilisation:': self.cpu_utilisation})
        df_ram = pd.DataFrame({'RAM Utilisation:': self.ram_utilisation})

        df_cpu.to_csv(folder_name + "cpu_utilisation_" + self.name + ".csv")
        df_ram.to_csv(folder_name + "ram_utilisation_" + self.name + ".csv")

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

    def getSystemScore(self, DataStorage):

        n = len(DataStorage.round_time)

        if(n == 0):
            return 0, True
        elif(n >= 1 and n < 10):
            return DataStorage.round_time[-1], False
        else:
            return self.predict_round_time(DataStorage), False
        

    def getDefaultSystemScore(self, DataStorage):
        n = len(DataStorage.round_time)
        
        if(n == 0):
            return 0, True
        else:
            return DataStorage.round_time[-1], False
            


    def predict_round_time(self, DataStorage):
        if DataStorage.participation_list[-1]: #i.e did participate in prev round
            lrp = self.linear_regression_prediction(DataStorage)
            round_time = (lrp + DataStorage.round_time[-1])/2
            return round_time
        else:
            lrp = self.linear_regression_prediction(DataStorage)
            return lrp
        
    def linear_regression_prediction(self, DataStorage):
        req_cpu_times = []
        req_ram_times = []
        reg_round_times = np.array(DataStorage.round_time)

        for i in range(len(DataStorage.participation_list)):
            if DataStorage.participation_list[i]:
                req_cpu_times.append(self.cpu_average_utilisation[i])
                req_ram_times.append(self.ram_average_utilisation[i])

        req_cpu_times = np.array(req_cpu_times)
        req_ram_times = np.array(req_ram_times)

        X = np.column_stack((np.ones_like(req_cpu_times), req_cpu_times, req_ram_times))

        # Use numpy's linear algebra solver to find the coefficients (beta)
        beta = np.linalg.lstsq(X, reg_round_times, rcond=None)[0]

        system_score = self.make_prediction(beta)

        return system_score

    def make_prediction(self, beta):
        beta0, beta1, beta2 = beta
        pred = beta0 + beta1*self.cpu_utilisation[-1][-1] + beta2*self.ram_utilisation[-1][-1] #maybe make this the average

        return pred

