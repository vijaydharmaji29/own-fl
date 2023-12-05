import pandas as pd
import os

class DataStorage:
    def __init__(self):
        self.round_start_times = []
        self.round_end_times = []

        self.round_1st_recieved = []
        self.round_last_received = []

    def write_analysis(self):
        print("Writing analysis")

        print(type(self.round_end_times[-1]))

        # self.round_start_times = [x.timestamp() for x in self.round_start_times]
        # self.round_end_times = [x.timestamp() for x in self.round_end_times]
        # self.round_1st_recieved = [x.timestamp() for x in self.round_1st_recieved]
        # self.round_last_received = [x.timestamp() for x in self.round_last_received]

        df1 = pd.DataFrame({"Round Start Times": self.round_start_times, "Round End Times":self.round_end_times})
        df2 = pd.DataFrame({"Round First Recieved":self.round_1st_recieved})
        df3 = pd.DataFrame({"Round Last Recieved":self.round_last_received})

        

        try:
            os.mkdir('./aggregator_test_files/')
        except:
            print("Directory already exists")

        df1.to_csv('./aggregator_test_files/round_times.csv')
        df2.to_csv('./aggregator_test_files/round_first_recieved_times.csv')
        df3.to_csv('./aggregator_test_files/round_last_recieved_times.csv')

        print("Done writing analysis")
        
        
