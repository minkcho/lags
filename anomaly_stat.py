import pandas as pd
import numpy as np

data_dir = '/export/data/cert-data/processed/4.2/'

class AnomalyStat():
    def __init__(self):
        self.user_dict = pd.read_csv(data_dir + 'dictionary.csv', sep=',')
        self.user_dict = self.user_dict['user']

        # this is for all cases
        self.answer = pd.read_csv('/home/minkcho/cert-data/answers/answer_r4.2_all.csv', sep=',')
        #self.answer = pd.read_csv('/home/minkcho/src/InsiderThreat/answer_r4.2-1_all.csv', sep=',')
        self.answer['date'] = pd.to_datetime(self.answer['date'])

        # this is for case 1 : abnomal case / use usb 
        self.answer1 = pd.read_csv('/home/minkcho/cert-data/answers/r4.2-1/answer_r4.2-1_all.csv', sep=',')
        self.answer1['date'] = pd.to_datetime(self.answer1['date'])
        # this is for case 2 : job website 
        self.answer2 = pd.read_csv('/home/minkcho/cert-data/answers/r4.2-2/answer_r4.2-2_all.csv', sep=',')
        self.answer2['date'] = pd.to_datetime(self.answer2['date'])
        # this is for case 3 : key logger 
        self.answer3 = pd.read_csv('/home/minkcho/cert-data/answers/r4.2-3/answer_r4.2-3_all.csv', sep=',')
        self.answer3['date'] = pd.to_datetime(self.answer3['date'])

        # it contains summary of all cases
        self.answer_period = pd.read_csv('/home/minkcho/cert-data/answers/answer_r4.2_between.csv', sep=',')
        self.answer_period['from_date'] = pd.to_datetime(self.answer_period['from_date'])
        self.answer_period['to_date'] = pd.to_datetime(self.answer_period['to_date'])
        
        self.threshold = 1.0

        # save anomalies
        self.anomalies_day = np.empty(shape=(1,5)) # np.array([['name', 'date', 'loss', 'ans_type', 'answer']])
        self.anomalies_between = np.empty(shape=(1,4)) #  np.array([['name', 'date', 'loss', 'answer']])
        self.anomalies_person = pd.read_csv('/home/minkcho/test/jang/insider/r4.2.ans_blank.csv', sep=',')

        self.total_anomalies_day = np.empty(shape=(1,5)) # np.array([['name', 'date', 'loss', 'answer', 'INFO']])
   
    def is_anomaly(self, date_x, n):
        user_n = self.user_dict[n]
        is_anomaly = not (self.answer[(self.answer['date']==date_x) & (self.answer['user']==user_n)]).empty
        anomaly_type = 0
        if( is_anomaly ):
            if(   not (self.answer1[(self.answer1['date']==date_x) & (self.answer1['user']==user_n)]).empty ):
                anomaly_type = 1 
            elif( not (self.answer2[(self.answer2['date']==date_x) & (self.answer2['user']==user_n)]).empty ):
                anomaly_type = 2 
            elif( not (self.answer3[(self.answer3['date']==date_x) & (self.answer3['user']==user_n)]).empty ):
                anomaly_type = 3
        return is_anomaly, anomaly_type

    def is_between(self, user, event_date):
        selected = self.answer_period[ self.answer_period['user'] == user]
        return (selected['from_date'] <= event_date).any() and (selected['to_date'] >= event_date).any()

    # if ploss > threshold then call save_anomaly()
    def save_anomaly(self, date_x, n, ploss, flag=True):
        self.anomaly_perday(date_x, self.user_dict[n], ploss, flag)
        if flag:
            self.anomaly_between(date_x, self.user_dict[n], ploss)

    
    def anomaly_perday(self, date_x, user_n, ploss, flag):
        # find in answer sheets
        correct_ans = not (self.answer[(self.answer['date']==date_x) & (self.answer['user']==user_n)]).empty
        ans_type = '0'
        if correct_ans:
            if not (self.answer1[(self.answer1['date']==date_x) & (self.answer1['user']==user_n)]).empty :
                ans_type = '1'
            elif not (self.answer3[(self.answer3['date']==date_x) & (self.answer3['user']==user_n)]).empty :
                ans_type = '3'
            else:
                ans_type = '2'

        if flag: 
            self.anomalies_day = np.append(self.anomalies_day, np.array([[user_n, date_x, ploss, ans_type, correct_ans]]), axis=0)
            # self.total_anomalies_day = np.append(self.total_anomalies_day, np.array([[user_n, date_x, ploss, correct_ans, 'ANOMAL']]), axis=0)
            if correct_ans:
                self.anomalies_person.loc[self.anomalies_person['user'] == user_n, 'answer'] = True
        #else:
            #self.total_anomalies_day = np.append(self.total_anomalies_day, np.array([[user_n, date_x, ploss, correct_ans, 'INFO']]), axis=0)
        
 
        return np.array([[user_n, date_x, ploss, correct_ans]])

    def anomaly_between(self, date_x, user_n, ploss):
        correct_ans_v2 = self.is_between(user_n, date_x)
        self.anomalies_between = np.append(self.anomalies_between, np.array([[user_n, date_x, ploss, correct_ans_v2]]), axis=0)
        if correct_ans_v2:
            self.anomalies_person.loc[self.anomalies_person['user'] == user_n, 'answer'] = True
        return np.array([[user_n, date_x, ploss, correct_ans_v2]])
    
    def print_stat(self):
        a = len(self.anomalies_day)
        b = np.sum( val == 'True' for val in self.anomalies_day[:,4] )
        c = a - b
        d = len(self.anomalies_between)
        e = np.sum( val == 'True' for val in self.anomalies_between[:,3]) 
        f = d - e
        print ('threshold : ', self.threshold)
        print ('[1]{} detected : true{}, false {}'.format(a, b, c))
        print ('[2]{} detected : true{}, false {}'.format(d, e, f))
        print ('[3]{} detected : true{}, false {}'.format( len(self.anomalies_person),  sum(self.anomalies_person['answer']),  len(self.anomalies_person) - sum(self.anomalies_person['answer'])))

    def print_all_stat(self):
        print ('ANS_Sheet DAY:\n')
        for i in range(len(self.anomalies_day)):
            print (self.anomalies_day[i])

        #print ('ANS_Sheet TOTAL DAY:\n', self.total_anomalies_day)
        print ('ANS_Sheet BETWEEN:\n')
        for i in range(len(self.anomalies_between)):
            print (self.anomalies_between[i])

        print ('ANS_Sheet PERSON:\n')
        for index, row in self.anomalies_person.iterrows():
            print (row['user'], ':', row['answer'])
       

    def checkfn_01(self, ploss_arr = [], factor=2.0):
        normal_std  = np.std(ploss_arr)
        normal_mean = np.mean(ploss_arr)
        self.threshold   = normal_mean + factor * normal_std
        count = np.sum(ploss_arr > self.threshold)
        return self.threshold, count

    def checkfn_02(self, ploss_arr = [], factor=2.0):
        normal_mean = np.mean(ploss_arr)
        selected_ploss = []
        for i in range(len(ploss_arr[0])):
            if ploss_arr[0][i] > normal_mean:
               selected_ploss.append(ploss_arr[0][i] - normal_mean)
        normal_std = np.sqrt( np.mean( np.power(selected_ploss, 2) ) )
        self.threshold   = normal_mean + factor * normal_std
        return self.threshold

    def checkfn_03(self, ploss_arr = [], factor=2.0):
        normal_mean = np.mean(ploss_arr)
        selected_ploss = []
        for i in range(len(ploss_arr[0])):
            if ploss_arr[0][i] > normal_mean:
               selected_ploss.append(ploss_arr[0][i] - normal_mean)
        normal_std = np.max( selected_ploss )
        self.threshold   = normal_mean + factor * normal_std
        return self.threshold

    def checkfn_04(self, ploss_arr = [], factor=2.0):
        normal_max  = np.max(ploss_arr)
        normal_std  = np.std(ploss_arr)
        normal_mean = np.mean(ploss_arr)
        self.threshold   = normal_max + factor * normal_std
        return self.threshold
             
