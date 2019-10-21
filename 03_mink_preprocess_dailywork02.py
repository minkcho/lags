
# coding: utf-8

# In[1]:

# preprocessing : convert user event logs to integer sequences

import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
data_dir = '/export/data/cert-data/processed/4.2/tmp_user/'
to_dir   = '/export/data/cert-data/processed/4.2/user/'

for i in range(0,1000,1):
    file_name = 'f_u{:03}.csv'.format(i)
    # user work파일 로드
    user = pd.read_csv(data_dir + file_name, sep=',')

    # 2010-01-01 부터 2011-05-30 날짜를 담은 리스트
    rng = pd.date_range('2010-01-01', '2011-05-30')
    total = pd.DataFrame()

    # conversion to date type & sort by date
    user['date'] = pd.to_datetime(user['date'])
    user = user.sort_values(by='date')

    #하루치 행위만을 따로 추출
    for date in rng:
        temp = user.loc[((date <= user['date']) & (user['date'] < date + 1))]
        temp = pd.DataFrame(temp)

        if not temp.empty:
            length = len(temp)

            #print(date,"is not empty")
            #각 행위를 편의상 숫자로 표현
            temp.loc[temp.activity == 'Logon',      'activity'] = 1
            temp.loc[temp.activity == 'http',       'activity'] = 2
            temp.loc[temp.activity == 'email',      'activity'] = 3
            temp.loc[temp.activity == 'file',       'activity'] = 4
            temp.loc[temp.activity == 'Connect',    'activity'] = 5
            temp.loc[temp.activity == 'Disconnect', 'activity'] = 6
            temp.loc[temp.activity == 'Logoff',     'activity'] = 7

            #하루의 작업 시퀀스를 나타내기 위한 처리 날짜, 시퀀스 길이, 시퀀스 순으로 정렬
            #user, date, activity => activity rows => (transpose) => [activity list] + [date, length] 
            dayact = pd.DataFrame(data = temp['activity'].values, index= list(range(0,len(temp))))
            dayact = np.transpose(dayact)
            dayact['date'] = date
            dayact['length'] = length
            cols = dayact.columns.tolist()
            cols = cols[-2:] + cols[:-2]    # move to (date, length) to the front 
            dayact = dayact[cols]

            #print(dayact)
            #시퀀스를 붙여나감
            total = pd.concat([total, dayact])

    total.to_csv(to_dir + 'daily_f_u{:03}.csv'.format(i), sep=',', float_format='%.f')
    print("Done. user {}".format(i))

