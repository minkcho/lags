import numpy as np
import pandas as pd

log = pd.read_csv('/home/minkcho/cert-data/r4.2/logon.csv', sep=',')

users = log['user'].unique()

no = 0
dic = pd.DataFrame(users, index=range(0,len(users)),columns=['user'])

dic.to_csv('/home/minkcho/src/InsiderThreat/dictionary.csv', sep=',')
