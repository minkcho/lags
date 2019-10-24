
# coding: utf-8

# In[21]:

# preprocessing CMU CERT r.42 dataset
#     : generate user based event log files from event-based files 

data_dir = './cert_data/r4.2/'
to_dir = './processed_data/r4.2/'


# In[22]:


# original : 01_make_user_dictionary
import numpy as np
import pandas as pd

log = pd.read_csv( data_dir + 'logon.csv', sep=',')
users = sorted(log['user'].unique())

no = 0
user_dict = pd.DataFrame(users, index=range(0,len(users)),columns=['user'])
user_dict.to_csv(to_dir + 'dictionary.csv', sep=',')

answer = pd.read_csv('./cert_data/answers/answer_r4.2_all_org.csv', sep=',')
answer['userid'] = answer['user'].apply(lambda x: user_dict.index[user_dict['user'] == x][0])
answer.to_csv(to_dir + 'answer_r4.2_all.csv',index=False)


# In[6]:


import numpy as np
import pandas as pd

filename = 'logon.csv'

dataset = pd.read_csv(data_dir + filename, sep=',')

df = pd.DataFrame(dataset)

df = df[['user', 'date', 'activity']]

df.to_csv(to_dir + 'pre01_' + filename, sep=',', index=False)


# In[11]:


import numpy as np
import pandas as pd

filename = 'http.csv'

dataset = pd.read_csv(data_dir + filename, sep=',', usecols=['user', 'date', 'pc'])

df = pd.DataFrame(dataset)

df = df[['user', 'date', 'pc']]
df.loc[:,'pc'] = "http"
df.columns = ['user', 'date', 'activity']        
        
df.to_csv(to_dir + 'pre01_' + filename, sep=',', index=False)


# In[7]:


import numpy as np
import pandas as pd

filename = 'email.csv'

dataset = pd.read_csv(data_dir + filename, sep=',', usecols=['user', 'date', 'pc'])

df = pd.DataFrame(dataset)

df = df[['user', 'date', 'pc']]
df.loc[:,'pc'] = "email"
df.columns = ['user', 'date', 'activity']        
        
df.to_csv(to_dir + 'pre01_' + filename, sep=',', index=False)


# In[8]:


import numpy as np
import pandas as pd

filename = 'file.csv'

dataset = pd.read_csv(data_dir + filename, sep=',', usecols=['user', 'date', 'pc'])

df = pd.DataFrame(dataset)

df = df[['user', 'date', 'pc']]
df.loc[:,'pc'] = "file"
df.columns = ['user', 'date', 'activity']        
        
df.to_csv(to_dir + 'pre01_' + filename, sep=',', index=False)


# In[9]:


import numpy as np
import pandas as pd

filename = 'device.csv'

dataset = pd.read_csv(data_dir + filename, sep=',', usecols=['user', 'date', 'activity'])

df = pd.DataFrame(dataset)

df = df[['user', 'date', 'activity']]
        
df.to_csv(to_dir + 'pre01_' + filename, sep=',', index=False)


# In[20]:


# run split into users : write to shell script (check peruser.sh)
import numpy as np
import pandas as pd
import os

log = pd.read_csv(to_dir + 'dictionary.csv', sep=',')
users = log['user']

print ('cd tmp_user')

idx = 0
for user in users:      
    filename = "f_u" + '{:03}'.format(idx) + ".csv"
    print('grep -h ' + user + ' ../pre01_*.csv > ' + filename)
    idx = idx + 1    
    
print ('sed -i \'1s/^/user,date,activity\\n/\' f_u*.csv')

# please cd $to_dir and run peruser.sh 
