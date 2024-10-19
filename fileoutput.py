import pandas as pd
import numpy as np

from preprocessing import get_patients,get_events_and_epochs,get_lame_list
from powerbandgeneration import get_eeg_psd

columns=list(range(19))

def reverse(data):
    print(data)


lise=[]

patients=get_patients()
lamelist=get_lame_list()
lamelist.append(32)
lamelist.append(40)
sorted(lamelist)
count=0
for i in range(88):
    if i in lamelist:
        continue
    print(f"------{i}-----------")
    event,epoch=get_events_and_epochs(raw=patients[i])

    data=get_eeg_psd(epoch)
    temp=data[4][0].reshape(19)
    lise.append(temp)
        


print(f"{len(lise)}---{len(lise[0])}")
df=pd.DataFrame(data=lise,columns=columns)
print(df.head())
df.to_csv('temp.csv')


