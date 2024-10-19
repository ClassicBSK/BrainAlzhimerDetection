import pandas as pd
from preprocessing import get_patients,get_events_and_epochs,get_lame_list

df=pd.read_csv('participants.tsv',delimiter='\t')

# print(df["Group"].unique())
 
df["Group"]=df["Group"].map({'A':0,'C':1,'F':2})



lamelist=get_lame_list()
lamelist.append(32)
lamelist.append(40)
sorted(lamelist)

lise=[]


for i in range(88):
    if i in lamelist:
        continue
    lise.append(df["Group"][i])

df1=pd.DataFrame(data=lise,columns=[1])
df1.to_csv('class.csv')