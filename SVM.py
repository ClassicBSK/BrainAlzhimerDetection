import pandas as pd

sye=pd.read_csv('Dataset\\participants.tsv',sep='\t')
print(sye.Group.unique())