import matplotlib
import mne
import numpy as np
from scipy.signal import welch
import pandas as pd

path="Dataset\\derivatives\\extracted\\sub-002_task-eyesclosed_eeg.set"
patients=[]

for i  in range(1,10):
    
    path="Dataset\\derivatives\\extracted\\sub-00"+str(i)+"_task-eyesclosed_eeg.set"
    data=mne.io.read_raw_eeglab(path,preload=True)
    data.resample(100)
    raw=data.get_data()
    patients.append(data)

for i  in range(10,89):

    path="Dataset\\derivatives\\extracted\\sub-0"+str(i)+"_task-eyesclosed_eeg.set"
    data=mne.io.read_raw_eeglab(path,preload=True)
    data.resample(100)
    raw=data.get_data()
    patients.append(data)



#patients=get_patients()
mini=len(patients[0][0])
for i in range(len(patients)):
    mini=min(len(patients[i][0]),mini)
    #print(len(patients[i][0]))

    
raw=mne.io.read_raw_eeglab(path)

data=raw.get_data()

#timepoints=data.__len__()
#print(raw)

print(mini)

#print(events)

#description=data.describe()
#print(timepoints)



#temp=welch(patients[0][0],fs=500)
#print(temp)
#print(len(temp[1]))

annotations_path="Dataset\\derivatives\\extracted\\annotations.tsv"

annots=pd.read_csv(annotations_path,delimiter="\t")
annots=annots['name']
annotations=annots.to_numpy()
#print(annots)

raw=mne.io.read_raw_eeglab(path)


channels=raw.info.ch_names
#print(raw.info.ch_names)

tmax=30.0-1.0/raw.info['sfreq']

def get_patients():
    return patients
def get_lame_list():
    lamelist=[]

    for i in reversed(range(88)):
        try:
            temp=mne.events_from_annotations(patients[i])
            #print(i)
            if len(temp[0]) == 0:
                lamelist.append(i)
            #print(i)
        except:
            print(i+" not there")
    return lamelist

def get_events_and_epochs(raw):

    events=mne.events_from_annotations(raw)
    #print(temp[0])
    epochs=mne.Epochs(raw=raw,tmax=tmax)
    #print(epochs)
    return events,epochs


#print(get_epochs_events())
'''
lamelist=get_lame_list()
print(lamelist)
'''