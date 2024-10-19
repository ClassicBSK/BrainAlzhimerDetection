from preprocessing import get_patients,get_events_and_epochs,get_lame_list
import numpy as np
import mne

freq_range={
    "delta":[0.5,4],
    "theta":[4,8],
    "alpha":[8,13],
    "beta":[13,25],
    "gamma":[25,45]
}

def get_eeg_psd(epochs:mne.Epochs):
    spectrum=epochs.compute_psd(method='welch',fmin=0.5,fmax=45)
    psds,freqs=spectrum.get_data(return_freqs=True)

    psds/=np.sum(psds,axis=-1,keepdims=True)

    stu=[]
    
    for i in freq_range:
        fmin=freq_range[i][0]
        fmax=freq_range[i][1]
        psd_band=psds[:,:,(freqs>=fmin)&(freqs<fmax)].mean(axis=-1)
        stu.append(psd_band)
    return stu

# patients=get_patients()
# lamelist=get_lame_list()
# event,epoch=get_events_and_epochs(raw=patients[3])

# get_eeg_psd(epoch)
