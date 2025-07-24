import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from sklearn import linear_model
import pandas as pd
from scipy import stats, signal
import os
import glob

# ABBREVIATIONS:
#   APP: appunti 
#   nofX = number of X

# ======================== PATH DEF ========================
exp_type = "doubledot"  # choose among looming, doubledot and gratings
fish_folder = fr'C:\Users\loren\.spyder-py3\2025_2P_CLass\{exp_type}\suite2p_output\suite2p\plane0'
regressor_file = glob.glob(os.path.join(fish_folder, '..', '..', '..', '*.csv'))[0]

# APP: suite2p output [from https://github.com/MouseLand/suite2p?tab=readme-ov-file#outputs]
    # F.npy: array of fluorescence traces (ROIs by timepoints)
    # Fneu.npy: array of neuropil fluorescence traces (ROIs by timepoints)
    # spks.npy: array of deconvolved traces (ROIs by timepoints)
    # stat.npy: array of statistics computed for each cell (ROIs by 1)
    # ops.npy: options and intermediate outputs
    # iscell.npy: specifies whether an ROI is a cell, first column is 0/1, and second column is probability that the ROI is a cell based on the default classifier
# APP: neuroni sulle righe, valori di fluorescenza dei 7200 timestamp sulle colonne
# APP(TL;DR): F.npy is an array of fluorescence trace with shape nofROI x timestamps
F = np.load(os.path.join(fish_folder, 'F.npy'))
stat = np.load(os.path.join(fish_folder, 'stat.npy'), allow_pickle=True)


# APP: csv that stores:
# - type of stimuli given to the fish 
# - frametime (both in seconds and in frames) on which the stimuli is given
df = pd.read_csv(regressor_file, delimiter=',').to_numpy()
df = df[1:-1:2, :]
# ======================== END PATH DEF ========================

# ======================== ZSCORE ========================
Fz=stats.zscore(F, axis=1)  # calculating the z-score of the fluorescence of each neuron
sos=signal.butter(5, (0.02,10), 'band', fs=30, output='sos')
Fz = signal.sosfiltfilt(sos, Fz, )

fig1=plt.figure()
plt.imshow(F, interpolation=None, aspect='auto' ,vmin=-0.5)
plt.xlabel('frames')
plt.ylabel('neurons')
plt.colorbar()
plt.title('rasterplot_Fraw')

fig2=plt.figure()
plt.imshow(Fz, interpolation=None, aspect='auto', vmin=0, vmax=4)
plt.xlabel('frames')
plt.ylabel('neurons')
plt.colorbar()
plt.title('rasterplot_Fzscored')

# plt.plot(np.mean(Fz, axis = 0)) 
# plt.plot(reg)
# plt.show()
# ======================== END ZSCORE ========================


nROIs=np.shape(Fz)[0]  # nof neurons (in realtà rappresentano il numero di ROI segmentate con suite2p, alcuni di questi hanno una probabilità di essere una cellula <20%)
frames=np.shape(Fz)[1]  # total nof frames
framerate = 30  #30fps
    
print('ROI',nROIs)
print('Frames', frames)

# ======================== CALCIUM IMPULSE RESPONSE FUNCTION (CIRF) MODELING ========================
time = np.linspace(0,frames-1,frames)/framerate

MIN_toff = 10
MAX_toff = 120  # in seconds, tau of the exp decay
toff_container = {}
for to in range(MIN_toff, MAX_toff, 1):
    to /= 10
    print(to)
    t=np.linspace(0,20,int(20*framerate))
    t=np.hstack((-np.ones(int(20*framerate)), t))
    e=np.exp(-t/to)*(1+np.sign(t))/2
    e=e/np.max(e)
    # plt.figure()
    # plt.plot(e)
    # plt.title(f"Calcium impulse response function [toff={to}s]")
    # plt.xlabel("frames")
    # plt.ylabel("Magnitude")
    # ======================== END CIRF MODELING ========================
    
    # ======================== REGRESSORS ========================
    frame_stim_events = df[:,2]   
    number_stim_events= np.shape(frame_stim_events)[0]  # per double dot è 12 dato che ci sono 12 coppie (accensione-spegnimento dello stimolo). 6 di queste sono per l'emicampo rostrale (quelle che popoleranno coeffs_forward), le altre per l'emicampo caudale (coeffs_backward)
    print("Numero totale di eventi di stimolo:", number_stim_events)
    stim_duration_frames= 12  # frame_stim_events[1]-frame_stim_events[0]
    
    reg = np.zeros((frames,number_stim_events))
    reg2 = np.zeros((frames,number_stim_events))
    for i in range(np.int16(number_stim_events)):
        # APP: set to 1 when a stimuli is given to the fish, 0 otherwise
        reg[np.int16(frame_stim_events[i]):np.int16(frame_stim_events[i]+stim_duration_frames),i]=1
        reg2[np.int16(frame_stim_events[i]):np.int16(frame_stim_events[i]+stim_duration_frames),i]=1
        
        # convolve the kernel over the "is-stimulus-given" function
        reg[:,i]=np.convolve(reg[:,i], e, mode='same')
        reg[:,i]=reg[:,i]/np.max(reg[:,i])
    
    # APP: add one last series to the regressors that models an always-active cell
    reg=np.hstack((reg,np.ones((frames,1))))
    
    # plt.figure()
    # plt.imshow(Fz, interpolation=None, aspect='auto',vmin=0, vmax=4)
    # plt.xlabel('frames')
    # plt.ylabel('neurons')
    # plt.plot(512-512*reg,linewidth=0.8)
    # plt.colorbar()
    # ======================== END REGRESSORS ========================
    
    # ======================== LINEAR MODEL FIT ========================
    win=3
    
    datan=Fz.copy()
    clf = linear_model.LinearRegression(fit_intercept=False)
    coeffs=np.zeros((nROIs,np.int16(number_stim_events+1)))   
    Tscores=np.zeros((nROIs,np.int16(number_stim_events+1)))    
    for k in range(nROIs):
        datan[k,:]=np.convolve(np.ones(win),datan[k,:],mode='same')/win 
        coeffs[k,:]=clf.fit(reg,datan[k,:]).coef_
        error=np.sqrt(
                (
                    np.dot(
                        (datan[k,:]-np.inner(coeffs[k,:],reg)),
                        (datan[k,:]-np.inner(coeffs[k,:],reg))
                        )
                )
            )
        Tscores[k,:]=np.divide(coeffs[k,:],error)
        #datan[:,j,k]=sc.stats.mstats.zscore(datan[:,j,k])#median filter              
        #datan[:,j,k]=(datan[:,j,k]-np.average(data[noise_bon:noise_boff,j,k]))/np.average(data[noise_bon:noise_boff,j,k])
    toff_container[to] = Tscores
to_toff_supermean = [np.mean(np.mean(matrix, axis=1)) for to, matrix in toff_container.items()]
mi = np.argmax(to_toff_supermean)
max_val = to_toff_supermean[mi]

# Plot the array
plt.figure(figsize=(8, 4))
plt.plot([i/10 for i in range(MIN_toff, MAX_toff, 1)], to_toff_supermean, color='blue', linewidth=2, label='to_toff_supermean')
max_idx = [i/10 for i in range(MIN_toff, MAX_toff, 1)][mi]
# Highlight maximum point
plt.scatter(max_idx, max_val, color='red', s=80, zorder=5, label=f"Max: {max_val:.2f}")
plt.annotate(f"Max = {max_idx:.2f}",
             (max_idx, max_val),
             textcoords="offset points", xytext=(10, 10),
             arrowprops=dict(arrowstyle="->", color='red'),
             color='red')

plt.title("to_toff_supermean (Maximum Highlighted)")
plt.xlabel("Index")
plt.ylabel("Value")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()



print("Numero di colonne in coeffs:", coeffs.shape[1])

# APP: for each neuron, get the median of the coeff fitted to the stimuli
coeffs_forward  = coeffs[:, ::2]
coeffs_backward = coeffs[:, 1::2]
median_coeffs_forward  = np.nanmedian(coeffs_forward, axis = 1)  # median over the fowards stimuli
median_coeffs_backward = np.nanmedian(coeffs_backward, axis = 1)  # median over the backwards stimuli

Tscores_forward = Tscores[:, ::2]
Tscores_backward = Tscores[:, 1::2]

median_Tscores_forward = np.nanmedian(Tscores_forward, axis = 1)
median_Tscores_backward = np.nanmedian(Tscores_backward, axis = 1)
# ======================== END LINEAR MODEL FIT ========================


pxy=np.zeros((nROIs,2))
for i in range(nROIs):
    pxy[i,:]=stat[i]['med']  # from the stat.npy file, get the center of each ROI (more or less the center position of a neuron)

# coeff plot #1: shows one from the coeffs_forward and one from the coeffs_backward
plt.figure()
plt.subplot(2,1,1)
plt.scatter(pxy[:,1],512-pxy[:,0],c=coeffs[:,2],alpha=0.5,cmap='bwr', norm=colors.CenteredNorm())
plt.colorbar()
plt.subplot(2,1,2)
plt.scatter(pxy[:,1],512-pxy[:,0],c=coeffs[:,1],alpha=0.5,cmap='bwr', norm=colors.CenteredNorm())
plt.colorbar()

# coeff plot #2: median of the coeffs_forward and coeffs_backward
plt.figure()
plt.subplot(2,1,1)
plt.title("Rostral stimuli")
plt.scatter(pxy[:,1],512-pxy[:,0],c=median_coeffs_forward,alpha=0.5,cmap='bwr',  vmin=-.5, vmax=.5)
plt.colorbar()
plt.subplot(2,1,2)
plt.title("Caudal stimuli")
plt.scatter(pxy[:,1],512-pxy[:,0],c=median_coeffs_backward,alpha=0.5,cmap='bwr',  vmin=-.5, vmax=.5)
plt.colorbar()

# tscore plot #1
plt.figure()
plt.subplot(2,1,1),plt.scatter(pxy[:,1],512-pxy[:,0],c=Tscores[:,2],alpha=0.5,cmap='bwr', norm=colors.CenteredNorm())
plt.colorbar()
plt.subplot(2,1,2),plt.scatter(pxy[:,1],512-pxy[:,0],c=Tscores[:,1],alpha=0.5,cmap='bwr', norm=colors.CenteredNorm())
plt.colorbar()

# tscore plot #2
plt.figure()
plt.subplot(2,1,1)
plt.title("Rostral stimuli")
plt.scatter(pxy[:,1],512-pxy[:,0],c=median_Tscores_forward,alpha=0.5,cmap='bwr', norm=colors.CenteredNorm())
plt.colorbar()
plt.subplot(2,1,2)
plt.title("Caudal stimuli")
plt.scatter(pxy[:,1],512-pxy[:,0],c=median_Tscores_backward,alpha=0.5,cmap='bwr', norm=colors.CenteredNorm())
plt.colorbar()


# shows the difference in fit coeffs between the Tscores_forward and Tscores_backward
fig4 = plt.figure()
plt.scatter(pxy[:,1],512-pxy[:,0],
            c=np.clip(Tscores_backward.mean(axis=1), 0, np.inf) - np.clip(Tscores_forward.mean(axis=1),0, np.inf),
            alpha=0.5,cmap='bwr', norm=colors.CenteredNorm())
plt.colorbar()
plt.title(f"Relative sensitivity during {exp_type}")
plt.xlabel("Caudal -> Rostral")
plt.ylabel("Dx -> Sx")

# nothing new
plt.figure()
plt.subplot(2,1,1)
plt.imshow(datan, interpolation=None, aspect='auto')
plt.xlabel('frames')
plt.ylabel('neurons')
plt.subplot(2,1,2)
plt.plot(reg,linewidth=0.8)
plt.plot(reg2,linewidth=0.8)
plt.colorbar()


# APP: facendo così si estrae quello con il valore assoluto migliore, non con la miglior media
# max_tscore = np.where(Tscores==np.max(Tscores))[0][0]

max_tscore = np.argmax(np.max(Tscores, axis=1))
plt.figure()
plt.title("Max t-score")
plt.plot(Fz[max_tscore ,:])
plt.plot(np.inner(coeffs[max_tscore ,:],reg))

median_highest = np.argmax(np.median(Tscores, axis=1))
plt.figure()
plt.title("Max median t-score")
plt.plot(Fz[median_highest ,:])
plt.plot(np.inner(coeffs[median_highest ,:],reg))
plt.show()

