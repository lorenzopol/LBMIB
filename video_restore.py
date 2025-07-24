# -*- coding: utf-8 -*-
"""
Created on Sun Jul 20 00:28:06 2025

@author: lorenzopol
"""
import numpy as np
import os
import cv2
exp_type = "doubledot"  # choose among looming, doubledot and gratings
fish_folder = fr'C:\Users\loren\.spyder-py3\2025_2P_CLass\{exp_type}\suite2p_output\suite2p\plane0'
F = np.load(os.path.join(fish_folder, 'F.npy'), allow_pickle=True)
stat = np.load(os.path.join(fish_folder, 'stat.npy'), allow_pickle=True)
iscell = np.load(os.path.join(fish_folder, 'iscell.npy'), allow_pickle=True)

cell_indices = np.where(iscell[:,0] == 1)[0]  # keep only real cells (even if all have been marked as real cells)
nframes = F.shape[1]
Ly = max([max(s['ypix']) for s in stat]) + 1
Lx = max([max(s['xpix']) for s in stat]) + 1


movie_recon = np.zeros((nframes, Ly, Lx), dtype=np.float32)

for j in cell_indices:
    trace = F[j, :]  # can be replaced with spks.npy 
    ypix, xpix, lam = stat[j]['ypix'], stat[j]['xpix'], stat[j]['lam']

    for x, y, l in zip(xpix, ypix, lam):
        movie_recon[:, y, x] += trace  * l  # remove l for the full fluorescence video

out = cv2.VideoWriter('reconstructed_activity.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (Lx, Ly), False)
for t in range(0, nframes, 5):  # one out of 5 frame. Full video is 7200 frames at 30fps (too long lol)
    frame = movie_recon[t]
    frame = (255 * (frame - frame.min()) / (np.ptp(frame) + 1e-9)).astype('uint8')
    out.write(frame)
out.release()

