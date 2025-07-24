# -*- coding: utf-8 -*-
"""
Created on Sun Jul 20 15:57:38 2025

@author: lorenzopol
"""

import cv2
import numpy as np
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt

# -------- PARAMETERS ----------
cap = cv2.VideoCapture(r'C:\Users\loren\.spyder-py3\2025_2P_CLass\reconstructed_activity.mp4')
ret, frame = cap.read()

height, width = frame.shape[:2]

# -------- BLOB DETECTOR SETTINGS ----------
params = cv2.SimpleBlobDetector_Params()
params.filterByColor = True
params.blobColor = 255  # white (bright) spots

params.filterByArea = True
params.minArea = 3      # decrease to pick up tiny neurons
params.maxArea = 2000   # increase if large bright blobs exist

params.filterByCircularity = True
params.minCircularity = 0.4 
 
params.filterByConvexity = True
params.minConvexity = 0.5

params.filterByInertia = True
params.minInertiaRatio = 0.1

params.minThreshold = 30 # detect faint neurons
params.maxThreshold = 255
params.thresholdStep = 3
detector = cv2.SimpleBlobDetector_create(params)

tracks = []  # list of neuron trajectories: each track is a list of (x, y) positions
max_distance = 5  # maximum distance to consider same neuron between frames

# -------- PROCESS FRAMES ----------
while ret:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    keypoints = detector.detect(gray)
    centroids = np.array([[kp.pt[0], kp.pt[1]] for kp in keypoints])  # (x, y)
    
    if len(tracks) == 0:
        # Initialize tracks with first frame detections
        for c in centroids:
            tracks.append([c])
    else:
        last_positions = np.array([t[-1] for t in tracks])
        if len(centroids) > 0 and len(last_positions) > 0:
            dists = cdist(last_positions, centroids)  # compute "all-vs-all" distance. opt?
            for i, t in enumerate(tracks):
                j = np.argmin(dists[i])
                if dists[i, j] < max_distance:
                    t.append(centroids[j])  # it was an already found neuron that has moved, store its new location
                else:
                    t.append(t[-1])  # it is a new neuron, store its positon as its first location
    for kp in keypoints:
        x, y = int(kp.pt[0]), int(kp.pt[1])
        cv2.circle(frame, (x, y), 4, (0, 255, 0), 1)
    
    cv2.imshow("Neuron Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    ret, frame = cap.read()

cap.release()
cv2.destroyAllWindows()

# -------- PLOT TRAJECTORIES ----------
plt.figure(figsize=(6,6))
for t in tracks:
    t = np.array(t)
    plt.plot(t[:,0], t[:,1], '-o', markersize=2)
plt.gca().invert_yaxis()
plt.title("Neuron Trajectories")
plt.xlabel("X (pixels)")
plt.ylabel("Y (pixels)")
plt.show()


n_tracks = np.array([np.array(t) for t in tracks])  # shape: (n_neurons, n_frames, 2)

# Compute mean displacement across all neurons at each frame
displacements = np.diff(n_tracks, axis=1)  # (n_neurons, n_frames-1, 2)
mean_displacement = displacements.mean(axis=0)    # (n_frames-1, 2)

import matplotlib.pyplot as plt
plt.figure()
plt.plot(mean_displacement[:,0], label="X global motion")
plt.plot(mean_displacement[:,1], label="Y global motion")
plt.legend()
plt.title("Estimated Brain Drift from Neuron Tracks")
plt.show()
