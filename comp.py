import numpy as np
import math
from scipy.signal import filtfilt, butter
import scipy.signal as signal

# Specify the sampling frequency and cutoff frequency

YAW_DRIFT = 1/(5900*3)

fs = 30  # Sampling frequency (Hz)
cutoff_hz = 1/30  # Cutoff frequency (Hz)
cutoff_hz2 = 2.5  # Cutoff frequency (Hz)

# Convert the cutoff frequency to a normalized frequency
nyquist_hz = fs / 2.0  # Nyquist frequency
normalized_cutoff_freq = cutoff_hz / nyquist_hz
normalized_cutoff_freq2 = cutoff_hz2 / nyquist_hz

# Create the filter coefficients

SAMPLE_WINDOW = 90*3
INITIAL_ANGLE = (0,0,0)

def complimentaryFilter(acc,gyro,t=None,dt=1/30,alpha=0.9, highpass=0.5, lowpass=0.05):
    global SAMPLE_WINDOW
    global INITIAL_ANGLE


    #gyro = filtfilt(*butter(5, highpass, btype='high'), gyro, axis=0) #high pass
    b, a = signal.butter(5, normalized_cutoff_freq, btype='high')
    b2, a2 = signal.butter(5, normalized_cutoff_freq2, btype='low')

    # Apply the filter to the signal
    #gyro = signal.filtfilt(b, a, gyro, axis=0)
    acc = signal.filtfilt(b2, a2, acc, axis=0)
    NUM = 0
    initial_angle = (0,0,0)
    if len(acc) > SAMPLE_WINDOW:
        NUM = len(acc)-SAMPLE_WINDOW
        initial_angle = INITIAL_ANGLE
        t = t[NUM:len(acc)]
    g_pitch, g_roll, g_yaw = initial_angle
    a_pitch, a_roll, a_yaw = initial_angle
    dt = np.diff(t).mean()
    orientation = np.zeros_like(acc[:SAMPLE_WINDOW])
    gyro_out = np.zeros_like(acc[:SAMPLE_WINDOW])
    acc_out = np.zeros_like(acc[:SAMPLE_WINDOW])

    #print(NUM, len(acc))
    for i in range(NUM, len(acc)):
        g_pitch += (gyro[i][0]*dt)# - (6.033e-4)/30
        g_roll += (gyro[i][1]*dt)
        g_yaw += (gyro[i][2]*dt) + YAW_DRIFT

        a_pitch = math.atan2(acc[i][1],acc[i][2])
        a_roll = math.atan2(-acc[i][0],math.sqrt(acc[i][1]*acc[i][1] + acc[i][2]*acc[i][2]))

        pitch = g_pitch*alpha + a_pitch*(1-alpha)
        roll = g_roll*alpha + a_roll*(1-alpha)
        yaw = g_yaw
        if i==NUM:
            INITIAL_ANGLE = (pitch, roll, yaw)
        orientation[i-NUM] = np.array([pitch, roll, yaw])
        gyro_out[i-NUM] = np.array([g_pitch, g_roll, g_yaw])
        acc_out[i-NUM] = np.array([a_pitch, a_roll, 0])
    return orientation, gyro_out, acc_out
