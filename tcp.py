import socket
import comp
import json    # or `import simplejson as json` if on Python < 2.6
import numpy as np
import math
#import plotter
import threading
import time

HOST = '192.168.1.56'  # The server's IP address
PORT = 58769          # The server's port


x0,y0,z0 = 117,21,29
highpass=1
lowpass=0.05

accel_data = []
gyro_data = []
t_arr = []

sample_num = 90*3
draw_freq = 15
last_draw = 0

np.set_printoptions(precision=3)

# Create a socket object
ANGLES = np.array([0,0,0])
def tcp_listen():
    global ANGLES, t_arr
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        # Connect to the server
        s.connect((HOST, PORT))
        print(f"Connected to {HOST}:{PORT}")
        
        while True:
            # Receive data from the server
            data = s.recv(1024)
            if not data:
                break
            # Print the received data to the console
            #print(f"Received data: {data.decode('utf-8')}")
            try:
                obj = json.loads(data)
            except:
                continue
            accel = float(obj["accelerometerAccelerationX"]), float(obj["accelerometerAccelerationY"]), float(obj["accelerometerAccelerationZ"])
            gyro = obj["gyroRotationX"], obj["gyroRotationY"], obj["gyroRotationZ"]
            t = obj["accelerometerTimestamp_sinceReboot"]
            t_arr.append(t)
            accel_data.append(accel)
            gyro_data.append(gyro)
            #print(accel)
            #acceleration = dict(json_data)
            if len(accel_data)>sample_num:
                #accel_data=accel_data[:sample_num]
                #gyro_data=gyro_data[:sample_num]
                orientation, gyro2, acc2 = comp.complimentaryFilter(np.array(accel_data, dtype=np.float32),np.array(gyro_data, dtype=np.float32),highpass=highpass,lowpass=lowpass, t=np.array(t_arr, dtype=np.float64))
                #for i in range(len(orientation)):
                #    x,y,z = orientation[i]
                #    orientation[i] = np.array([fix_angle(x), fix_angle(y), fix_angle(z)])
                #orientation= np.unwrap(orientation)
                last_value = orientation[-1]
                x,y,z = last_value
                #x,y,z = fix_angle(x), fix_angle(y), fix_angle(z)
                ANGLES = np.array([x,y,z])
                #print(x,y,z)
                accel_x, accel_y, accel_z = accel
                #print("ANGL %.5f \t %.5f \t %.5f" % (math.degrees(x)+180,math.degrees(y),math.degrees(z)))
                #print("ANGL %.5f \t %.5f \t %.5f" % (x,y,z))

                #print("ACEL %.5f \t %.5f \t %.5f" % (accel_x,accel_y,accel_z))
                #accel_data = []
                #gyro_data = []

threading.Thread(target=tcp_listen).start()

def fix_angle(angle):
    #while(angle >= 3.14): #angle in radians
    #    angle -= 2*3.14
    while(angle < 0):
        angle += 2*3.14
    return angle

import world_engine
engine = world_engine.Engine("smartWatchCoordinatePlane.csv", coordinate_translation=(-1,-1,-1))
while True:
    time.sleep(1/24)
    engine.update(ANGLES)
    x,y,z = ANGLES
    print("ANGL %.5f \t %.5f \t %.5f" % (math.degrees(x)+180,math.degrees(y),math.degrees(z)))
    engine.raycast()