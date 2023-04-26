import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import plot, ion, show
import pandas as pd
from matplotlib.animation import FuncAnimation
import random
import time
import math

"""
apartment_coords = pd.read_csv("smartWatchCoordinatePlane.csv")
x0,y0,z0 = 117,21,29
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
# Define a 3D point cloud to transform
x = np.linspace(-10, 10, 11)
y = np.linspace(-10, 10, 11)
X, Y = np.meshgrid(x, y)
Z = np.zeros_like(X)

max_coord = -999999
for idx, row in apartment_coords.iterrows():
    ax.scatter(row["x"], row["y"], row["z"])
    max_coord = max(np.max(np.abs(np.array([row["x"], row["y"], row["z"]]))), max_coord)


ax.set_xlim(0, max_coord)
ax.set_ylim(0, max_coord)
ax.set_zlim(0, max_coord)
plt.gca().invert_xaxis()
"""


class Engine():
    def __init__(self, coordinate_file, coordinate_translation=(1,1,1)):
        self.coordinate_system = pd.read_csv(coordinate_file)
        self.coordinate_translation = coordinate_translation
        self.arm_length = 26
        self.start_position = (117,21,41)
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.delete = []
        # Define a 3D point cloud to transform
        x = np.linspace(-10, 10, 11)
        y = np.linspace(-10, 10, 11)
        X, Y = np.meshgrid(x, y)
        Z = np.zeros_like(X)
        self.X, self.Y, self.Z = X,Y,Z

        default = np.zeros((11,11))    
        self.surface = self.ax.plot_surface(default, default, default)

        self.objects = {}
        self.waypoints = {}
        max_coord = -999999
        for _, row in self.coordinate_system.iterrows():
            if row["Type"]=="IoT Device":
                self.objects[row["Object"]] = np.array([row["x"], row["y"], row["z"]])
            elif row["Type"]=="Waypoint":
                self.waypoints[row["Object"]] = np.array([row["x"], row["y"], row["z"]])
            self.ax.scatter(row["x"], row["y"], row["z"])
            max_coord = max(np.max(np.abs(np.array([row["x"], row["y"], row["z"]]))), max_coord)


        self.ax.set_xlim(0, max_coord)
        self.ax.set_ylim(0, max_coord)
        self.ax.set_zlim(0, max_coord)
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        plt.gca().invert_xaxis()
        plt.show(block=False)

    def setStartWaypoint(self, waypointName):
        self.start_position = self.waypoints[waypointName]

    


    def update(self, orientation):
        X = self.X
        Y = self.Y
        Z = self.Z
        translate_x, translate_y, translate_z = self.coordinate_translation
        x0,y0,z0 = self.start_position
        pitch, roll, yaw = translate_x*orientation[0],translate_y*orientation[1],translate_z*orientation[2]
        self.orientation = (pitch,roll,yaw)
        self.location = self.start_position #TODO update with arm model
        # Reshape the 3D point cloud into a 2D array
        P = np.vstack([X.ravel(), Y.ravel(), Z.ravel()])

        # Create a rotation matrix from the pitch, roll, and yaw angles for the current frame
        Rx = np.array([[1, 0, 0],
                    [0, np.cos(pitch), -np.sin(pitch)],
                    [0, np.sin(pitch), np.cos(pitch)]])
        Ry = np.array([[np.cos(roll), 0, np.sin(roll)],
                    [0, 1, 0],
                    [-np.sin(roll), 0, np.cos(roll)]])
        Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                    [np.sin(yaw), np.cos(yaw), 0],
                    [0, 0, 1]])
        R = np.dot(Rz, np.dot(Ry, Rx))

        # Transform the point cloud using the rotation matrix and subtract the translation offset
        P_rotated = np.dot(R, P)
        P_rotated = -P_rotated + np.array([[x0], [y0], [z0]])

        # Reshape the transformed point cloud back into a 3D array
        X_rotated = P_rotated[0, :].reshape(X.shape)
        Y_rotated = P_rotated[1, :].reshape(Y.shape)
        Z_rotated = P_rotated[2, :].reshape(Z.shape)

        # Clear the previous plot and plot the rotated point cloud for the current frame
        self.surface.remove()
        self.surface = self.ax.plot_surface(X_rotated, Y_rotated, Z_rotated)
        self.surface.set_facecolors('red')
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def raycast(self, max_length=240, resolution=1):
        pitch, _, yaw = self.orientation
        translate_x, translate_y, translate_z = self.coordinate_translation
        pitch*=translate_x
        #yaw*=translate_z
        x,y,z = self.location
        radius = np.linspace(1,max_length,num=(int(max_length/resolution)))
        #weights = np.linspace(1,max_length,num=(max_length//resolution))/len(radius)

        t = pitch + 3.14/2
        s = yaw + 3.14/2

        xOffset = self.arm_length*np.cos(s)*np.sin(t)
        yOffset = self.arm_length*np.sin(s)*np.sin(t)
        zOffset = self.arm_length*np.cos(t)

        rayX = radius*np.cos(s)*np.sin(t)
        rayY = radius*np.sin(s)*np.sin(t)
        rayZ = radius*np.cos(t)

        rayX+=(x+xOffset)
        rayY+=(y+yOffset)
        rayZ+=(z+zOffset)

        for obj in self.delete:
            obj.remove()
        self.delete = []
        point = self.ax.scatter(rayX, rayY, rayZ, c="red")
        self.delete.append(point)

        closest_point = None
        closest_val = 0
        distances = []
        for object in self.objects:
            dist = distance(self.objects[object], np.array([rayX, rayY, rayZ]))
            closest = np.min(dist)
            distances.append([object, closest])
            if closest_point is None or closest < closest_val:
                closest_val = closest
                closest_point = object

        for obj in distances:
            print(obj)
        print(closest_point)







        

    def get_target(self, max_distance=120):
        pass

def distance(p1,p2):
    x1,y1,z1 = p1
    x2,y2,z2 = p2
    dx,dy,dz = (x2-x1), (y2-y1), (z2-z1)
    return np.sqrt(dx*dx+dy*dy+dz*dz)

"""
def update(orientation):
    global SURFACE,X,Y,Z
    pitch, roll, yaw = -orientation[0],-orientation[1],-orientation[2]
    # Reshape the 3D point cloud into a 2D array
    P = np.vstack([X.ravel(), Y.ravel(), Z.ravel()])

    # Create a rotation matrix from the pitch, roll, and yaw angles for the current frame
    Rx = np.array([[1, 0, 0],
                [0, np.cos(pitch), -np.sin(pitch)],
                [0, np.sin(pitch), np.cos(pitch)]])
    Ry = np.array([[np.cos(roll), 0, np.sin(roll)],
                [0, 1, 0],
                [-np.sin(roll), 0, np.cos(roll)]])
    Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                [np.sin(yaw), np.cos(yaw), 0],
                [0, 0, 1]])
    R = np.dot(Rz, np.dot(Ry, Rx))

    # Transform the point cloud using the rotation matrix and subtract the translation offset
    P_rotated = np.dot(R, P)
    P_rotated = -P_rotated + np.array([[x0], [y0], [z0]])

    # Reshape the transformed point cloud back into a 3D array
    X_rotated = P_rotated[0, :].reshape(X.shape)
    Y_rotated = P_rotated[1, :].reshape(Y.shape)
    Z_rotated = P_rotated[2, :].reshape(Z.shape)

    # Clear the previous plot and plot the rotated point cloud for the current frame
    SURFACE.remove()
    SURFACE = ax.plot_surface(X_rotated, Y_rotated, Z_rotated)
    SURFACE.set_facecolors('red')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    fig.canvas.draw()
    fig.canvas.flush_events()
    #ax.set_title(f'Frame {frame+1}/{n_frames}')

# Create the animation using FuncAnimation
#anim = FuncAnimation(fig, update, frames=frames, interval=1000)

#plt.show(block=False)
"""