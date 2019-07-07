"""
Created on Mon Nov 26 10:49:30 2018

@author: marc
"""

import math
import random
import matplotlib.pylab as plt
import os
import numpy as np


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

plt.close('all')

#img_size=[100,100]

#img=Image.new('RGB',(img_size[0],img_size[1]))


x=1

red  =    [x*np.cos(0*np.pi/2),  x*np.sin(0*np.pi/2) , 0,'r'] # r
blue  =   [x*np.cos(0.5*np.pi/2) , x*np.sin(0.5*np.pi/2) ,0,'b'] # b 
green  =  [x*np.cos(1*np.pi/2), x*np.sin(1*np.pi/2)  ,0,'g'] # g
cyan  =   [x*np.cos(1.5*np.pi/2)  ,  x*np.sin(1.5*np.pi/2) ,0,'c'] # c
magenta  =    [x*np.cos(2*np.pi/2)  , x*np.sin(2*np.pi/2)  ,0,'m'] # m 
yellow  = [x*np.cos(2.5*np.pi/2)  ,  x*np.sin(2.5*np.pi/2)  ,0,'y'] # y
black  =  [x*np.cos(3*np.pi/2)  ,  x*np.sin(3*np.pi/2) ,0,'k']  # k
pink  =   [x*np.cos(3.5*np.pi/2) , x*np.sin(3.5*np.pi/2)  ,0,'pink'] # w



colors=[red,blue,green,cyan,magenta,yellow,black,pink]



# Make data
R=0.1
u = np.linspace(0, 2 * np.pi, 10)
v = np.linspace(0, np.pi, 10)

def sphere_coordinates(R,x0,y0,z0,color):
    x = x0+R * np.outer(np.cos(u), np.sin(v))
    y = y0+R * np.outer(np.sin(u), np.sin(v))
    z = z0+R * np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(x, y, z, color=color)
    
R=1

 # Plot the surface

fig = plt.figure( figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')

for i in range(len(colors)):
    sphere_coordinates(R,colors[i][0],colors[i][1],colors[i][2],colors[i][3])

#ax.set_xlim(-1, 1)
#ax.set_ylim(-1, 1)
#ax.set_zlim(-1, 1)

ax.set_xlim(1.5*x, -1.5*x)
ax.set_ylim(1.5*x, -1.5*x)
ax.set_zlim(1.5*x, -1.5*x)


#distance=15
#elevation=0
#angle=0


def from_Cartesian_to_Spherical(x,y,z):
    return [
         (x**2+y**2+z**2)**0.5,
         np.arccos(z/((x**2+y**2+z**2)**0.5)),
         np.arctan(y/x)
    ]




x=0.01
y=0.01
z=0.01

cartesian=from_Cartesian_to_Spherical(x,y,z)
distance=cartesian[0]
elevation=cartesian[1]
angle=cartesian[2]



#ax.view_init(elevation,angle)
#ax.dist=distance

#ax.set_axis_off()
#ax.grid(False)

"""
ax1=ax
ax2=ax


xlm=ax1.get_xlim3d() #These are two tupples
ylm=ax1.get_ylim3d() #we use them in the next
zlm=ax1.get_zlim3d() #graph to reproduce the magnification from mousing
axx=ax1.get_axes()
azm=axx.azim
ele=axx.elev


ax2.view_init(elev=ele, azim=azm) #Reproduce view
ax2.set_xlim3d(xlm[0],xlm[1])     #Reproduce magnification
ax2.set_ylim3d(ylm[0],ylm[1])     #...
ax2.set_zlim3d(zlm[0],zlm[1]) 
"""

plt.show()

    
"""


distance=6
elevation=10

#moving plot

distance_range=np.linspace(1,360,50)
#distance_range=np.sqrt(distance_range)


for angle in distance_range:#360):
    print(distance)
    for i in range(len(colors)):
        ax.scatter(colors[i][0],colors[i][1],colors[i][2],s=10*apparent_position_to_corner(colors[i],distance,elevation,angle),c=colors[i][3])
    ax.view_init(elevation,angle)
    ax.dist=distance
    #ax.set_axis_off()
    #ax.grid(False)
    plt.draw()
    plt.pause(0.01)



"""

