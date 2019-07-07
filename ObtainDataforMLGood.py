#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 16 10:15:13 2019

@author: marc
"""



import matplotlib.pylab as plt

plt.close('all')
# Turn interactive plotting off
plt.ioff()

import os
cwd = os.getcwd()

#https://github.com/miezekatze/evolisa

from PIL import Image


import matplotlib.pylab as plt
import numpy as np
import random

plt.close('all')

import time

start = time.clock()
#your code here    
def distance_to_center_point_i(x0,y0,xi,yi,f,theta, Half_Field_of_view):
    

    # angle where the point is
    phi=np.arctan2(yi-y0,xi-x0)
    
    angle_max_left= theta - Half_Field_of_view
    angle_max_right= theta + Half_Field_of_view
    #print(angle_max_left,phi,angle_max_right)    


    if (angle_max_left < phi < angle_max_right) or (angle_max_left < phi-2*np.pi < angle_max_right) or (angle_max_left < phi+2*np.pi < angle_max_right) :
            #compute vector to where we point
            
            distance = -f* np.tan(phi-theta)
        
                
    else:
            distance=np.NaN
                
    return distance
    
from scipy import linalg
def fit_circle_2d(x, y, w=[]):
    
    A = np.array([x, y, np.ones(len(x))]).T
    b = x**2 + y**2
    
    # Modify A,b for weighted least squares
    if len(w) == len(x):
        W = np.diag(w)
        A = np.dot(W,A)
        b = np.dot(W,b)
    
    # Solve by method of least squares
    c = linalg.lstsq(A,b)[0]
    
    # Get circle parameters from solution c
    xc = c[0]/2
    yc = c[1]/2
    r = np.sqrt(c[2] + xc**2 + yc**2)
    return xc, yc, r


def findcircles(redx,redy):    
    xc_red, yc_red, r_red = fit_circle_2d(redx, redy)

    #plt.scatter(xc_red,yc_red,color='yellow')
    
    return xc_red
    

def find_distances(xc_red):
    #colors=[red,blue,green,cyan,magenta,yellow,black,pink]
    if xc_red<1:
        distance=np.NaN
    else:
        distance=(xc_red-img_size[0]/2)*0.00176-0.03444
        
    return distance


    
def plot_results(distances):
    
    fig, ax = plt.subplots()
    
    for i in range(len(colors)):
        plt.scatter(distances[i], 0, color=colors[i][3], s=100)

    #plt.axis([-1,1,-1,1])
    for spine in plt.gca().spines.values():
        spine.set_visible(False) 
        
    fig.patch.set_visible(False)
    ax.axis('off')
    #plt.show()
    plt.savefig('images/foto.jpg',dpi=100)
    


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


red_c=[254,0,0]
blue_c=[0,0,254]
green_c=[0,128,1]
cyan_c=[0,192,191]
magenta_c=[191,0,191]
yellow_c=[191,191,0]
black_c=[0,0,0]
pink_c=[255,192,203]


Howmanydata=5000

data=[]
myfile = open('target.txt', 'w')
myfile2 = open('data.txt', 'w')

for i in range(Howmanydata):
    # we start at position
    x0=random.uniform(-0.5,0.5)
    y0=random.uniform(-0.5,0.5)
    
    #looking at an angle
    theta=random.uniform(0,359)
     
    theta=theta*np.pi/180    
    
    data_real=[x0,y0,theta]
    
    for item in data_real:
            myfile.write("%s\t" % item)
    myfile.write("\n" )
    
    # distance of focal length
    # SHOULD BE LOWER THAN DISTANCE TO POINTS
    x=1
    f=x/2
    # How many angles we see.
    Field_of_view=100
    Field_of_view=Field_of_view*np.pi/180
    Half_Field_of_view=Field_of_view/2
    
    
    
    #create a distance from all points
    distances=[]
    for i in range(len(colors)):
        distances.append(distance_to_center_point_i(x0 , y0, colors[i][0], colors[i][1], f, theta , Half_Field_of_view ))
       
    print('Original distances are {}'.format(distances))
        
        
    
    plot_results(distances)        
    
    img = Image.open("images/foto.jpg")
    img_size = img.size
    
    
    redx=[]
    redy=[]
    
    greenx=[]
    greeny=[]
    
    bluex=[]
    bluey=[]
    
    cyanx=[]
    cyany=[]
    
    magentax=[]
    magentay=[]
    
    yellowx=[]
    yellowy=[]
    
    blackx=[]
    blacky=[]
    
    pinkx=[]
    pinky=[]
    
    
    for y in range(0, img.size[1]):
            for x in range(0, img.size[0]):
                r, g, b = img.getpixel((x, y))
                if (r<250 and g<250 and b<250): #[r,g,b] != white:    
                    
                    if(200<r and g<50 and b<50):
                        redx.append(x)
                        redy.append(y)
                    elif(r<50 and 100<g and b<50):
                        greenx.append(x)
                        greeny.append(y)
                    elif(r<50 and g<50 and b>200):
                        bluex.append(x)
                        bluey.append(y)
                    elif(r<50 and 150<g<210 and 150<b<210):
                        cyanx.append(x)
                        cyany.append(y)
                    elif(150<r<210 and g<50 and 150<b<210):
                        magentax.append(x)
                        magentay.append(y)
                    elif(150<r<210 and 150<g<210 and b<50):
                        yellowx.append(x)
                        yellowy.append(y)
                    elif(r<50 and g<50 and b<50):
                        blackx.append(x)
                        blacky.append(y)
                    elif(r>200 and 170<g<220 and 170<b<220):
                        pinkx.append(x)
                        pinky.append(y)
                        
    
                        
    redx=np.array(redx)
    redy=np.array(redy)
    bluex=np.array(bluex)
    bluey=np.array(bluey)
    greenx=np.array(greenx)
    greeny=np.array(greeny)
    cyanx=np.array(cyanx)
    cyany=np.array(cyany)
    magentax=np.array(magentax)
    magentay=np.array(magentay)
    yellowx=np.array(yellowx)
    yellowy=np.array(yellowy)
    blackx=np.array(blackx)
    blacky=np.array(blacky)
    pinkx=np.array(pinkx)
    pinky=np.array(pinky)
                        
    """
    plt.figure()
    
    plt.scatter(redx,redy,color='r')
    plt.scatter(greenx,greeny,color='g')
    plt.scatter(bluex,bluey,color='b')
    plt.scatter(cyanx,cyany,color='cyan')
    plt.scatter(magentax,magentay,color='magenta')
    plt.scatter(yellowx,yellowy,color='yellow')
    plt.scatter(blackx,blacky,color='k')
    plt.scatter(pinkx,pinky,color='pink')
    plt.axis([0,img_size[0],0,img_size[1]])
    """
    
    xc_red=findcircles(redx,redy)
    xc_blue=findcircles(bluex,bluey)
    xc_green=findcircles(greenx,greeny)
    xc_cyan=findcircles(cyanx,cyany)
    xc_magenta=findcircles(magentax,magentay)
    xc_yellow=findcircles(yellowx,yellowy)
    xc_black=findcircles(blackx,blacky)
    xc_pink=findcircles(pinkx,pinky)
    
    
    
    Computed_distances=[]
    Computed_distances.append(find_distances(xc_red))
    Computed_distances.append(find_distances(xc_blue))
    Computed_distances.append(find_distances(xc_green))
    Computed_distances.append(find_distances(xc_cyan))
    Computed_distances.append(find_distances(xc_magenta))
    Computed_distances.append(find_distances(xc_yellow))
    Computed_distances.append(find_distances(xc_black))
    Computed_distances.append(find_distances(xc_pink))
    print('Computed distances are {}'.format(Computed_distances))
    data.append(Computed_distances)
    for item in Computed_distances:
            myfile2.write("%s\t" % item)
    myfile2.write("\n" )


myfile.close()
myfile2.close()

print(time.clock() - start)

"""
y=[0.4315909396792524, 0.002775479861402098, -0.32877606326108455,0.46463873096925346, 0.014150106394310266, -0.3223589352274715,-0.06056483164117936,0.18677064833578438, -0.14179065895920714, -0.5292300450156595,0.15173568]
x=[232.85190129553553, -19.999999999998806, -120.85574235010472,227.90183350048665, -25.13446998514013, -215.00000000000028,8.105903226,232.95960267406372, 26.851901295532173, -214.99999999999966,8.148098]

x=np.array(x)
y=np.array(y)
from scipy import stats
plt.figure()
plt.scatter(x,y)
slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)
"""