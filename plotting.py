


import matplotlib.pylab as plt
from scipy.optimize import fsolve


import math
import numpy as np


plt.close('all')


x=1
r=1
		
red  =    [x*np.cos(0*np.pi/2),  x*np.sin(0*np.pi/2) , 0,'r'] # r
blue  =   [x*np.cos(0.5*np.pi/2) , x*np.sin(0.5*np.pi/2) ,0,'b'] # b 
green  =  [x*np.cos(1*np.pi/2), x*np.sin(1*np.pi/2)  ,0,'g'] # g
cyan  =   [x*np.cos(1.5*np.pi/2)  ,  x*np.sin(1.5*np.pi/2) ,0,'c'] # c
magenta  =    [x*np.cos(2*np.pi/2)  , x*np.sin(2*np.pi/2)  ,0,'m'] # m 
yellow  = [x*np.cos(2.5*np.pi/2)  ,  x*np.sin(2.5*np.pi/2)  ,0,'y'] # y
black  =  [x*np.cos(3*np.pi/2)  ,  x*np.sin(3*np.pi/2) ,0,'k']  # k
pink  =   [x*np.cos(3.5*np.pi/2) , x*np.sin(3.5*np.pi/2)  ,0,'pink'] # w



def eq(x0,y0,theta):
    f1 = math.tan(math.pi-phi1-theta) - (y1-y0)/(x1-x0)
    f2 = math.tan(math.pi-phi2-theta) - (y2-y0)/(x2-x0)
    f3 = math.tan(math.pi-phi3-theta) - (y3-y0)/(x3-x0)
    return f1,f2,f3

def equations(p):
    x0,y0,theta = p
    f1 = lambdaa*(math.tan(math.pi-phi1-theta) - (y1-y0)/(x1-x0)) +(1-lambdaa)*(-x0 )
    f2 = lambdaa*(math.tan(math.pi-phi2-theta) - (y2-y0)/(x2-x0)) +(1-lambdaa)*(-y0 )
    f3 = lambdaa*(math.tan(math.pi-phi3-theta) - (y3-y0)/(x3-x0)) +(1-lambdaa)*(-theta)
    return (f1,f2,f3)


d1=0.375640000000002
d2=-0.027218328908727732
d3=-0.4129110992936491



x1=green[0]
y1=green[1]

x2=cyan[0]
y2=cyan[1]

x3=magenta[0]
y3=magenta[0]
#nan	nan	0.375640000000002	-0.027218328908727732	-0.4129110992936491	nan	nan	nan

f=0.0327927

phi1=np.arctan2(d1,f)
phi2=np.arctan2(d2,f)
phi3=np.arctan2(d3,f)
    

from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt


from numpy import exp,arange
import matplotlib.pylab as plt

# the function that I'm going to plot
def z_func(x0,y0):
 return math.tan(math.pi-phi1-theta) - (y1-y0)/(x1-x0)

def z_func_x(y0,theta):
 return np.tan(math.pi-phi1-theta) - (y1-y0)/(x1-x)

def z_func_y(x0,theta):
 return np.tan(math.pi-phi1-theta) - (y1-y)/(x1-x0)


theta_range=np.linspace(0,2*np.pi,101)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')


for i in range(len(theta_range)):
    theta=theta_range[i] 
    x = np.linspace(-1.0,1.0,10)
    y = np.linspace(-1.0,1.0,10)
    X,Y = plt.meshgrid(x, y) # grid of point
    Z = z_func(X, Y) # evaluation of the function on the grid
    
    ax.cla()
    #plt.pcolor(Z,vmin=-20,vmax=20)
    # Plot a basic wireframe.
    ax.plot_surface(X, Y, Z, rstride=10, cstride=10)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel(r'$tan(\phi_1+\theta)+(y_1-y_0)/(x_1-x_0)$')
    
    ax.set_zlim([-25,25])
    #plt.colorbar()
    ax.set_title(r'theta is {0:.2f} degrees'.format(theta*180/np.pi))
    plt.pause(0.2)
    



x_range=np.linspace(-1.0,1.0,101)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')


for i in range(len(x_range)):
    theta=np.linspace(0,2*np.pi,10)
    x = x_range[i]
    y = np.linspace(-1.0,1.0,10)
    X,Y = plt.meshgrid(theta,y) # grid of point
    Z = z_func_x(X, Y) # evaluation of the function on the grid
    
    ax.cla()
    #plt.pcolor(Z,vmin=-20,vmax=20)
    # Plot a basic wireframe.
    ax.plot_surface(X, Y, Z, rstride=10, cstride=10, cmap='jet',)
    
    ax.set_xlabel(r'$\theta$')
    ax.set_ylabel('Y')
    ax.set_zlabel(r'$tan(\phi_i+\theta)+(y_1-y_0)/(x_1-x_0)$')
    
    
    ax.set_zlim([-25,25])
    #plt.colorbar()
    ax.set_title('x is {0:.2f}'.format(x))
    plt.pause(0.2)
    


y_range=np.linspace(-1.0,1.0,101)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')


for i in range(len(x_range)):
    theta=np.linspace(0,2*np.pi,10)
    y = y_range[i]
    x = np.linspace(-1.0,1.0,10)
    X,Y = plt.meshgrid(x,theta) # grid of point
    Z = z_func_y(X, Y) # evaluation of the function on the grid
    
    ax.cla()
    #plt.pcolor(Z,vmin=-20,vmax=20)
    # Plot a basic wireframe.
    ax.plot_surface(X, Y, Z, rstride=10, cstride=10, cmap='jet',)
    
    ax.set_ylabel(r'$\theta$')
    ax.set_xlabel('X')
    ax.set_zlabel(r'$tan(\phi_i+\theta)+(y_1-y_0)/(x_1-x_0)$')
    
    
    ax.set_zlim([-25,25])
    #plt.colorbar()
    ax.set_title('y is {0:.2f}'.format(y))
    plt.pause(0.2)
    

