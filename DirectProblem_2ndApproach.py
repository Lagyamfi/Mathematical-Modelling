import matplotlib.pylab as plt
import os
cwd = os.getcwd()
from PIL import Image
import matplotlib.pylab as plt
import numpy as np
from scipy import linalg

plt.close('all')

""" Define Our System """

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

""" Define X_0, Y_0 and Theta """

# we start at position
x0=0.6
y0=-0.4

#looking at an angle
# In degreees!
theta=125
 


theta=theta*np.pi/180    



""" Other camera variables"""
# distance of focal length
# SHOULD BE LOWER THAN DISTANCE TO POINTS
f=x/2
# How many angles we see.
Field_of_view=100
Field_of_view=Field_of_view*np.pi/180
Half_Field_of_view=Field_of_view/2

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
    

def plot_them_all_with_angle( x0, y0, colors , f , theta, Half_Field_of_view):
    
    plt.figure()
    
    # MY POSITION
    plt.scatter(x0,y0,color='black', marker='X', label=r'Your position', s=200)

    # RANGE OF WHAT WE SEE
    angle_max_right= theta + Half_Field_of_view
    angle_max_left= theta - Half_Field_of_view
    
    right_line_x=[]
    right_line_y=[]
    left_line_x=[]
    left_line_y=[]

    radius=np.linspace(0.01,x*2,100)
    
    for i in range(len(radius)):
        
        right_line_x.append(x0+ radius[i]*np.cos(angle_max_right) )
        left_line_x.append(x0+ radius[i]*np.cos(angle_max_left) )
        right_line_y.append(y0+ radius[i]*np.sin(angle_max_right) )
        left_line_y.append(y0+ radius[i]*np.sin(angle_max_left) )
    
    plt.plot(right_line_x,right_line_y,color='black')
    plt.plot(left_line_x,left_line_y,color='black')
    plt.xlabel(r'$X_0$')
    plt.ylabel(r'$Y_0$')
    plt.title(r'Real position')
    #plot CORNERS
    for i in range(len(colors)):
        plt.scatter( colors[i][0], colors[i][1], color=colors[i][3], s=100)    
        
    #PLOT ARROW    
    u=f*np.cos(theta)
    v=f*np.sin(theta)
    plt.arrow(x0,y0,u,v,head_width=0.05, head_length=0.1, fc='k', ec='k')
    
    plt.legend()
    plt.axis([-x*1.3,x*1.3,-x*1.3,x*1.3])
    plt.savefig('img1.png', bbox='tight')

    
    
    
plot_them_all_with_angle(x0,y0,colors, f , theta, Half_Field_of_view)    


#create a distance from all points
distances=[]
for i in range(len(colors)):
    distances.append(distance_to_center_point_i(x0 , y0, colors[i][0], colors[i][1], f, theta , Half_Field_of_view ))
    
print(distances)
    
    
    
    
def plot_results(distances):
    
    plt.figure()
    
    for i in range(len(colors)):
        plt.scatter(distances[i], 0, color=colors[i][3], s=100)

    plt.xlabel(r'$d_i$')
    plt.ylabel(r'Nothing')
    plt.title(r'Distance to the center')
    #plt.axis([-1,1,-1,1])

    plt.show()
    plt.savefig('img2.png', bbox='tight')


plot_results(distances)        