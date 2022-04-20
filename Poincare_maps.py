import numpy as np 
import matplotlib.pyplot as plt

rmax =10
N = 1000
M = 100

# the line y= r
r = np.linspace(0.001,rmax,N)
y = r


def  Poincare_map(r): 
    A = (1 +np.exp(- 1) * (r**(-2) -1) )**(-1/2)
    return A

def Poincare_map_challange(r,A,omega): 
    return np.pi + (r-np.pi)*np.exp(( np.exp(-2*omega)-1)*(-A/omega))

def cobweb(r0,M):
    p_list = np.zeros(M)
    r_list = np.zeros(M)
    r_list[0] = r0
    p_list[0] = 0
    r_list[1] = r0
    p_list[1] = Poincare_map(r0)

    for i in range (2,M): 
        if(i%2 != 0): #if odd
            p_list[i] = Poincare_map(p_list[i-1])
            r_list[i] = r_list[i-1]
    
        else: #if even
            p_list[i] = p_list[i-1]
            r_list[i] = r[np.argmin(np.abs(y-p_list[i]))]
            
    return r_list, p_list     



    
def cobweb_challange(r0,M,A,omega):
    p_list = np.zeros(M)
    r_list = np.zeros(M)
    r_list[0] = r0
    p_list[0] = 0
    r_list[1] = r0
    p_list[1] = Poincare_map_challange(r0,A,omega)

    for i in range (2,M): 
        if(i%2 != 0): #if odd
            p_list[i] = Poincare_map_challange(p_list[i-1],A,omega)
            r_list[i] = r_list[i-1]
    
        else: #if even
            p_list[i] = p_list[i-1]
            r_list[i] = r[np.argmin(np.abs(y-p_list[i]))]
            
    return r_list, p_list     

"""
#Warm-up 1a)
first_plot =  cobweb(0.1,M)
second_plot = cobweb(2.7,M)

plt.plot(r,y,label="y(r) = r")
plt.plot(first_plot[0],first_plot[1], label ="Web")
plt.plot(second_plot[0],second_plot[1], label="Web")
plt.plot(r,Poincare_map(r), label = "y(r) = P(r)")
plt.xlabel("r")
plt.ylabel("y")
plt.ylim(0,1.5)

#Warm-up 1b) 
# M = 30 
N_list = np.linspace(0,30,30)
r_rist2 = cobweb(0.1, 30)
plt.plot(N_list,cobweb(0.1,M)[0])
plt.xlabel("iteration")
plt.ylabel("r")
"""

#challange 1b) 

A  = -2
omega = 7
r0 = 0.1

plt.plot(r,Poincare_map_challange(r,A,omega), label = "y(r) = P(r)")
plt.plot(r,y,label="y(r) = r")
plt.plot(cobweb_challange(r0,M,A,omega)[0],cobweb_challange(r0,M,A,omega)[1], label = "cobweb")
plt.xlabel("$r$")
plt.ylabel("$y$")
plt.legend()
plt.show() 

print("poincare map=",Poincare_map_challange(r,1,1))

