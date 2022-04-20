import numpy as np
import matplotlib.pyplot as plt
from   numba import njit
from   scipy.signal import find_peaks
import time

t_start = time.time()

#Lorents equations 
@njit
def x_dot(sigma,x,y):
    return sigma*(y-x)

@njit        
def y_dot(rho,x,y,z):
    return (x*(rho-z)-y)

@njit
def z_dot(beta,x,y,z):
    return (x*y-beta*z)
    
#4th order Runge Kutta 
@njit
def RK4(sigma,rho,beta,x,y,z,dt):
    k1x = x_dot(sigma,x,y)
    k1y = y_dot(rho,x,y,z)
    k1z = z_dot(beta,x,y,z)
    
    k2x = x_dot(sigma,x+0.5*dt*k1x,y)
    k2y = y_dot(rho,x,y+0.5*dt*k1y,z)
    k2z = z_dot(beta,x,y,z+0.5*dt*k1z)
    
    k3x = x_dot(sigma,x+0.5*dt*k2x,y)
    k3y = y_dot(rho,x,y+0.5*dt*k2y,z)
    k3z = z_dot(beta,x,y,z+0.5*dt*k2z)
    
    k4x = x_dot(sigma,x+dt*k3x,y)
    k4y = y_dot(rho,x,y+dt*k3y,z)
    k4z = z_dot(beta,x,y,z+dt*k3z)
    
    x+=dt/6*(k1x+2*k2x+2*k3x+k4x)
    y+=dt/6*(k1y+2*k2y+2*k3y+k4y)
    z+=dt/6*(k1z+2*k2z+2*k3z+k4z)
    
    return x,y,z
    
#time evolve the system

def time_evolution(sigma,rho,beta,x0,y0,z0,t_start,t_end,dt):
    N = int((t_end-t_start)/dt)
    x = np.zeros(N+1)
    y = np.zeros(N+1)
    z = np.zeros(N+1)
    T = np.zeros(N+1)
    x[0],y[0],z[0] = x0, y0, z0
    for i in range(N):
        x[i+1],y[i+1],z[i+1] = RK4(sigma,rho,beta,x[i],y[i],z[i],dt)
        T[i+1] = T[i]+dt
    return x,y,z,T

#initial conditions 
sigma = 10.
rho   = 28.
beta  = 8./3 
dt    = 0.001
x0    = 0.001
y0    = 0.001
z0    = 0.001
T     = 50

#solve the Lorenz equations until from time 0 to time T
x,y,z,Tvec = time_evolution(sigma,rho,beta,x0,y0,z0,0,T,dt)


#plot the solution to the Lorenz equations
#plt.ylabel('z')
#plt.xlabel('t')
#plt.plot(Tvec,z)
#plt.plot(x,z1)


#find peaks 
PEAKS, _ = find_peaks(z)
nmax     = len(PEAKS)
zn       = z[PEAKS][:nmax-1]
zn1      =  z[PEAKS][1:]


# plot z_n+1 as a function of z_n 
#plt.figure()
#plt.ylabel('$z_{n+1}$')
#plt.xlabel('$z_n$')
#plt.plot(zn,zn1, "o")



def separation_distance(Tc,t_start = 0,dt = 0.001): 
    sigma = 10
    rho   = 28
    beta  = 8/3 
    T     = 50 

    #create random initial conditions 
    x_0 = np.random.uniform(10**(-6),4*10**(-6)) * (-1)**(np.random.randint(0,3, size=1))
    y_0 = np.random.uniform(10**(-6),4*10**(-6))  *(-1)**(np.random.randint(0,3, size=1))
    z_0 = np.random.uniform(10**(-6),4*10**(-6)) *(-1)**(np.random.randint(0,3, size=1))

    x,y,z,_ = time_evolution(sigma,rho,beta,x_0,y_0,z_0,t_start,T,dt)

    # Creating random values for x1, by constrctuing a ball with radii r around x0, where the random values are the angles, ie where you are on the sphere. 
    theta = np.random.uniform(0, np.pi)
    phi = np.random.uniform(0, 2 * np.pi)
    r = 10**(-6)

    x_1 = r * np.cos(phi) *np.sin(theta) + x[-1]
    y_1 = r* np.sin(theta) * np.sin(phi) + y[-1]
    z_1 = r * np.cos(theta) +z[-1]
    
    #constructing the vectors 
    x_0_vec = np.array([x[-1],y[-1],z[-1]])
    x_1_vec = np.array([x_1, y_1, z_1])
 

    x1, y1,z1,_  = time_evolution(sigma,rho,beta, x_0_vec[0], x_0_vec[1],x_0_vec[2],t_start,Tc,dt)
    x2, y2,z2,t2 = time_evolution(sigma,rho,beta, x_1_vec[0], x_1_vec[1], x_1_vec[2],t_start,Tc,dt)

    dist_x       = (x1-x2) **2
    dist_y       = (y1-y2)**2
    dist_z       = (z1-z2)**2 
    sep_dist     = np.sqrt(dist_x + dist_y + dist_z)

    return t2, sep_dist


# plot the logarithm of the elements in the vector as a function of dimensionless time
t2,norm = separation_distance(30)
plt.plot(t2,np.log(norm))
plt.xlabel("$t$")
plt.ylabel("$ln \delta(t)$")



"""
plt.figure()
# linear fitting
t2, norm = separation_distance(15) 
line = np.polyfit(t2,np.log(norm),1)
plt.plot(t2, line[0] *t2 + line[1])
plt.xlabel("$t$")
plt.ylabel("$ ln || x_{1} - x_{0}||$")
print ("line=", line)
"""

plt.plot(t2,np.log(norm))
#function for finding the average Liapunov exponent 
def average_lambda(Tc,n): 
    lambda_array        = np.zeros(n)
    t2, norm            = separation_distance(Tc)
    norm_array          = np.zeros((n,np.size(norm)))
    t_array             = np.zeros((n,np.size(t2)))
    norm_array[0]       = np.log(norm)
    t_array[0]          = t2
    for i in range(1,n):
        t, norm = separation_distance(Tc) 
        t_array[i] = t
        norm_array[i]   = np.log(norm)
        line            = np.polyfit(t2,np.log(norm),1)
        lambda_array[i] = line[0]

    A = np.sum(lambda_array) *1/n 
    return A,t_array,norm_array

"""
A, t_array , norm_array = average_lambda(15,10)
# finding the average lambda 
print("average = ", A)
"""


"""
# making the super-facy plot
plt.plot(t_array.T,norm_array.T)
plt.xlabel("$t$", size = 20)
plt.ylabel("$ln \delta(t)$",size = 20)
#plt.plot(t,norm_array[0])
#plt.plot(t,norm_array[1])
#plt.plot(t,norm_array[2])
#plt.plot(t,norm_array[3])
#plt.plot(t,norm_array[4])
plt.plot(t_array[0],np.log(10**(-6)*np.exp(0.9*t_array[0])), label= "$\delta_{0} e^{0.9 t}$",color = "black")
plt.legend(fontsize = 17)
plt.show()
"""
#plt.semilogy(t2,sep_dist)



#plt.show()


print("Time spent:")
print(time.time()-t_start)

# show plots
plt.show()

