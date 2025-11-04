import numpy as np
import matplotlib.pyplot as plt

#hey guys its kinda scuffed but i managed to make 

# Defining our problem

length = 50 #mm
time = 0.5 #seconds
nodes = 40

#things to control for lukes simulation

temp = 1e9 # temperature of the comet on the surface
r1 = 30 #outer radius of the planet
r2 = 25 #outer radius of the first layer
r3 = 10 #outer radius of the core of the planet
a_space = 0.001 #diffusivity of space
a1 = 500 #diffusivity of outer layer
a2 = 20 #diffusivity of middle layer
a3 = 10 #diffusivity of core

#doing a better a array

x_list = np.linspace(0,nodes-1, nodes)
y_list = np.linspace(0,nodes-1, nodes)
X, Y = np.meshgrid(x_list, y_list)

a = np.full((nodes,nodes), a_space)
a[np.sqrt(X**2 + Y**2) <= r1] = a1
a[np.sqrt(X**2 + Y**2) <= r2] = a2
a[np.sqrt(X**2 + Y**2) <= r3] = a3

# Initialization 

dx = length / (nodes-1)
dy = length / (nodes-1)

dt = min(   dx**2 / (4 * np.max(a)),     dy**2 / (4 * np.max(a)))

t_nodes = int(time/dt) + 1

u = np.zeros((nodes, nodes)) + 20 # Plate is initially as 20 degres C

# Boundary Conditions 

comet = int(30 / np.sqrt(2)) + 1
u[comet - 2: comet, comet - 2: comet] = np.full((2,2), temp)

# Visualizing with a plot

fig, axis = plt.subplots()

pcm = axis.pcolormesh(u, cmap=plt.cm.jet, norm = 'log')
plt.colorbar(pcm, ax=axis)

#plotting radii
values = np.linspace(0,nodes,100)
pr1 = axis.plot(values, np.sqrt(r1**2 - values**2))
pr2 = axis.plot(values, np.sqrt(r2**2 - values**2))
pr3 = axis.plot(values, np.sqrt(r3**2 - values**2))
axis.scatter(int(comet) - 1, int(comet) - 1, s = 2)

# Simulating

counter = 0

while counter < time :

    w = u.copy()

    #trying to make this shit work boundary shit
    w[:,0] = w[:,1]
    w[:,-1] = w[:,-2]
    w[0,:] = w[1,:]
    w[-1,:] = w[-2,:]

    ddx = (w[2:, 1:-1] - 2*w[1:-1, 1:-1] + w[:-2, 1:-1]) / dx**2
    ddy = (w[1:-1, 2:] - 2*w[1:-1, 1:-1] + w[1:-1, :-2]) / dy**2
    u[1:-1, 1:-1] = w[1:-1, 1:-1] + dt * a[1:-1, 1:-1] * (ddx + ddy)

    #trying to make this shit work boundary shit
    u[:,0] = u[:,1]
    u[:,-1] = u[:,-2]
    u[0,:] = u[1,:]
    u[-1,:] = u[-2,:]

    counter += dt

    print("t: {:.3f} [s], Average temperature: {:.2f} Celcius".format(counter, np.average(u)))

    # Updating the plot

    pcm.set_array(u)
    axis.set_title("Distribution at t: {:.3f} [s].".format(counter))
    plt.pause(0.01)


plt.show()
