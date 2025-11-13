import numpy as np
import matplotlib.pyplot as plt
from matplotlib.textpath import TextPath
import matplotlib.transforms as transforms

alpha = 0.1 # heat dif constant

letter = "EDGAR"
def letter_m_ini(x, y, N, temp_inside, temp_outside):
    tp = TextPath((0, 0), letter, size=1)
    bbox = tp.get_extents()

    scale = 0.8 / max(bbox.width, bbox.height)
    trans = transforms.Affine2D().scale(scale).translate(0.1, 0.1)
    path = tp.transformed(trans)

    xv, yv = np.meshgrid(x, y, indexing='xy')
    pts = np.vstack([xv.ravel(), yv.ravel()]).T

    inside = path.contains_points(pts).reshape((N, N))

    eps = 5 * 0.01 / (2 * np.sqrt(2) * np.arctanh(0.9))
    phi = np.where(inside, temp_inside, temp_outside)
    return np.tanh(phi / (np.sqrt(2.0) * eps))

def resplot(x, y, u_pred, dt, max_iter):
    """
    Plots the results at 4 different time steps.
    """
    fig = plt.figure(figsize=(8, 2))
    fig.subplots_adjust(wspace=0.4)  # increase gap between subplots
    
    # Plot 1: Initial state
    ax = plt.subplot(141)
    im1 = ax.imshow(u_pred[0], interpolation='nearest', cmap='jet',
                 extent=[x.min(), x.max(), y.min(), y.max()],
                 origin='lower', aspect='equal')
    ax.set_title('$u_0$', fontsize=15, pad=4)
    ax.text(0.5, -0.12, '$t = 0$', transform=ax.transAxes,
            ha='center', va='top', fontsize=12)
    ax.axis('off')
    im1.set_clim(-1, 1)
    
    # Plot 2: 25% complete
    ax = plt.subplot(142)
    l = int(0.25*max_iter)
    ax.imshow(u_pred[l], interpolation='nearest', cmap='jet',
                 extent=[x.min(), x.max(), y.min(), y.max()],
                 origin='lower', aspect='equal')
    ax.set_title('$u_{%d}$' % l, fontsize=15, pad=4)
    ax.text(0.5, -0.12, '$t = %.3f$' % (dt*l), transform=ax.transAxes,
            ha='center', va='top', fontsize=12)
    ax.axis('off')
    im2 = ax.images[0]
    im2.set_clim(-1, 1)
    
    # Plot 3: 50% complete
    ax = plt.subplot(143)
    l = int(0.5*max_iter)
    ax.imshow(u_pred[l], interpolation='nearest', cmap='jet',
                 extent=[x.min(), x.max(), y.min(), y.max()],
                 origin='lower', aspect='equal')
    ax.set_title('$u_{%d}$' % l, fontsize=15, pad=4)
    ax.text(0.5, -0.12, '$t = %.3f$' % (dt*l), transform=ax.transAxes,
            ha='center', va='top', fontsize=12)
    ax.axis('off')
    im3 = ax.images[0]
    im3.set_clim(-1, 1)
    
    # Plot 4: Final state
    ax = plt.subplot(144)
    im = ax.imshow(u_pred[-1], interpolation='nearest', cmap='jet',
                 extent=[x.min(), x.max(), y.min(), y.max()],
                 origin='lower', aspect='equal')
    ax.set_title('$u_{%d}$' % max_iter, fontsize=15, pad=4)
    ax.text(0.5, -0.12, '$t = %.3f$' % (dt*max_iter), transform=ax.transAxes,
            ha='center', va='top', fontsize=12)
    ax.axis('off')
    im.set_clim(-1, 1)
    
    # Save the figure
    fig.colorbar(im, ax=fig.axes, shrink=0.8, pad=0.02)
    plt.savefig('./cn2d_numpy.png')
    print("4. Plot saved as cn2d_numpy.png")

def tri_disc(N, a):
    """
    Creates the 1D implicit operator matrix (I - a*D_xx)
    with Neumann boundary conditions.
    """
    M = (np.diag(-a * np.ones(N-1), k=-1) +
         np.diag((1+2*a) * np.ones(N), k=0) +
         np.diag(-a * np.ones(N-1), k=1))
    
    M[0, 0] = 1+a
    M[0, 1] = -a
    M[-1, -1] = 1+a
    M[-1, -2] = -a
    
    return M

print("Running the thing")

# grid
N = 100
dt = 1e-4
x = np.linspace(0, 1, N)
y = np.linspace(0, 1, N)

h = 1/N
k = alpha * dt/(2*h**2)

T_font = 1.0        # Set letter temperature
T_surrounding = -0.5  # Set surrounding temperature

u0 = np.array(letter_m_ini(x, y, N, T_font, T_surrounding), dtype=np.float32)

A = np.array(tri_disc(N, k), dtype=np.float32)
C = np.array(tri_disc(N, k), dtype=np.float32) 
print("2. Generated discretization matrices")

A_inv = np.linalg.inv(A)
C_inv = np.linalg.inv(C)

u_pred = [np.copy(u0)]
u = np.copy(u0)
u_star = np.zeros((N,N), dtype=np.float32)
max_iter = 100

print("3. Started iteration session")
for it in range(max_iter-1):
    
    u_padded = np.pad(u, 1, mode='edge')
    
    S_north = u_padded[0:-2, 1:-1]
    S_south = u_padded[2:, 1:-1]
    S_west  = u_padded[1:-1, 0:-2]
    S_east  = u_padded[1:-1, 2:]
    S_center = (1 - 4*k) * u
    
    S = k * (S_north + S_south + S_west + S_east) + S_center

    u_star[1:-1] = np.dot(S[1:-1], A_inv.T)
    
    u_star[0,0] = k*(u[1,0]-2*u[0,0]+u[0,1])+u[0,0]
    u_star[0,-1] = k*(u[1,-1]-2*u[0,-1]+u[0,-2])+u[0,-1]
    u_star[-1,0] = k*(u[-2,0]-2*u[-1,0]+u[-1,1])+u[-1,0]
    u_star[-1,-1] = k*(u[-2,-1]-2*u[-1,-1]+u[-1,-2])+u[-1,-1]
    u_star[0,1:-1] = k*(-3*u[0,1:-1]+u[1,1:-1]+u[0,:-2]+u[0,2:])+u[0,1:-1]
    u_star[-1,1:-1] = k*(-3*u[-1,1:-1]+u[-2,1:-1]+u[-1,:-2]+u[-1,2:])+u[-1,1:-1]

    u = np.dot(C_inv, u_star.T)
    
    u = u.T 

    u_pred.append(np.copy(u))

resplot(x, y, u_pred, dt, max_iter)
