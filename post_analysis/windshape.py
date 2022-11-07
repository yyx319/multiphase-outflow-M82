"""
Script to produce edge-on, face-on, and 3D perspective views of the wind
"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.collections import PolyCollection

########################################################################
# Geometric parameters
########################################################################

# Launch region size in kpc
rlaunch = 0.25

# Plot window limits, in kpc
lim = [-2.5, 2.5]

# HI North
theta_HI_N_in = 30.08 * np.pi/180
theta_HI_N_out = 85.96 * np.pi/180
phi_HI_N = 4.73 * np.pi/180

# HI South
theta_HI_S_in = 42.86 * np.pi/180
theta_HI_S_out = 65.26 * np.pi/180
phi_HI_S = 6.09 * np.pi/180

# CO North
theta_CO_N_in = 11.34 * np.pi/180
theta_CO_N_out = 80.87 * np.pi/180
phi_CO_N = -3.16 * np.pi/180

# CO South
theta_CO_S_in = 26.63 * np.pi/180
theta_CO_S_out = 54.60 * np.pi/180
phi_CO_S = 11.57 * np.pi/180


########################################################################
# Useful little geometric routines
########################################################################
def rotmat(vec):
    """
    Returns the rotation matrix that rotates a vector vec
    so that it lies along the z axis.
    
    Parameters:
        vec : array(3)
            input vector; need not be a unit vector
            
    Returns:
        rotmat : array(3,3)
            the rotation matrix that rotates vec to lie along
            the z axis
    """
    # Form unit vector
    nvec = vec/np.sqrt(np.sum(vec**2))
    nx = nvec[0]
    ny = nvec[1]
    nz = nvec[2]
    nxy = np.sqrt(nvec[0]**2 + nvec[1]**2)
    # Form rotation matrix
    if nxy > 0:
        rotmat = np.array([
                [(ny**2+nx**2*nz)/nxy**2, nx*ny*(nz-1)/nxy**2, -nx],
                [nx*ny*(nz-1)/nxy**2, (nx**2+ny**2*nz)/nxy**2, -ny],
                [nx, ny, nz]])
    else:
        rotmat = np.array(np.eye(3,3))
    return rotmat

def zrot(vec, x, y, z):
    """
    Given a vector and a set of coordinate arrays x, y, z, return
    x, y, and z rotated under a rotation such that vec lies along
    the z axis
    
    Parameters:
        vec : array(3)
            input vector; need not be a unit vector
        x, y, z : array
            coordinate arrays to be rotated; must all have the same
            shape
            
    Returns:
        xr, yr, zr : array
            rotated versions of x, y, and z
    """
    # Get rotation matrix
    rmat = rotmat(vec)
    # Form array of vectors to be rotated
    sh = x.shape
    vecrot = np.transpose(np.vstack((x.flatten(), y.flatten(), z.flatten())))
    # Rotate
    xyzr = np.einsum('ij,ki', rmat, vecrot)
    # Extract result and return
    xr = xyzr[0,:].reshape(sh)
    yr = xyzr[1,:].reshape(sh)
    zr = xyzr[2,:].reshape(sh)
    return xr, yr, zr

def rodrot(vec, theta):
    """
    This returns the rotation matrix for rotation about a vector vec
    by an angle theta, computed using Rodrigues' rotation formula
    
    Parameters:
        vec : array(3)
            vector about which to rotate
        theta : float
            rotation angle
            
    Returns:
        rotmat : array(3,3)
            rotation matrix
    """
    u = vec/np.sqrt(np.sum(vec**2))
    ux = u[0]
    uy = u[1]
    uz = u[2]
    c = np.cos(theta)
    s = np.sin(theta)
    rotmat = np.array([
            [c+ux**2*(1-c),    ux*uy*(1-c)-uz*s, ux*uz*(1-c)+uy*s],
            [uy*ux*(1-c)+uz*s, c+uy**2*(1-c),    uy*uz*(1-c)-ux*s],
            [uz*ux*(1-c)-uy*s, uz*uy*(1-c)+ux*s, c+uz**2*(1-c)]
        ])
    return rotmat

def boxclip(x, y, z, xlim, ylim, zlim):
    """
    Given a set of coordinates (x,y,z) and a set of box limits 
    (xlim, ylim, zlim), return versions of (x,y,z) with the values
    of points outside the box set to NaN
    
    Parameters:
        x, y, z : array
            input coordinates
        xlim, ylim, zlim : array(2)
            lower and upper coordinate limits for the clipping box
    
    Returns:
        xc, yc, zc : array
            clipped versions of x, y, z with points outside the
            clipping box replaced by NaN
    """
    idx = np.logical_or.reduce((
            x < xlim[0], x > xlim[1],
            y < ylim[0], y > ylim[1],
            z < zlim[0], z > zlim[1]))
    xc = np.copy(x)
    xc[idx] = np.nan
    yc = np.copy(y)
    yc[idx] = np.nan
    zc = np.copy(z)
    zc[idx] = np.nan
    return xc, yc, zc

def arc(vec1, vec2, n=50, r=1.0):
    """
    Given a pair of vectors with their bases at the origin, return
    a set of points forming an arc of constant radius between them,
    uniformly spaced in angle
    
    Parameters:
        vec1, vec2 : array(3)
            vectors between which arc is to be found
        n : int
            number of points in the arc
        r : float
            radius of the arc
            
    Returns:
        x, y, z : array
            3d (x,y,z) coordinates of the points that constitute
            the arc
    """
    # First step: take the cross product of the two input vectors to
    # find an orthogonal vector about which we can rotate
    avec = np.asarray(vec1)
    bvec = np.asarray(vec2)
    bvecn = bvec/np.sqrt(np.sum(bvec**2))
    ovec = np.array([avec[1]*bvec[2]-avec[2]*bvec[1],
                     avec[2]*bvec[0]-avec[0]*bvec[2],
                     avec[0]*bvec[1]-avec[1]*bvec[0]])
    ovec = ovec / np.sqrt(np.sum(ovec**2))
    # Second step: get the angle between the two vectors
    theta = np.arccos(np.sum(avec*bvec) / \
                      np.sqrt(np.sum(avec**2)*np.sum(bvec**2)))
    # Third step: construct arc, using Rodrigues's rotation formula
    x = np.zeros(n)
    y = np.zeros(n)
    z = np.zeros(n)
    for i, th in enumerate(np.linspace(0, theta, n)):
        pt = np.einsum('ij,i', rodrot(ovec, th), r*bvecn)
        x[i] = pt[0]
        y[i] = pt[1]
        z[i] = pt[2]
    # Return
    return np.array(x), np.array(y), np.array(z)

########################################################################
# Plotting methods
########################################################################

# Method to plot cross section of wind cones parallel to line of sight
def plot_cones_parallel(ax, lim,
                        theta_N_in, theta_N_out, phi_N,
                        theta_S_in, theta_S_out, phi_S,
                        color='C0', alpha=0.5, npt=50):

    # Region at x > 0
    x = np.linspace(0, lim[1], npt)
    y_in = x / np.tan(theta_N_in + phi_N)
    y_out = x /np.tan(theta_N_out + phi_N)
    ax.fill_between(x, y_in, y_out, color=color,
                    alpha=alpha, lw=0)
    y_in = -x / np.tan(theta_S_in + phi_S)
    y_out = -x /np.tan(theta_S_out + phi_S)
    ax.fill_between(x, y_in, y_out, color=color,
                    alpha=alpha, lw=0)

    # Region at x < 0
    x = np.linspace(lim[0], 0, npt)
    y_in = x / np.tan(-theta_N_in + phi_N)
    y_out = x /np.tan(-theta_N_out + phi_N)
    ax.fill_between(x, y_in, y_out, color=color,
                    alpha=alpha, lw=0)
    y_in = -x / np.tan(-theta_S_in + phi_S)
    y_out = -x /np.tan(-theta_S_out + phi_S)
    ax.fill_between(x, y_in, y_out, color=color,
                    alpha=alpha, lw=0)

# Method to plot cross section of wind cones perpendicular to line of
# sight
def plot_cones_perpendicular(ax, lim,
                             theta_N_in, theta_N_out, phi_N,
                             theta_S_in, theta_S_out, phi_S,
                             color='C0', alpha=0.5, npt=50):

    # Northern hemisphere
    vec = np.array([np.sin(phi_N), 0, np.cos(phi_N)])
    z = np.array([1])
    y = z * np.sqrt(np.tan(theta_N_in)**2-np.tan(phi_N)**2)
    x = -z * np.tan(theta_N_in)
    xr, yr_in, zr_in = zrot(vec, x, y, z)
    y = z * np.sqrt(np.tan(theta_N_out)**2-np.tan(phi_N)**2)
    x = -z * np.tan(theta_N_out)
    xr, yr_out, zr_out = zrot(vec, x, y, z)
    y = np.linspace(0, lim[1])
    ax.fill_between(y, y * zr_out/yr_out, y * zr_in/yr_in,
                    color=color, alpha=alpha, lw=0)
    ax.fill_between(-y, y * zr_out/yr_out, y * zr_in/yr_in,
                    color=color, alpha=alpha, lw=0)
    
    # Southern hemisphere
    y = z * np.sqrt(np.tan(theta_S_in)**2-np.tan(phi_N)**2)
    x = -z * np.tan(theta_S_in)
    xr, yr_in, zr_in = zrot(vec, x, y, z)
    y = z * np.sqrt(np.tan(theta_S_out)**2-np.tan(phi_N)**2)
    x = -z * np.tan(theta_S_out)
    xr, yr_out, zr_out = zrot(vec, x, y, z)
    y = np.linspace(0, lim[1])
    ax.fill_between(y, -y * zr_out/yr_out, -y * zr_in/yr_in,
                    color=color, alpha=alpha, lw=0)
    ax.fill_between(-y, -y * zr_out/yr_out, -y * zr_in/yr_in,
                    color=color, alpha=alpha, lw=0)
    
# Method to generate perspective view of wind cone
def plot_cones_perspective(ax, lim, rsphere,
                           theta_N_in, theta_N_out, phi_N,
                           theta_S_in, theta_S_out, phi_S,
                           color='C0', alpha=0.25, npt=2000):

    # Northern hemisphere
    x = np.linspace(lim[0], lim[1], npt)
    xxc, yyc = np.meshgrid(x, x)
    vec = np.array([np.sin(phi_N), 0, np.cos(phi_N)])

    # Outer cone; create by using equation of cone aligned to z axis,
    # then rotating and clipping to the plot volume
    zc = np.sqrt(xxc**2+yyc**2) / np.tan(theta_N_out)
    xxc, yyc, zc = zrot(vec, xxc, yyc, zc)
    xxc, yyc, zc = boxclip(xxc, yyc, zc, lim, lim, lim)
    ax.plot_surface(xxc, yyc, zc, color=color, alpha=alpha, lw=0,
                    rasterized=True)

    # Inner cone
    zc = np.sqrt(xxc**2+yyc**2) / np.tan(theta_N_in)
    xxc, yyc, zc = zrot(vec, xxc, yyc, zc)
    xxc, yyc, zc = boxclip(xxc, yyc, zc, lim, lim, lim)
    ax.plot_surface(xxc, yyc, zc, color=color, alpha=alpha, lw=0,
                    rasterized=True)

    # Repeat for Southern hemisphere
    x = np.linspace(lim[0], lim[1], npt)
    xxc, yyc = np.meshgrid(x, x)
    vec = np.array([np.sin(phi_S), 0, np.cos(phi_S)])
    zc = -np.sqrt(xxc**2+yyc**2) / np.tan(theta_S_out)
    xxc, yyc, zc = zrot(vec, xxc, yyc, zc)
    xxc, yyc, zc = boxclip(xxc, yyc, zc, lim, lim, lim)
    ax.plot_surface(xxc, yyc, zc, color=color, alpha=alpha, lw=0,
                    rasterized=True)
    zc = -np.sqrt(xxc**2+yyc**2) / np.tan(theta_S_in)
    xxc, yyc, zc = zrot(vec, xxc, yyc, zc)
    xxc, yyc, zc = boxclip(xxc, yyc, zc, lim, lim, lim)
    ax.plot_surface(xxc, yyc, zc, color=color, alpha=alpha, lw=0,
                    rasterized=True)

    # Draw sphere
    x = np.linspace(-rsphere, rsphere, 2000)
    xxs, yys = np.meshgrid(x, x)
    rho = np.sqrt(xxs**2 + yys**2)
    zs = np.sqrt(rsphere**2 - np.minimum(rho, 1.0)**2)
    zs[zs == 0.0] = np.nan
    ax.plot_surface(xxs, yys, zs, color='k',
                    alpha=alpha, lw=0, rasterized=True)
    ax.plot_surface(xxs, yys, -zs, color='k',
                    alpha=alpha, lw=0, rasterized=True)

    
########################################################################
# Make plot
########################################################################

# Set plotting preferences
plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=12)

# Open figure
fig = plt.figure(1, figsize=(5,5))
fig.clf()

# Cut along line of sight
ax = fig.add_subplot(2,2,1)
plot_cones_parallel(ax, lim, theta_HI_N_in, theta_HI_N_out, phi_HI_N,
                    theta_HI_S_in, theta_HI_S_out, phi_HI_S, color='C0')
plot_cones_parallel(ax, lim, theta_CO_N_in, theta_CO_N_out, phi_CO_N,
                    theta_CO_S_in, theta_CO_S_out, phi_CO_S, color='C1')
x = np.linspace(-rlaunch, rlaunch)
ax.fill_between(x, np.sqrt(rlaunch**2-x**2),
                -np.sqrt(rlaunch**2-x**2), color='#888888',
                alpha=1)
ax.arrow(1, -2, -1, 0, head_width=0.1, color='k')
ax.text(0.5, -1.85, 'To Sun', ha='center', fontsize=8)
ax.set_aspect('equal')
ax.set_xlim(lim)
ax.set_ylim(lim)
ax.set_xlabel('$s$ [kpc]')
ax.set_ylabel(r'$\varpi$ [kpc]')

# Dummy fill for legend
ax.fill_between([-10,-10], [-10,-10], color='C0', alpha=0.5, lw=0, 
                label=r'H~\textsc{i}')
ax.fill_between([-10,-10], [-10,-10], color='C1', alpha=0.5, lw=0,
                label=r'CO')
ax.fill_between([-10,-10], [-10,-10], color='k', alpha=0.5, lw=0,
                label=r'Launch region')

# Cut perpendicular to LOS
ax = fig.add_subplot(2,2,2)
plot_cones_perpendicular(ax, lim, theta_HI_N_in, theta_HI_N_out, phi_HI_N,
                         theta_HI_S_in, theta_HI_S_out, phi_HI_S, color='C0')
plot_cones_perpendicular(ax, lim, theta_CO_N_in, theta_CO_N_out, phi_CO_N,
                         theta_CO_S_in, theta_CO_S_out, phi_CO_S, color='C1')
x = np.linspace(-rlaunch, rlaunch)
ax.fill_between(x, np.sqrt(rlaunch**2-x**2),
                -np.sqrt(rlaunch**2-x**2), color='#888888',
                alpha=1)
ax.text(0, -2, '$\otimes$', ha='center', fontsize=10)
ax.text(0, -1.7, 'To Sun', ha='center', fontsize=8)
ax.set_aspect('equal')
ax.set_xlim(lim)
ax.set_ylim(lim)
ax.yaxis.tick_right()
ax.set_xlabel(r'$\varpi_t$ [kpc]')
ax.yaxis.set_label_position("right")
ax.set_ylabel(r'$\varpi$ [kpc]')

# Perspective view of HI
dist = 10
ax = fig.add_subplot(2,2,3, projection='3d')
plot_cones_perspective(ax, lim, rlaunch,
                       theta_HI_N_in, theta_HI_N_out, phi_HI_N,
                       theta_HI_S_in, theta_HI_S_out, phi_HI_S, color='C0')
ax.set_xlim3d(lim)
ax.set_ylim3d(lim)
ax.set_zlim3d(lim)
ax.set_box_aspect((1,1,1))
ax.view_init(elev = 20, azim = -45)
ax.set_xlabel(r'$s$ [kpc]')
ax.set_ylabel(r'$\varpi_t$ [kpc]')
#ax.set_zlabel(r'$\varpi$')
ax.set_zticklabels('')
ax.dist = dist

# Perspective view of CO
ax = fig.add_subplot(2,2,4, projection='3d')
plot_cones_perspective(ax, lim, rlaunch,
                       theta_CO_N_in, theta_CO_N_out, phi_CO_N,
                       theta_CO_S_in, theta_CO_S_out, phi_CO_S, color='C1')
ax.set_xlim3d(lim)
ax.set_ylim3d(lim)
ax.set_zlim3d(lim)
ax.set_box_aspect((1,1,1))
ax.view_init(elev = 20, azim = -45)
ax.set_xlabel(r'$s$')
ax.set_ylabel(r'$\varpi_t$ [kpc]')
ax.set_zlabel(r'$\varpi$ [kpc]')
ax.dist = dist

# Add legend
fig.legend(loc='upper center', ncol=3, prop={'size': 10})

# Adjust spacing
plt.subplots_adjust(left=0.13, right=0.87, hspace=0.15, top=0.93)

# Save
plt.savefig('windshape.pdf')
