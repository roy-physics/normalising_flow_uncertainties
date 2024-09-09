import numpy as np
from galpy.potential import MiyamotoNagaiPotential
from galpy.orbit import Orbit
import argparse
import matplotlib.pyplot as plt

print('Modules loaded :)')

# Create an ArgumentParser object
parser = argparse.ArgumentParser(description="Generate particles in a potential")

# Add arguments
parser.add_argument('num_particles', type=int, help='Number of particles to generate')
parser.add_argument('filename_extension', type=str, help='Filename extension for saving the file')
parser.add_argument('random_seed', type=int, help='Random number seed')

# Parse the command-line arguments
args = parser.parse_args()

# Access the arguments
num_particles = args.num_particles
filename_extension = args.filename_extension
random_seed = args.random_seed

# Print the arguments to verify (optional)
print(f"Number of particles: {num_particles}")
print(f"Filename extension: {filename_extension}")
print(f"Random seed: {random_seed}")

# Now you can use num_particles, filename_extension, and random_seed in your script
np.random.seed(random_seed)

# Miyamoto-Nagai params
Rscale=1.0
zscale=0.1
mn_potential = MiyamotoNagaiPotential(amp=1., a=Rscale, b=zscale)

# Sample radial positions R from an exponential distribution
#sampled_R = -Rscale * np.log(np.random.uniform(0, 1, num_particles))
u = np.random.uniform(0, 1, num_particles)
sampled_R = -Rscale * np.log(1 - u + u * np.exp(-1))  # Sampling from the correct distribution
# Sample vertical positions z from a symmetric exponential distribution
sampled_z = -zscale * np.log(np.random.uniform(0, 1, num_particles))
sampled_z *= np.random.choice([-1, 1], num_particles)  # randomly flip sign to account for symmetry

"""
# Parameters for the Miyamoto-Nagai potential
a = Rscale  # radial scale length in kpc
b = zscale  # vertical scale height in kpc
M = 1.0  # mass in arbitrary units

# Define the density function for the Miyamoto-Nagai profile
def miyamoto_nagai_density(R, z, a, b, M):
    b2 = b**2
    z2b2 = np.sqrt(z**2 + b2)
    denominator = (R**2 + (a + z2b2)**2)**(5/2) * (z**2 + b2)**(3/2)
    numerator = a * R**2 + (a + 3 * z2b2) * (a + z2b2)
    return (b2 * M / (4 * np.pi)) * numerator / denominator

# Sampling bounds
R_max = 10.0  # kpc, maximum radial distance
z_max = 5.0 * b  # limit for vertical distance, typically a few scale heights

# Number of particles to sample
#num_particles = 1000

# Rejection sampling
sampled_R = []
sampled_z = []

# Find the maximum density (use the center of the potential R=0, z=0)
rho_max = miyamoto_nagai_density(0, 0, a, b, M)

while len(sampled_R) < num_particles:
    # Generate candidate positions
    R_cand = np.random.uniform(0, R_max)
    z_cand = np.random.uniform(-z_max, z_max)
    
    # Calculate the density at the candidate position
    rho_cand = miyamoto_nagai_density(R_cand, z_cand, a, b, M)
    
    # Accept or reject based on a random uniform comparison
    if rho_cand / rho_max > np.random.uniform(0, 1):
        sampled_R.append(R_cand)
        sampled_z.append(z_cand)

# Print a few values to check
for i in range(5):
    print(f"Particle {i+1}: R = {sampled_R[i]:.2f} kpc, z = {sampled_z[i]:.2f} kpc")"""


# Convert lists to arrays
sampled_R = np.array(sampled_R)
sampled_z = np.array(sampled_z)

phi = np.random.uniform(0, 2*np.pi, num_particles)  # Azimuthal angles

# Convert sampled_R and phi into Cartesian coordinates
x = sampled_R * np.cos(phi)
y = sampled_R * np.sin(phi)
z = sampled_z

# Compute densities at these positions (this step is usually for validation, not sampling)
#densities = mn_potential.dens(R, z)

# Calculate the circular velocity at each R
#v_c = mn_potential.vcirc(sampled_R)
# Calculate the spherical radial force and effective circular velocity at (R, z)
v_c = np.sqrt(-mn_potential.rforce(sampled_R, sampled_z) * sampled_R)

# Generate delta_1, delta_2, delta_3 from a normal distribution
delta_1 = np.random.normal(0, 1, num_particles)
delta_2 = np.random.normal(0, 1, num_particles)
delta_3 = np.random.normal(0, 1, num_particles)

"""
# Compute the velocity components
v_R = 0.05 * v_c * delta_1
v_T = v_c * (1 - 0.1 * np.abs(delta_2))
v_z = 0.05 * v_c * delta_3
"""

"""# Radial vector (x, y, z)
#r_vec = np.array([x, y, z])
r_vec = np.vstack((x, y, z)).T  # Shape (num_particles, 3)

# Unit vector along z-axis
k_vec = np.array([0, 0, 1])

# Compute the tangential velocity unit vector using cross product
v_T_unit = np.cross(k_vec, r_vec)

# Normalize the tangential velocity unit vector
v_T_unit_norm = np.linalg.norm(v_T_unit, axis=0)  # Compute the norm (magnitude)
v_T_unit = v_T_unit / v_T_unit_norm  # Normalize

# Now v_T_unit is the unit vector in the plane connecting the particle and the origin

# Tangential velocity in the orbital plane
v_T = v_c * (1 - 0.1 * np.abs(delta_2))

# v_R in the orbital plane
v_R = 0.05 * v_c * delta_1

# Now we calculate the tangential velocity along the proper plane
# Compute the normalized direction vector from the origin to the particle's position
r_norm = np.sqrt(x**2 + y**2 + z**2)

print(v_R.shape,x.shape,r_norm.shape,v_T.shape,v_T_unit.shape)

# Convert the tangential velocity back into Cartesian components
vx = v_R * (x / r_norm) + v_T * v_T_unit[:,0]
vy = v_R * (y / r_norm) + v_T * v_T_unit[:,1]
vz = v_R * (z / r_norm) + 0.05 * v_c * delta_3  # Small perturbation in vertical velocity"""

# Radial velocity perturbation
v_R = 0.05 * v_c * delta_1

# Now we calculate the tangential velocity along the proper plane
# Compute the normalized direction vector from the origin to the particle's position
r_norm = np.sqrt(x**2 + y**2 + z**2)

# Tangential velocity in the plane passing through the origin and the particle's position
v_T = v_c * (1 - 0.1 * np.abs(delta_2))

# Tangential velocity unit vector in the plane through the origin and particle
v_T_unit_x = -(y / np.sqrt(x**2 + y**2))  # x-component of tangential unit vector
v_T_unit_y = (x / np.sqrt(x**2 + y**2))   # y-component of tangential unit vector
#v_T_unit_x = -(y / r_norm)  # x-component of tangential unit vector
#v_T_unit_y = (x / r_norm)   # y-component of tangential unit vector
v_T_unit_z = np.zeros_like(v_T)            # No z-component for tangential velocity

# Convert the tangential velocity back into Cartesian coordinates
vx = v_R * (x / r_norm) + v_T * v_T_unit_x
vy = v_R * (y / r_norm) + v_T * v_T_unit_y
vz = v_R * (z / r_norm) + 0.05 * v_c * delta_3  # Small perturbation in vertical velocity"""

# Convert Cartesian velocities to cylindrical velocities
# Radial distance (R)
R = np.sqrt(x**2 + y**2)

# Azimuthal angle (phi)
#phi = np.arctan2(y, x)

# Radial velocity
v_R = (x * vx + y * vy) / R

# Tangential velocity
v_T = (-y * vx + x * vy) / R

# Vertical velocity remains the same
v_z = vz

# Stack initial conditions into a single array (shape: [num_orbits, 6])
initial_conditions = np.column_stack([sampled_R, v_R, v_T, sampled_z, v_z, phi])

# Print a few values for inspection
for i in range(5):  # Just print the first 5 particles
    print(f"Particle {i+1}: R = {sampled_R[i]:.2f}, z = {sampled_z[i]:.2f}, v_R = {v_R[i]:.2f}, v_T = {v_T[i]:.2f}, v_z = {v_z[i]:.2f}")  

## Integrate the orbits

# delete pre-existing orbits object
#del orbits

# Initialize the orbits with these initial conditions
orbits = Orbit(initial_conditions)

# Time array for integration
#times = np.linspace(0., 1000., 1000)
times = np.arange(0., 1000., 1.0)

# Integrate all orbits in the defined potential
orbits.integrate(times, mn_potential)

# Extract final positions in cylindrical coordinates
final_R = orbits.R(times[-1])
final_z = orbits.z(times[-1])
final_phi = orbits.phi(times[-1])

# Extract final positions in Cartesian coordinates
final_x = orbits.x(times[-1])
final_y = orbits.y(times[-1])
final_z_cartesian = orbits.z(times[-1])  # z is the same in both systems

# Extract final velocities in cylindrical coordinates
final_vR = orbits.vR(times[-1])   # Radial velocity
final_vz = orbits.vz(times[-1])   # Vertical velocity
final_vT = orbits.vT(times[-1])   # Tangential velocity (in cylindrical coordinates)

# Extract final velocities in cylindrical coordinates
final_vx = orbits.vx(times[-1])
final_vy = orbits.vx(times[-1])
# final_vz is the same in either coordinate frame


## Save output to an array

bodies = np.zeros((num_particles, 7))  # Each row: mass, x, y, z, vx, vy, vz
bodies[:, 0] = 1.0 / num_particles  # Set mass, assumed to be equal for simplicity

# Convert cylindrical coordinates (sampled_R, sampled_z, phi) to Cartesian positions (x, y, z)
bodies[:, 1] = final_x  # x
bodies[:, 2] = final_y  # y
bodies[:, 3] = final_z  # z

bodies[:, 4] = final_vx  # x
bodies[:, 5] = final_vy  # y
bodies[:, 6] = final_vz  # z

np.save(f'./MN_{num_particles}_particles_iteration_{filename_extension}.npy', bodies)


print("Particle phase space saved :)")

## Plotting the final distribution to check/debug
from matplotlib.colors import LogNorm

plt.figure(figsize=(10, 5))

plt.subplot(2, 1, 1)

nbins = 200
bins = [np.linspace(-1.,1.,nbins),np.linspace(-0.5,0.5,nbins)]

plt.hist2d(final_x, final_z, bins=bins, cmap=plt.cm.magma, norm=LogNorm()); plt.colorbar(label='Counts')
plt.xlabel('x [kpc]')
plt.ylabel('z [kpc]')
plt.title('Final x-z Distribution')
plt.xlim([bins[0][0],bins[0][-1]])
plt.ylim([bins[-1][0],bins[-1][-1]])


plt.subplot(2, 1, 2)

nbins = 200
bins = [np.linspace(-0.3,0.3,nbins),np.linspace(-0.7,0.7,nbins)]
plt.hist2d(final_vR, final_vz, bins=bins, cmap=plt.cm.magma, norm=LogNorm()); plt.colorbar(label='Counts')
plt.xlabel('vR')
plt.ylabel('vz')
plt.title('Final vR-vz Distribution')


plt.tight_layout()


plt.savefig('./MN_disk_test.png', dpi=300)  # Save with 300 dpi for higher quality



