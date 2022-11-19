#%%
import meep as mp
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

Si = mp.Medium(epsilon=3.4)
SiO2 = mp.Medium(epsilon=1.44)

# Variables for Geometry
um_scale = 1.0
wvl = 1.55
wg_height = 0.22
w_Input = 0.5
w_Output = 1.7
l_Taper = 5.7
l_wg = 3
w_PML = 1
buffer = wvl * 3

l_cell = l_Taper + 2*l_wg
w_cell = w_Output + 2*w_PML + buffer
h_cell = wg_height + 2*w_PML + buffer

taper_y1 = w_Output/2
taper_y2 = w_Input/2
taper_x = l_Taper/2

wg_center_x = l_Taper + l_wg/2

# Grating variables- Takes in the grating duty cycle (gdc) and the period, calculates the width of the grating from this
gdc = 0.5       # grating duty cycle
period = 0.19      # 190nm = period of the grating      
l_Grating = l_Taper
g_num = l_Grating / period
g_width = (gdc*l_Grating)/g_num

# Set up frequency range
freq= 1/wvl
fwidth= 0.2*freq
nfreqs = int(wvl*1000*0.2)

# Resolution so that there are over 10 pixels per wvl
resolution = 80

## Geometry-- MMI centered at the origin
cell = mp.Vector3(l_cell, w_cell, h_cell)

#%%
# Waveguide across the cell, centered at the source's y position
geometry_straight = [mp.Block(mp.Vector3(l_cell, w_Input, wg_height), center=mp.Vector3(0,0,0), material=Si)]
#%%
#Adding the basic geometry (access waveguides, tapers, MMI body)
geometry = [mp.Block(mp.Vector3(l_wg, w_Input, wg_height), center=mp.Vector3(-wg_center_x, 0, 0), material=Si),
               mp.Block(mp.Vector3(l_wg, w_Output, wg_height), center=mp.Vector3(wg_center_x, 0, 0), material=Si),
               mp.Prism(vertices=[mp.Vector3(-taper_x, taper_y1, 0), mp.Vector3(-taper_x, taper_y2, 0), mp.Vector3(-taper_x, w_Output/2, 0), mp.Vector3(-taper_x, -w_Output/2, 0)], height= wg_height, material=Si)]
#Adding the grating
for j in range(int(g_num)):
    g_center = l_Taper - g_width/2 - j*(period)
    geometry.append(mp.Block(size=mp.Vector3(g_width, w_Output, wg_height), center=mp.Vector3(g_center, 0, 0), material=SiO2))
#Adding the internal tapers
wg_y1 = w_Output/2-w_Input/2
wg_y2 = w_Output/2+w_Input/2
geometry.append(mp.Prism(vertices=[mp.Vector3(taper_x, 0, 0), mp.Vector3(taper_x,wg_y1,0), mp.Vector3(taper_x, wg_y2,0)], height=wg_height, material= Si))

#%%
PML_layers = [mp.PML(w_PML)]

# Source-- Eigenmode source must extend outside of the waveguide-- extended it by a wvl, so half on each side
sources = [mp.EigenModeSource(src= mp.GaussianSource(freq, fwidth=fwidth), center= mp.Vector3(-l_Taper/2-0.5, 0, 0), size=mp.Vector3(0, w_Input+wvl, wvl*2),eig_parity=mp.ODD_Z+mp.EVEN_Y, direction=mp.X)]

# Monitor positions for each waveguide
mon_center = mp.Vector3(-(l_Taper/2+0.25), 0, 0)
mon_out_center = mp.Vector3((l_Taper/2+0.25), 0, 0)

mon_size = mp.Vector3(0, w_Input+wvl, wg_height+wvl)
#%%
# Normalization simulation
sim = mp.Simulation(cell_size=cell,
                    geometry=geometry_straight,
                    boundary_layers=PML_layers,
                    sources=sources,
                    default_material=SiO2,
                    resolution=resolution,
                    eps_averaging=False
)
incident_mode = sim.add_mode_monitor(freq, fwidth, nfreqs, mp.ModeRegion(center= mp.Vector3(mon_center), size=mon_size, direction=mp.Z))

sim.run(until= 500)

# This will have everything coming out of the source so can be used as denominator
incident_mode_alpha = sim.get_eigenmode_coefficients(incident_mode,[1],eig_parity=mp.ODD_Z+mp.EVEN_Y, direction=mp.Z).alpha[0,:,0]

incident_flux_data = sim.get_flux_data(incident_mode)

#%%
# Show the simulation
plt.figure()
sim.plot3D()
plt.axis("off")
plt.savefig("StraightGeometry.png")

# Shows the electric field in the z direction
ez_data = sim.get_array(center=mp.Vector3(), size=cell, component=mp.Ez)
plt.imsave("StraightEzFields.png", ez_data.transpose(), cmap='RdBu')

#%%
sim.reset_meep()
#%%
# Simulation with the scattering geometry
sim = mp.Simulation(cell_size=cell,
                    geometry=geometry,
                    boundary_layers=PML_layers,
                    sources=sources,
                    default_material=SiO2,
                    resolution=resolution,
                    eps_averaging=False
)

#Monitors for the input ports
input_mode = sim.add_mode_monitor(freq, fwidth, nfreqs, mp.ModeRegion(center= mon_center, size=mon_size, direction=mp.Z))
#Monitors for the output ports
output_mode= sim.add_mode_monitor(freq, fwidth, nfreqs, mp.ModeRegion(center= mon_out_center, size=mon_size, direction=mp.Z))
#%%
# Load minus flux data before running the scattering simulation
sim.load_minus_flux_data(input_mode, incident_flux_data)
#%%
sim.run(until=500)
#%%
#Finding the forward eigenmode coefficients for the input port with the source and the two output ports
outputAlpha = sim.get_eigenmode_coefficients(output_mode,[1],eig_parity=mp.ODD_Z+mp.EVEN_Y, direction=mp.Z).alpha[0,:,0]
#%%
inputAlpha = sim.get_eigenmode_coefficients(input_mode,[1],eig_parity=mp.ODD_Z+mp.EVEN_Y, direction=mp.Z).alpha[0,:,1]
#%%
# Calculate the transmittance for the upper and lower output ports
trans = abs(outputAlpha[:])**2/abs(incident_mode_alpha[:])**2
#%%
# Calculate reflectance for the input ports
refl = abs(inputAlpha[:])**2/abs(incident_mode_alpha[:])**2

freqs= mp.get_flux_freqs(input_mode)

# Create dataframe to store alpha coefficients, transmission, reflectance, and phase
df = pd.DataFrame({"Frequency": freqs, "StraightAlpha": incident_mode_alpha, "InputAlpha": inputAlpha, "Transmission": trans})
df.to_csv("MMIOutput.csv")

#%%
# Show the simulation
plt.figure()
sim.plot3D()
plt.axis("off")
plt.savefig("MMIGeometry3D.png")

# Shows the electric field in the z direction
ez_data = sim.get_array(center=mp.Vector3(), size=cell, component=mp.Ez)
plt.imsave("MMIEzFields.png", ez_data.transpose(), cmap='RdBu')

#%%
# Plotting the throughput
freqs = np.array(freqs)
wvls = 1/freqs[:]
g = np.exp((-(freqs[:]-freq)**2/fwidth**2))

fig, ax = plt.subplots()
ax.plot(wvls[:],trans, color='blue')
ax.set_xlabel('Wavelength (um)')
fig.savefig("TransmissionLow.png")

#%%
# Plotting the reflectance
fig, ax = plt.subplots()
ax.plot(wvls[:], refl, color='red')
ax.set_xlabel('Wavelength (um)')
fig.savefig("ReflectanceHigh.png")