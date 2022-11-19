import meep as mp
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from meep.materials import Si
from meep.materials import SiO2

# Variables for Geometry
um_scale = 1.0
wvl = 1.55
wg_height = 0.22
w_MMI = 3.25
l_MMI = 14.06
w_Input = 0.5
w_Output = 1.7
l_Taper = 5.7
separation = 0.3
l_Access = 3
w_Absorber = 1
end_of_input = l_MMI/2 + l_Taper+ 0.5
buffer = wvl * 3

l_cell = l_MMI + 2*l_Taper + 2*l_Access + 2*w_Absorber + buffer
w_cell = w_MMI + 2*w_Absorber + buffer
h_cell = wg_height + 2*w_Absorber + buffer

taper_y1 = w_Output/2 + w_Input/2 + separation/2
taper_y2 = w_Output/2 - w_Input/2 + separation/2
output_y1 = separation/2
output_y2 = output_y1 + w_Output
taper_x1 = l_MMI/2
taper_x2 = l_MMI/2 + l_Taper
wg_end_x = taper_x2 + 0.5

wg_center_y = w_Output/2 + separation/2
l_wg = l_cell/2 - taper_x2
wg_center_x = l_MMI/2 + l_Taper + l_wg/2

# Grating variables- Takes in the grating duty cycle (gdc) and the period, calculates the width of the grating from this
gdc = 0.5       # grating duty cycle
period = 0.19      # 190nm = period of the grating      
l_Grating = l_Taper *2 + l_MMI
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
geometry_straight = [mp.Block(mp.Vector3(l_cell, w_cell, wg_height), center= mp.Vector3(0,0,0), material=SiO2),
                        mp.Block(mp.Vector3(l_cell, w_Input, wg_height), center=mp.Vector3(0,-wg_center_y,0), material=Si)]
#%%
#Adding the basic geometry (access waveguides, tapers, MMI body)
geometry = [mp.Block(mp.Vector3(l_cell, w_cell, wg_height), center = mp.Vector3(0,0,0), material=SiO2),
               mp.Block(mp.Vector3(l_MMI, w_MMI, wg_height), center=mp.Vector3(0,0,0), material=Si),
               mp.Block(mp.Vector3(l_wg, w_Input, wg_height), center=mp.Vector3(-wg_center_x, wg_center_y,0), material=Si),
               mp.Block(mp.Vector3(l_wg, w_Input, wg_height), center=mp.Vector3(-wg_center_x, -wg_center_y,0), material=Si),
               mp.Block(mp.Vector3(l_wg, w_Input, wg_height), center=mp.Vector3(wg_center_x, wg_center_y,0), material=Si),
               mp.Block(mp.Vector3(l_wg, w_Input, wg_height), center=mp.Vector3(wg_center_x, -wg_center_y,0), material=Si),
               mp.Prism(vertices=[mp.Vector3(-taper_x2, taper_y1, 0), mp.Vector3(-taper_x2, taper_y2, 0), mp.Vector3(-taper_x1, output_y1, 0), mp.Vector3(-taper_x1, output_y2, 0)], height= wg_height, material=Si),
               mp.Prism(vertices=[mp.Vector3(-taper_x2, -taper_y1, 0), mp.Vector3(-taper_x2, -taper_y2, 0), mp.Vector3(-taper_x1, -output_y1, 0), mp.Vector3(-taper_x1, -output_y2, 0)], height= wg_height, material=Si),
               mp.Prism(vertices=[mp.Vector3(taper_x2, taper_y1, 0), mp.Vector3(taper_x2, taper_y2, 0), mp.Vector3(taper_x1, output_y1, 0), mp.Vector3(taper_x1, output_y2, 0)], height= wg_height, material=Si),
               mp.Prism(vertices=[mp.Vector3(taper_x2, -taper_y1, 0), mp.Vector3(taper_x2, -taper_y2, 0), mp.Vector3(taper_x1, -output_y1, 0), mp.Vector3(taper_x1, -output_y2, 0)], height= wg_height, material=Si)]
#Adding the grating
for j in range(int(g_num)):
    g_center = l_MMI/2 + l_Taper - g_width/2 - j*(period)
    geometry.append(mp.Block(size=mp.Vector3(g_width, output_y2*2, wg_height), center=mp.Vector3(g_center, 0, 0), material=SiO2))
#Adding the internal tapers
wg_y1 = w_Output/2-w_Input/2+separation/2
wg_y2 = w_Output/2+w_Input/2+separation/2
geometry.append(mp.Prism(vertices=[mp.Vector3(taper_x1, wg_center_y, 0), mp.Vector3(taper_x2,wg_y1,0), mp.Vector3(taper_x2, wg_y2,0)], height=wg_height, material= Si))
geometry.append(mp.Prism(vertices=[mp.Vector3(taper_x1, -wg_center_y, 0), mp.Vector3(taper_x2,-wg_y1,0), mp.Vector3(taper_x2,-wg_y2,0)], height=wg_height, material= Si))
geometry.append(mp.Prism(vertices=[mp.Vector3(-taper_x1, wg_center_y, 0), mp.Vector3(-taper_x2, wg_y1,0), mp.Vector3(-taper_x2, wg_y2,0)], height=wg_height, material= Si))
geometry.append(mp.Prism(vertices=[mp.Vector3(-taper_x1, -wg_center_y, 0), mp.Vector3(-taper_x2,-wg_y1,0), mp.Vector3(-taper_x2,-wg_y2,0)], height=wg_height, material= Si))

#%%
Absorber_layers = [mp.Absorber(w_Absorber)]

# Source-- Eigenmode source must extend outside of the waveguide-- extended it by a wvl, so half on each side
sources = [mp.EigenModeSource(src= mp.GaussianSource(freq, fwidth=fwidth), center= mp.Vector3(-end_of_input-0.25, -wg_center_y, 0), size=mp.Vector3(0, w_Input+wvl, 2.8),eig_parity=mp.ODD_Z+mp.EVEN_Y, direction=mp.X)]

# Monitor positions for each waveguide
monLo_center = mp.Vector3(wg_end_x, wg_center_y,0)
monHi_center = mp.Vector3(wg_end_x, -wg_center_y,0)

mon_size = mp.Vector3(0, w_Input+wvl, wg_height+wvl)
#%%
# Normalization simulation
sim = mp.Simulation(cell_size=cell,
                    geometry=geometry_straight,
                    boundary_layers=Absorber_layers,
                    sources=sources,
                    default_material=SiO2,
                    resolution=resolution,
                    eps_averaging=False
)
#incident_mode = sim.add_mode_monitor(freq, fwidth, nfreqs, mp.ModeRegion(center= mp.Vector3(-end_of_input,-wg_center_y,0), size=mon_size, direction=mp.Z))
incident_flux = sim.add_flux(freq, fwidth, nfreqs, mp.FluxRegion(center= mp.Vector3(-end_of_input,-wg_center_y,0), size=mon_size, direction=mp.Z))

sim.run(until= mp.stop_when_fields_decayed(dt=50, c=mp.Ez, pt=monHi_center, decay_by=1e-4))

# This will have everything coming out of the source so can be used as denominator
#incident_mode_alpha = sim.get_eigenmode_coefficients(incident_mode,[1],eig_parity=mp.ODD_Z+mp.EVEN_Y, direction=mp.Z).alpha[0,:,0]

incident_flux_data = sim.get_flux_data(incident_flux)

#%%
sim.reset_meep()
#%%
# Simulation with the scattering geometry
sim = mp.Simulation(cell_size=cell,
                    geometry=geometry,
                    boundary_layers=Absorber_layers,
                    sources=sources,
                    default_material=SiO2,
                    resolution=resolution,
                    eps_averaging=False
)

#Monitors for the input ports
input_flux = sim.add_flux(freq, fwidth, nfreqs, mp.FluxRegion(center= mp.Vector3(-end_of_input,-wg_center_y,0), size=mon_size, direction=mp.Z))
input_hi_flux = sim.add_flux(freq, fwidth, nfreqs, mp.FluxRegion(center= mp.Vector3(-end_of_input,wg_center_y,0), size=mon_size, direction=mp.Z))
#Monitors for the output ports
output_lo_flux= sim.add_flux(freq, fwidth, nfreqs, mp.FluxRegion(center= monLo_center, size=mon_size, direction=mp.Z))
output_hi_flux= sim.add_flux(freq, fwidth, nfreqs, mp.FluxRegion(center= monHi_center, size=mon_size, direction=mp.Z))
#%%
# Load minus flux data before running the scattering simulation
sim.load_minus_flux_data(input_flux, incident_flux_data)
#%%
sim.run(until= mp.stop_when_fields_decayed(dt=50,c=mp.Ez, pt=monHi_center, decay_by=1e-4))
#%%
#Finding the forward eigenmode coefficients for the input port with the source and the two output ports
outputLoFluxData = sim.get_fluxes(output_lo_flux)
outputHiFluxData = sim.get_fluxes(output_hi_flux)
#%%
inputLoFluxData = sim.get_fluxes(input_flux)
inputHiFluxData = sim.get_fluxes(input_hi_flux)
#%%
# Calculate the transmittance for the upper and lower output ports
trans_Lo = abs(outputLoFluxData[:])**2/abs(incident_flux[:])**2
trans_Hi = abs(outputHiFluxData[:])**2/abs(incident_flux[:])**2
#%%
# Calculate reflectance for the input ports
refl_Lo = abs(inputLoFluxData[:])**2/abs(incident_flux[:])**2
refl_Hi = abs(inputHiFluxData[:])**2/abs(incident_flux[:])**2

freqs= mp.get_flux_freqs(input_mode)

# Shows the electric field in the z direction
ez_data = sim.get_array(center=mp.Vector3(), size=mp.Vector3(l_cell, w_cell, 0), component=mp.Ez)
plt.imsave("MMIEzFields.png", ez_data.transpose(), cmap='RdBu')

#%%
# Plotting the throughput
freqs = np.array(freqs)
wvls = 1/freqs[:]
g = np.exp((-(freqs[:]-freq)**2/fwidth**2))

fig, ax = plt.subplots()
ax.plot(wvls[:],trans_Lo, color='blue')
ax.set_xlabel('Wavelength (um)')
fig.savefig("TransmissionLow.png")

fig, ax = plt.subplots()
ax.plot(wvls[:],trans_Hi, color='red')
ax.set_xlabel('Wavelength (um)')
fig.savefig("TransmissionHigh.png")

#%%
# Plotting the reflectance
fig, ax = plt.subplots()
ax.plot(wvls[:], refl_Lo, color='blue')
ax.set_xlabel('Wavelength (um)')
fig.savefig("ReflectanceLow.png")

fig, ax = plt.subplots()
ax.plot(wvls[:], refl_Hi, color='red')
ax.set_xlabel('Wavelength (um)')
fig.savefig("ReflectanceHigh.png")

# Plot the phase
fig, ax = plt.subplots()
ax.plot(wvls[:], outputLoPhase, color='blue')
ax.plot(wvls[:], outputHiPhase, color='red')
ax.set_xlabel('Wavelength (um)')
fig.savefig("Phase.png")