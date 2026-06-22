# To find the number of vortices that move in a given time interval 

from manim import *
import pandas as pd
import numpy as np

nv = 5000 # Number of vortices
R = 10 # Radius of the star
a = 0.056 # Lattice spacing of pinning sites

epsilon = a # A vortex is considered to have moved in the chosen slice of time if its displacement is greater than epsilon

t_start = 913.5 # Start time of the glitch/feature of interest
t_end = 917.5 # End time of the glitch/feature of interest

# Reading the complete output data obtained from the simulation (sim_vortex_pos.dat or sim_vortex_pos_minimal.dat)
data = pd.read_csv(r'~/ananth/superspin/runs/20260401/set4/run1/sim_vortex_pos.dat', delimiter='\t', skiprows = range(1,10*913), nrows = 10*10)
omegacvst = pd.read_csv(r'~/ananth/superspin/runs/20260401/set4/run1/sim_omega_c.dat', delimiter='\t')

# Restricting to the duration of a chosen glitch
data_select = data[(data['t/T_0']>t_start) & (data['t/T_0']<t_end)]
data_select = np.array(data_select)

omegacvst_select = omegacvst[(omegacvst['t/T_0']>t_start) & (omegacvst['t/T_0']<t_end)]
omegacvst_select = np.array(omegacvst_select)

times = data_select[:,0]
x = data_select[:,1:nv+1]
y = data_select[:,nv+1:(2*nv)+1]
z = 0
rot = omegacvst_select[:,1]

# Making the data fit the screen
shrink = 0.3 # The pinning radius should also be shrunk accordingly
x = x*shrink
y = y*shrink
z = z*shrink

# Initializing arrays
index_moving=[]
index_fixed=[]
dots_moving=[]
locations_moving=[]
dots_fixed=[]
trace=[]

# Finding the index of moving vortices
# A vortex is considered to move if it changes position by a distance greater than epsilon, in the given slice of time
for i in range(0,nv,1):
	delta_r = np.sqrt((x[-1,i]-x[0,i])**2 + (y[-1,i]-y[0,i])**2)
	if(delta_r > epsilon*shrink):
		index_moving.append(i)
		dots_moving.append(Dot(point=[x[0,i]-11/3,y[0,i],z], radius=0.01, color=YELLOW)) # Coloured dots to indicate moving vortices
	else:
		index_fixed.append(i)
		dots_fixed.append(Dot(point=[x[0,i]-11/3,y[0,i],z], radius=0.01, color=GREY, fill_opacity=0.5)) # Grey fixed dots to indicate fixed vortices
   
print("Number of moving vortices", len(dots_moving))
print("Total number of vortices", len(dots_moving) + len(dots_fixed))
