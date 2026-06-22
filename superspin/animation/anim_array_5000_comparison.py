# Animating the evolution of the vortex array in a neutron star - Anantharaman S V, Ashoka University
# To be used alongside the code for stabilization and dynamics of superfluid vortices in a neutron star
# Version 1 of code completed - June 22, 2026, 04:00:00

# open terminal and go to a folder of your choice
# Run command: manim -pqm script_name.py MyScene

from manim import *
import pandas as pd
import numpy as np

class MyScene(Scene):
    def construct(self):
        # Add title
        Title = "Vortex array evolution in neutron stars with weak pinning (left) and strong pinning (right)"
        Subtitle = "20260401/setB/run1 and 20260401/setA/run1"

        title = Text(Title).scale(0.4).move_to([0,11/3,0])
        self.add(title)

        subtitle = Text(Subtitle).scale(0.2).move_to([0,10/3,0])
        self.add(subtitle)

        # Declare times of interest
        t_interest = range(0,2001,20)

        # Animate
        for t in t_interest:
            # Adding elements
            # Timestamp
            Time = "Time = " + str(round(t,1)) + " $T_0$"
            time = Tex(Time).scale(0.7).move_to([0,-10/3,0])
            self.add(time)

            # Vortices - First star
            dots1 = []
            nv = 5000 # Number of vortices
            resolution = 20 # Number of datapoints per T_0
            # Data relevant to the current slice of time
            location = '/mnt/usb-Realtek_RTL9210_NVME_012345679989-0:0-part2/backup_work/vortex/runs/20260401/setB/run1/sim_vortex_pos.dat'
            data = np.loadtxt(location, delimiter='\t', skiprows=1+(t*resolution), max_rows=1)
            data_x = data[1:nv+1]
            data_y = data[nv+1:(2*nv)+1]
            data_z = 0
            # Making the data fit the screen
            shrink = 0.3
            data_x = data_x*shrink
            data_y = data_y*shrink
            data_z = data_z*shrink
            # Add dots
            for i in range(0,nv,1):
                dots1.append(Dot(point=[data_x[i]-(11/3),data_y[i],data_z], radius=0.01, color=GOLD))
            self.add(*dots1)
            
            # Vortices - Second star
            dots2 = []
            # Data relevant to the current slice of time
            location = '/mnt/usb-Realtek_RTL9210_NVME_012345679989-0:0-part2/backup_work/vortex/runs/20260401/setA/run1/sim_vortex_pos.dat'
            data = np.loadtxt(location, delimiter='\t', skiprows=1+(t*resolution), max_rows=1)
            data_x = data[1:nv+1]
            data_y = data[nv+1:(2*nv)+1]
            data_z = 0
            # Making the data fit the screen
            shrink = 0.3
            data_x = data_x*shrink
            data_y = data_y*shrink
            data_z = data_z*shrink
            # Add dots
            for i in range(0,nv,1):
                dots2.append(Dot(point=[data_x[i]+(11/3),data_y[i],data_z], radius=0.01, color=GOLD))
            self.add(*dots2)
            
            # Wait
            self.wait(0.2)
            
            # Removing elements
            self.remove(time)
            self.remove(*dots1)
            self.remove(*dots2)
            
