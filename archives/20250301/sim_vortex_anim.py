# Animating the dynamics of superfluid vortices in a neutron star - Anantharaman S V, Ashoka University
# To be used alongside the code for 'Stabilization and dynamics of superfluid vortices in a neutron star 

# To animate superfluid vortices in a neutron star 

# Version 1 of code completed - May 29, 2024, 13:00:00
# Version 2 of code completed - Sep 21, 2024, 00:00:00

# Run command: manim -pqm script_name.py MyScene

from manim import *
import pandas as pd
import numpy as np

class MyScene(Scene):
	def construct(self):
		Title = "Progression of a stresswave glitch"
		Subtitle = "stresswave/run1 - 87.25 to 91.25"

		nv = 2000 # Number of vortices
		R = 10 # Radius of the star
		Xi = 0.0125 # Pinning radius
		a = 0.1253 # Lattice spacing of pinning sites

		epsilon = a # A vortex is considered to have moved in the chosen slice of time if its displacement is greater than epsilon

		t_start = 87.25 # Start time of the glitch/feature of interest
		t_end = 91.25 # End time of the glitch/feature of interest

		# Reading the complete output data obtained from the simulation (sim_vortex_pos.dat or sim_vortex_pos_minimal.dat)
		data = pd.read_csv(r'D:\Codes\Vortex\runs\paper1\stresswave_40trig_thresh_0p7\run1\sim_vortex_pos_minimal.dat', delimiter='\t')
		omegacvst = pd.read_csv(r'D:\Codes\Vortex\runs\paper1\stresswave_40trig_thresh_0p7\run1\sim_omega_c.dat', delimiter='\t')

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
				dots_moving.append(Dot(point=[x[0,i]-11/3,y[0,i],z], radius=0.03, color=YELLOW)) # Coloured dots to indicate moving vortices
			else:
				index_fixed.append(i)
				dots_fixed.append(Dot(point=[x[0,i]-11/3,y[0,i],z], radius=0.03, color=GREY, fill_opacity=0.5)) # Grey fixed dots to indicate fixed vortices

		# Adding dots on screen
		self.add(*dots_moving)
		self.add(*dots_fixed)

		# Adding trace
		for dot in dots_moving:
			trace.append(TracedPath(dot.get_center, dissipating_time=1, stroke_opacity=[0, 1]))
		self.add(*trace)

		# Adding title and subtitle
		title = Text(Title).scale(0.4).move_to([0,11/3,0])
		self.add(title)

		subtitle = Text(Subtitle).scale(0.2).move_to([0,10/3,0])
		self.add(subtitle)

		# Adding axes
		rotmin = round(rot.min(),4)
		rotmax = round(rot.max(),4)
		rotstep = round((rot.max() - rot.min())/10,4)

		ax = Axes(x_range = (t_start, t_end,1), y_range = (rotmin-rotstep, rotmax+rotstep, rotstep), tips = True).add_coordinates().scale(0.5).move_to([11/3,0,0])
		x_lab = ax.get_x_axis_label("t~(T_{0})").scale(0.7)
		y_lab = ax.get_y_axis_label("\Omega_{c}~(\Omega_{0})").scale(0.7)
		self.add(ax,x_lab,y_lab)

		# Animating
		time_tot = data_select.shape[0]
		for t in range(1,time_tot,1):

			# Adding elements

			# Timestamp
			Time = "Time = " + str(round(times[t],1)) + " $T_0$"
			time = Tex(Time).scale(0.7).move_to([0,-10/3,0])
			self.add(time)

			# Graph
			graph = ax.plot_line_graph(omegacvst_select[0:(t*10),0], omegacvst_select[0:(t*10),1], add_vertex_dots=False, line_color = BLUE)
			self.add(graph)

			# *10 is used to reconcile the rotation rate data with the decimated position data

			# Moving vorties
			locations_moving = []
			for i in index_moving:
				locations_moving.append([x[t,i]-11/3,y[t,i],z])
			animations = [ApplyMethod(dot.move_to,location) for dot,location in zip(dots_moving,locations_moving)]
			self.play(*animations, rate_func=linear, run_time = 1)

			# Removing elements
			self.remove(time)
			self.remove(graph)