# Usage notes for SUPERSPIN
# Neutron star superfluid vortex simulator
# Anantharaman S V
# Ashoka University
# August 20, 2025

# Context
The simulator described here was developed to aid the research undertaken during my PhD.
It is by no means a complete description of the state within a neutron star, much of which lies on the frontiers of current physics research.

# Credits
The conceptual idea of the simulator was first established by Howitt et al. (2020). https://doi.org/10.1093/mnras/staa2314
Discussions with Dr. George Howitt were of great help in the initial stages of construction.
The current version of the simulator has come a far way from the original by George.
Much of the augmentation was discussed in depth with Prof. Dipankar Bhattacharya, my doctoral advisor.
His patience and incisive comments provided the scaffold upon with this work was built.

# Brief idea
Rotating neutron stars are expected to have superfluid interiors.
Superfluids carry rotation by means of quantized vortices.
The rotational evolution of the star is determined by the interaction of quantized vortices with other constituents present within.
In the crust of the star, a lattice of nuclei is expected to pin vortices.
In the core of the star, flux tubes may have a similar effect.
As the star slows down, the superfluid within also tries to reduce its speed by removing some vortices.
But pinning obstructs this.
When the speed difference between the crust and the superfluid exceeds a threshold, some of the vortices could overcome pinning and move out.
This leads to a transfer of angular momentum to the crust, resulting in an increase in its rotation rate.
Such an increase is referred to as a glitch.
The simulator is designed to test this idea using a two dimensional model of the star and allow for extensions as required.

# Files
The script for running the codes, 'runscript.sh', is hosted in the root folder.
The code files are hosted in 'codes'.
The compiled files are stored in 'compiles'
The output gets generated in 'output'.
Output of interest is to be archived in 'runs' for temporary storage. An example archive has left in place.
The code for animation is placed in 'animation'.
A jupyter notebook for analysis, 'analysis.ipynb', is available in 'saves'.

# Running the script -- Requires 'gcc'
Move to the root folder in terminal and run the following command
./runscript.sh

# Accessing the analysis
Move to the 'saves' folder in terminal and run the following command
jupyter lab

# Usage -- Main code -- Requires C++ library 'boost' placed at the directory mentioned in the runscript
The parameters applicable to the system are explained and defined in the 'parameters.h' file.
They are to be edited directly in the file and saved, before running the script.

# Usage -- Animation code -- Requires python package 'manim' placed at the directory mentioned in the runscript
The animation code for 2000 vortices and 10000 vortices are saved as different files.
The essential change between the two are the sizes of the various display elements, chosen for best visual appeal.
The directory of the data files, the duration to animate, and the title of the animation are all set within the code.
It is recommended that no more than one glitch be animated at a given time.
The animation code is not parallelised and hence quite slow.

# Usage -- Analysis code -- Requires python packages 'pandas', 'jupyter', 'matplotlib', 'numpy', 'seaborn'


# License -- Unlicense
This is free and unencumbered software released into the public domain.

Anyone is free to copy, modify, publish, use, compile, sell, or
distribute this software, either in source code form or as a compiled
binary, for any purpose, commercial or non-commercial, and by any
means.

In jurisdictions that recognize copyright laws, the author or authors
of this software dedicate any and all copyright interest in the
software to the public domain. We make this dedication for the benefit
of the public at large and to the detriment of our heirs and
successors. We intend this dedication to be an overt act of
relinquishment in perpetuity of all present and future rights to this
software under copyright law.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE AUTHORS BE LIABLE FOR ANY CLAIM, DAMAGES OR
OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
OTHER DEALINGS IN THE SOFTWARE.

For more information, please refer to <https://unlicense.org>
