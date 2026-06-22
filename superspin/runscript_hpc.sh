
#! /bin/bash
#PBS -N Sim_Vortex
#PBS -o out.log
#PBS -e err.log
#PBS -l host=compute3
#PBS -l ncpus=104
#PBS -q gpu

module load compiler/gcc-12.0
cd superspin/
g++ -c ./codes/main.cpp -Ofast -o ./compiles/main.o -fopenmp
g++ -c ./codes/initialization.cpp -Ofast -o ./compiles/initialization.o -fopenmp
g++ -c ./codes/utility.cpp -Ofast -o ./compiles/utility.o -fopenmp
g++ -c ./codes/integration.cpp -Ofast -o ./compiles/integration.o -I ~/boost_1_88_0 -fopenmp
g++ -c ./codes/support.cpp -Ofast -o ./compiles/support.o -fopenmp
g++ -c ./codes/writes.cpp -Ofast -o ./compiles/writes.o -fopenmp
g++ -c ./codes/pinning.cpp -Ofast -o ./compiles/pinning.o -fopenmp
g++ -c ./codes/triggers.cpp -Ofast -o ./compiles/triggers.o -fopenmp
g++ -c ./codes/glitchfinder.cpp -Ofast -o ./compiles/glitchfinder.o -fopenmp
g++ ./compiles/main.o ./compiles/initialization.o ./compiles/utility.o ./compiles/integration.o ./compiles/support.o ./compiles/writes.o ./compiles/pinning.o ./compiles/triggers.o ./compiles/glitchfinder.o -Ofast -o ./compiles/program -fopenmp
./compiles/program > output_terminal.txt
