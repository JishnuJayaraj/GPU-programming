# Molecular Dynamics

This projects simulate Velocity verlet molecular dynamics. The task is to simulate trajectories on ensemble of N given particles based on physical principles of Newtons law of motion and Lennard Jones potential.


The program has following functionality

1. Particels modeled n 3dimensional space. with each particle having velocity, position, acceleration and associated force. 
2. Instead of using brute force nearest neighbour a binning or linked cell algortihm is implimented for scalability. 
3. The results are saved as vtk files that can be viewed using paraview.
