#include <cuda_runtime.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <cstddef>
#include <sys/time.h>
#include <boost/lexical_cast.hpp>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

using std::cout; using std::cin; using std::endl;
using std::string; using std::vector;

typedef double dType;

/********* generic functions **********************/

void checkError(cudaError_t err){
    if (err != cudaSuccess){
        std::cout << cudaGetErrorString(err) << std::endl;
        exit(-1);
    }
}

/********************* Exercise02*****************************/
__global__ void setForces(int nMols, dType* positions_x, dType* positions_y, dType* positions_z, dType* force_x, dType* force_y, dType* force_z, int epsilon, int sigma){
    /*sets forces based on given position of the particles*/
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < nMols){
        force_x[idx] = force_y[idx] = force_z[idx] = 0;
	    for(int i = 0; i < nMols; i++){
		    if (idx == i)
		            continue;
		    else {
		        double distance = sqrt(pow(positions_x[idx] - positions_x[i],2) + pow(positions_y[idx] - positions_y[i],2) + pow(positions_z[idx] - positions_z[i],2));
		        double constant = (24 * epsilon/pow(distance,2)) * pow(sigma/distance,6) * (2 * pow(sigma/distance,6) - 1);
		        force_x[idx] += constant * (positions_x[idx] - positions_x[i]);
		        force_y[idx] += constant * (positions_y[idx] - positions_y[i]);
		        force_z[idx] += constant * (positions_z[idx] - positions_z[i]);
		    }
	    }
    }
}

__global__ void updatePosition (int nMols,dType* masses,dType* positions_x,dType* positions_y,dType* positions_z,dType* velocities_x,dType* velocities_y,dType* velocities_z,dType* force_x,dType* force_y,dType* force_z,double timestep_length){
    /*updates the position of the particles for the new time step*/
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < nMols){
        positions_x[idx] += ( (timestep_length * velocities_x[idx]) + (force_x[idx] * pow(timestep_length,2) * 0.5 / masses[idx]) );
        positions_y[idx] += ( (timestep_length * velocities_y[idx]) + (force_y[idx] * pow(timestep_length,2) * 0.5 / masses[idx]) );
        positions_z[idx] += ( (timestep_length * velocities_z[idx]) + (force_z[idx] * pow(timestep_length,2) * 0.5 / masses[idx]) );
    }
}

__global__ void updateOldForces(int nMols, dType* force_x, dType* force_y, dType* force_z, dType* force_old_x, dType* force_old_y, dType* force_old_z) {
    /*sets old forces to new forces for use in consecutive step*/
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < nMols){
        force_old_x[idx] = force_x[idx];
        force_old_y[idx] = force_y[idx];
        force_old_z[idx] = force_z[idx];
    }
}

__global__ void updateVelocities(int nMols, dType* masses,dType* velocities_x, dType* velocities_y, dType* velocities_z, dType* force_x, dType* force_y, dType* force_z, dType* force_old_x, dType* force_old_y, dType* force_old_z, double timestep_length){
  /* updates new velocities at a time step */
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < nMols){
    velocities_x[idx] += ( (force_old_x[idx]  + force_x[idx]) * timestep_length * 0.5 / masses[idx] );
    velocities_y[idx] += ( (force_old_y[idx]  + force_y[idx]) * timestep_length * 0.5 / masses[idx] );
    velocities_z[idx] += ( (force_old_z[idx]  + force_z[idx]) * timestep_length * 0.5 / masses[idx] );
  }
}


/**********************************  Exercise 03  ********************************************/

__global__ void periodicBC(int nMols, dType* positions_x, dType* positions_y, dType* positions_z, float x_min, float x_max, float y_min, float y_max, float z_min, float z_max){
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < nMols){
    // X direction
    if(positions_x[idx] < x_min) {
      double offset = abs(x_min - positions_x[idx]);
      positions_x[idx] = x_max - offset;
    }
    if (positions_x[idx] > x_max) {
      double offset = abs(x_max - positions_x[idx]);
      positions_x[idx] = x_min + offset;
    }
    //Y direction
    if(positions_y[idx] < y_min) {
      double offset = abs(y_min - positions_y[idx]);
      positions_y[idx] = y_max - offset;
    }
    if (positions_y[idx] > y_max) {
      double offset = abs(y_max - positions_y[idx]);
      positions_y[idx] = y_min + offset;
    }
    //Z direction
    if(positions_z[idx] < z_min) {
      double offset = abs(z_min - positions_z[idx]);
      positions_z[idx] = z_max - offset;
    }
    if (positions_z[idx] > z_max) {
      double offset = abs(z_max - positions_z[idx]);
      positions_z[idx] = z_min + offset;
    }
  }
}
