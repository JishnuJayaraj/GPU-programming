#include <cuda_runtime.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <cstddef>
#include <sys/time.h>
#include <boost/lexical_cast.hpp>

#include "kernel.h"

using std::cout; using std::cin; using std::endl;
using std::string; using std::vector;

typedef double dType;

int main(int argc, char* argv[]) {
	//simulation parameters
	std::string 	part_input_file;
	double 			timestep_length;
	double 			time_end;
	double 			epsilon;
	double 			sigma;
	int 			part_out_freq;
	std::string 	part_out_name_base;
	int 			vtk_out_freq;
	std::string 	vtk_out_name_base;
	int 			cl_workgroup_1dsize;

	//Reading input file parameters through a par file
	string file_name = argv[1];
	std::ifstream inputFile;
	inputFile.open(file_name.c_str());
	if(inputFile.is_open()){
		string line;
		while(getline(inputFile, line)){
			std::stringstream word(line);
			string var,value = "";
			word >> var;
			if(var == "part_input_file")
				word >> part_input_file;
			else if (var == "timestep_length")
				word >> timestep_length;
			else if (var == "time_end")
				word >> time_end;
			else if (var == "epsilon")
				word >> epsilon;
			else if (var == "sigma")
				word >> sigma;
			else if (var == "part_out_freq")
				word >> part_out_freq;
			else if (var == "part_out_name_base")
				word >> part_out_name_base;
			else if (var == "vtk_out_freq")
				word >> vtk_out_freq;
			else if (var == "vtk_out_name_base")
				word >> vtk_out_name_base;
			else
				word >> cl_workgroup_1dsize;
		}
	}
	inputFile.close();

  int nMols;
	//assign the initial state configuration to host variables by reading part_input_file
	inputFile.open(part_input_file.c_str());
	string line;
	getline(inputFile, line);
	std::stringstream word(line);
	word >> nMols;
  //creating host variables
  dType *h_masses = (dType*)malloc(nMols * sizeof(dType));
  dType *h_positions_x = (dType*)malloc(nMols * sizeof(dType));
  dType *h_positions_y = (dType*)malloc(nMols * sizeof(dType));
  dType *h_positions_z = (dType*)malloc(nMols * sizeof(dType));
  dType *h_velocities_x = (dType*)malloc(nMols * sizeof(dType));
  dType *h_velocities_y = (dType*)malloc(nMols * sizeof(dType));
  dType *h_velocities_z = (dType*)malloc(nMols * sizeof(dType));
	int i = 0;
	while(getline(inputFile, line)){
		std::stringstream word(line);
		word >> h_masses[i] >> h_positions_x[i] >> h_positions_y[i] >> h_positions_z[i] >> h_velocities_x[i] >> h_velocities_y[i] >> h_velocities_z[i];
		i++;
	}

	//creating device memory
	dType *d_masses;
	dType *d_positions_x;
	dType *d_positions_y;
	dType *d_positions_z;
	dType *d_velocities_x;
	dType *d_velocities_y;
	dType *d_velocities_z;
	dType *d_force_x;
	dType *d_force_y;
	dType *d_force_z;
	dType *d_force_old_x;
	dType *d_force_old_y;
	dType *d_force_old_z;

	//GPU implementations
	long long nBytes = nMols * sizeof(dType);

	// memory allocations
	checkError( cudaMalloc(&d_masses, nBytes) );
	checkError( cudaMalloc(&d_positions_x, nBytes) );
	checkError( cudaMalloc(&d_positions_y, nBytes) );
	checkError( cudaMalloc(&d_positions_z, nBytes) );
	checkError( cudaMalloc(&d_velocities_x, nBytes) );
	checkError( cudaMalloc(&d_velocities_y, nBytes) );
	checkError( cudaMalloc(&d_velocities_z, nBytes) );
	checkError( cudaMalloc(&d_force_x, nBytes) );
	checkError( cudaMalloc(&d_force_y, nBytes) );
	checkError( cudaMalloc(&d_force_z, nBytes) );
	checkError( cudaMalloc(&d_force_old_x, nBytes) );
	checkError( cudaMalloc(&d_force_old_y, nBytes) );
	checkError( cudaMalloc(&d_force_old_z, nBytes) );

	// copy data to cuda arrays
	checkError( cudaMemcpy(d_masses, &h_masses[0], nBytes, cudaMemcpyHostToDevice) );
	checkError( cudaMemcpy(d_positions_x, &h_positions_x[0], nBytes, cudaMemcpyHostToDevice) );
	checkError( cudaMemcpy(d_positions_y, &h_positions_y[0], nBytes, cudaMemcpyHostToDevice) );
	checkError( cudaMemcpy(d_positions_z, &h_positions_z[0], nBytes, cudaMemcpyHostToDevice) );
	checkError( cudaMemcpy(d_velocities_x, &h_velocities_x[0], nBytes, cudaMemcpyHostToDevice) );
	checkError( cudaMemcpy(d_velocities_y, &h_velocities_y[0], nBytes, cudaMemcpyHostToDevice) );
	checkError( cudaMemcpy(d_velocities_z, &h_velocities_z[0], nBytes, cudaMemcpyHostToDevice) );
	checkError( cudaDeviceSynchronize() );

    //taking the specifications of the device into consideration for perfect kernel launch
  cudaDeviceProp props;
  cudaGetDeviceProperties(&props, 0);
  int maxThreads = props.maxThreadsPerBlock;
  int nThreads = 0, nBlocks = 1;
  if (maxThreads > nMols){
    nThreads = nMols;
    nBlocks = 1;
  } else {
    nThreads = maxThreads;
    nBlocks += nMols/maxThreads;
  }

	//Kernel implementations

  //calculate initial forces for time = 0
  setForces<<<nBlocks, nThreads>>>(nMols, d_positions_x, d_positions_y, d_positions_z, d_force_x, d_force_y, d_force_z, epsilon, sigma);
  checkError( cudaPeekAtLastError() );

  /*writing the initial configuration to .out and .vtk files*/
  int count = 0;
  //.out files
  string fileName = "./output/" + part_out_name_base + boost::lexical_cast<std::string>(count) + ".out";
    std::ofstream outFile(fileName.c_str());
      if (outFile.is_open()) {
        outFile << nMols << "\n";
        for (int j = 0; j < nMols; j++)
          outFile << std::fixed << h_masses[j] << " " << h_positions_x[j] << " "<< h_positions_y[j] << " " << h_positions_z[j] << " " << h_velocities_x[j] << " " << h_velocities_y[j] << " " << h_velocities_z[j] << "\n";
          outFile.close();
        } else
          cout << "File opening/creating of " << fileName << " failed\n";
    //.vtk file
  fileName = "./output/" + part_out_name_base + boost::lexical_cast<std::string>(count) + ".vtk";
    outFile.open(fileName.c_str());
    if (outFile.is_open()) {
      outFile << "# vtk DataFile Version 4.0\nhesp visualization file\nASCII\nDATASET UNSTRUCTURED_GRID\nPOINTS ";
      outFile << nMols << " double\n";
      for (int j = 0; j < nMols; j++)
        outFile << std::fixed << h_positions_x[j] << " " << h_positions_y[j]  << " " << h_positions_z[j] << "\n";
        outFile << "CELLS 0 0\nCELL_TYPES 0\nPOINT_DATA " << nMols << "\nSCALARS m double\nLOOKUP_TABLE default\n";
        for (int j = 0; j < nMols; j++)
          outFile << std::fixed << h_masses[j] << "\n";
        outFile << "VECTORS v double\n";
        for (int j = 0; j < nMols; j++)
          outFile << std::fixed << h_velocities_x[j] << " " << h_velocities_y[j] << " " << h_velocities_z[j] << "\n";
        outFile.close();
    } else
        cout << "File opening/creating of " << fileName << " failed\n";
    count++;

    //for other times doing the core algorithm
	for(double t = timestep_length; t <= time_end; t += timestep_length) {

        //updating positions of the particles
        updatePosition<<<nBlocks,nThreads>>>(nMols, d_masses, d_positions_x, d_positions_y, d_positions_z, d_velocities_x, d_velocities_y, d_velocities_z, d_force_x, d_force_y, d_force_z, timestep_length);
        checkError( cudaPeekAtLastError() );
        //set old forces
        updateOldForces<<<nBlocks,nThreads>>>(nMols, d_force_x, d_force_y, d_force_z, d_force_old_x, d_force_old_y, d_force_old_z);
        checkError( cudaPeekAtLastError() );
        //set forces
        setForces<<<nBlocks, nThreads>>>(nMols, d_positions_x, d_positions_y, d_positions_z, d_force_x, d_force_y, d_force_z, epsilon, sigma);
        checkError( cudaPeekAtLastError() );
        //update Velcoities
        updateVelocities<<<nBlocks, nThreads>>>(nMols, d_masses, d_velocities_x, d_velocities_y, d_velocities_z, d_force_x, d_force_y, d_force_z, d_force_old_x, d_force_old_y, d_force_old_z, timestep_length);
        checkError( cudaPeekAtLastError() );
        /*writing output into .out and .vtk files*/
        if(count % vtk_out_freq == 0 || count % vtk_out_freq == 0){
          //updating host memory when necessary
          checkError( cudaMemcpy(&h_positions_x[0], d_positions_x, nBytes, cudaMemcpyDeviceToHost));
          checkError( cudaMemcpy(&h_positions_y[0], d_positions_y, nBytes, cudaMemcpyDeviceToHost));
          checkError( cudaMemcpy(&h_positions_z[0], d_positions_z, nBytes, cudaMemcpyDeviceToHost));
          checkError( cudaMemcpy(&h_velocities_x[0], d_velocities_x, nBytes, cudaMemcpyDeviceToHost));
          checkError( cudaMemcpy(&h_velocities_y[0], d_velocities_y, nBytes, cudaMemcpyDeviceToHost));
          checkError( cudaMemcpy(&h_velocities_x[0], d_velocities_z, nBytes, cudaMemcpyDeviceToHost));
        }

        if(count % part_out_freq == 0){
            //.out files
            string fileName = "./output/" + part_out_name_base + boost::lexical_cast<std::string>(count) + ".out";
            std::ofstream outFile(fileName.c_str());
            if (outFile.is_open()) {
              outFile << nMols << "\n";
              for (int j = 0; j < nMols; j++)
                outFile << std::fixed << h_masses[j] << " " << h_positions_x[j] << " "<< h_positions_y[j] << " " << h_positions_z[j] << " " << h_velocities_x[j] << " " << h_velocities_y[j] << " " << h_velocities_z[j] << "\n";
                outFile.close();
            } else
                cout << "File opening/creating of " << fileName << " failed\n";
        }
        //.vtk files
        if(count % vtk_out_freq == 0){
          fileName = "./output/" + part_out_name_base + boost::lexical_cast<std::string>(count) + ".vtk";
            outFile.open(fileName.c_str());
            if (outFile.is_open()) {
              outFile << "# vtk DataFile Version 4.0\nhesp visualization file\nASCII\nDATASET UNSTRUCTURED_GRID\nPOINTS ";
              outFile << nMols << " double\n";
              for (int j = 0; j < nMols; j++)
                outFile << std::fixed << h_positions_x[j] << " " << h_positions_y[j]  << " " << h_positions_z[j] << "\n";
                outFile << "CELLS 0 0\nCELL_TYPES 0\nPOINT_DATA " << nMols << "\nSCALARS m double\nLOOKUP_TABLE default\n";
                for (int j = 0; j < nMols; j++)
                  outFile << std::fixed << h_masses[j] << "\n";
                outFile << "VECTORS v double\n";
                for (int j = 0; j < nMols; j++)
                  outFile << std::fixed << h_velocities_x[j] << " " << h_velocities_y[j] << " " << h_velocities_z[j] << "\n";
                outFile.close();
            } else
                cout << "File opening/creating of " << fileName << " failed\n";
        }
        count++;
	}

	// free memories GPU
	checkError( cudaFree(d_masses) );
	checkError( cudaFree(d_positions_x) );
	checkError( cudaFree(d_positions_y) );
	checkError( cudaFree(d_positions_z) );
	checkError( cudaFree(d_velocities_x) );
	checkError( cudaFree(d_velocities_y) );
	checkError( cudaFree(d_velocities_z) );
	checkError( cudaFree(d_force_x) );
	checkError( cudaFree(d_force_y) );
	checkError( cudaFree(d_force_z) );
	checkError( cudaFree(d_force_old_x) );
	checkError( cudaFree(d_force_old_y) );
	checkError( cudaFree(d_force_old_z) );

	//free CPU variables
	free(h_masses);
	free(h_positions_x);
	free(h_positions_y);
	free(h_positions_z);
	free(h_velocities_x);
	free(h_velocities_y);
	free(h_velocities_z);

  return EXIT_SUCCESS;
}
