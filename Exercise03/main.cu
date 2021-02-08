#include "kernel.h"

using std::cout; using std::cin; using std::endl;
using std::string; using std::vector;

typedef double dType;

int main(int argc, char* argv[]) {
	//simulation parameters
	std::string part_input_file;
	double timestep_length;
	double time_end;
	double epsilon;
	double sigma;
	int	part_out_freq;
	std::string 	part_out_name_base;
	int	vtk_out_freq;
	std::string vtk_out_name_base;
	int	cl_workgroup_1dsize;
	int cl_workgroup_3dsize_x;
	int cl_workgroup_3dsize_y;
	int cl_workgroup_3dsize_z;
	float x_min;
	float x_max;
	float y_min;
	float y_max;
	float z_min;
	float z_max;
	unsigned int x_n;
	unsigned int y_n;
	unsigned int z_n;
	double r_cut;

	//Reading input file parameters through a par file
	string file_name = "./input/grid.par";  //argv[1;
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
			else if (var == "cl_workgroup_1dsize")
				word >> cl_workgroup_1dsize;
			else if (var == "cl_workgroup_3dsize_x")
				word >> cl_workgroup_3dsize_x;
			else if (var == "cl_workgroup_3dsize_y")
				word >> cl_workgroup_3dsize_y;
			else if (var == "cl_workgroup_3dsize_z")
				word >> cl_workgroup_3dsize_z;
			else if (var == "x_min")
				word >> x_min;
			else if (var == "x_max")
				word >> x_max;
			else if (var == "y_min")
				word >> y_min;
			else if (var == "y_max")
				word >> y_max;
			else if (var == "z_min")
				word >> z_min;
			else if (var == "z_max")
				word >> z_max;
			else if (var == "x_n")
				word >> x_n;
			else if (var == "y_n")
				word >> y_n;
			else if (var == "z_n")
					word >> z_n;
			else
				word >> r_cut;
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
	thrust::host_vector<dType> h_masses(nMols);
	thrust::host_vector<dType> h_positions_x(nMols);
	thrust::host_vector<dType> h_positions_y(nMols);
	thrust::host_vector<dType> h_positions_z(nMols);
	thrust::host_vector<dType> h_velocities_x(nMols);
	thrust::host_vector<dType> h_velocities_y(nMols);
	thrust::host_vector<dType> h_velocities_z(nMols);

	int i = 0;
	while(getline(inputFile, line)){
		std::stringstream word(line);
		word >> h_masses[i] >> h_positions_x[i] >> h_positions_y[i] >> h_positions_z[i] >> h_velocities_x[i] >> h_velocities_y[i] >> h_velocities_z[i];
		i++;
	}

	//creating device memory
	thrust::device_vector<dType> d_masses = h_masses;
	thrust::device_vector<dType> d_positions_x = h_positions_x;
	thrust::device_vector<dType> d_positions_y = h_positions_y;
	thrust::device_vector<dType> d_positions_z = h_positions_z;
	thrust::device_vector<dType> d_velocities_x = h_velocities_x;
	thrust::device_vector<dType> d_velocities_y = h_velocities_y;
	thrust::device_vector<dType> d_velocities_z = h_velocities_z;
	thrust::device_vector<dType> d_force_x(nMols);
	thrust::device_vector<dType> d_force_y(nMols);
	thrust::device_vector<dType> d_force_z(nMols);
	thrust::device_vector<dType> d_force_old_x(nMols);
	thrust::device_vector<dType> d_force_old_y(nMols);
	thrust::device_vector<dType> d_force_old_z(nMols);

	//GPU implementations

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
	periodicBC<<<nBlocks, nThreads>>>(nMols, thrust::raw_pointer_cast(&d_positions_x[0]), thrust::raw_pointer_cast(&d_positions_y[0]), thrust::raw_pointer_cast(&d_positions_z[0]), x_min, x_max, y_min, y_max, z_min, z_max);
	checkError( cudaPeekAtLastError() );

  setForces<<<nBlocks, nThreads>>>(nMols, thrust::raw_pointer_cast(&d_positions_x[0]), thrust::raw_pointer_cast(&d_positions_y[0]), thrust::raw_pointer_cast(&d_positions_z[0]), thrust::raw_pointer_cast(&d_force_x[0]), thrust::raw_pointer_cast(&d_force_y[0]), thrust::raw_pointer_cast(&d_force_z[0]), epsilon, sigma);
  checkError( cudaPeekAtLastError() );

  /*writing the initial configuration to .out and .vtk files*/
  int count = 0;
  //.out files
	system("exec rm -r ./output/*"); //deltes the contents of the folder output
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
        updatePosition<<<nBlocks,nThreads>>>(nMols, thrust::raw_pointer_cast(&d_masses[0]), thrust::raw_pointer_cast(&d_positions_x[0]), thrust::raw_pointer_cast(&d_positions_y[0]), thrust::raw_pointer_cast(&d_positions_z[0]), thrust::raw_pointer_cast(&d_velocities_x[0]), thrust::raw_pointer_cast(&d_velocities_y[0]), thrust::raw_pointer_cast(&d_velocities_z[0]), thrust::raw_pointer_cast(&d_force_x[0]), thrust::raw_pointer_cast(&d_force_y[0]), thrust::raw_pointer_cast(&d_force_z[0]), timestep_length);
        checkError( cudaPeekAtLastError() );

				periodicBC<<<nBlocks, nThreads>>>(nMols, thrust::raw_pointer_cast(&d_positions_x[0]), thrust::raw_pointer_cast(&d_positions_y[0]), thrust::raw_pointer_cast(&d_positions_z[0]), x_min, x_max, y_min, y_max, z_min, z_max);
				checkError( cudaPeekAtLastError() );

        //set old forces
        updateOldForces<<<nBlocks,nThreads>>>(nMols, thrust::raw_pointer_cast(&d_force_x[0]), thrust::raw_pointer_cast(&d_force_y[0]), thrust::raw_pointer_cast(&d_force_z[0]), thrust::raw_pointer_cast(&d_force_old_x[0]), thrust::raw_pointer_cast(&d_force_old_y[0]), thrust::raw_pointer_cast(&d_force_old_z[0]));
        checkError( cudaPeekAtLastError() );
        //set forces
        setForces<<<nBlocks, nThreads>>>(nMols, thrust::raw_pointer_cast(&d_positions_x[0]), thrust::raw_pointer_cast(&d_positions_y[0]), thrust::raw_pointer_cast(&d_positions_z[0]), thrust::raw_pointer_cast(&d_force_x[0]), thrust::raw_pointer_cast(&d_force_y[0]), thrust::raw_pointer_cast(&d_force_z[0]), epsilon, sigma);
        checkError( cudaPeekAtLastError() );
        //update Velcoities
        updateVelocities<<<nBlocks, nThreads>>>(nMols, thrust::raw_pointer_cast(&d_masses[0]), thrust::raw_pointer_cast(&d_velocities_x[0]), thrust::raw_pointer_cast(&d_velocities_y[0]), thrust::raw_pointer_cast(&d_velocities_z[0]), thrust::raw_pointer_cast(&d_force_x[0]), thrust::raw_pointer_cast(&d_force_y[0]), thrust::raw_pointer_cast(&d_force_z[0]), thrust::raw_pointer_cast(&d_force_old_x[0]), thrust::raw_pointer_cast(&d_force_old_y[0]), thrust::raw_pointer_cast(&d_force_old_z[0]), timestep_length);
        checkError( cudaPeekAtLastError() );
        /*writing output into .out and .vtk files*/
        if(count % vtk_out_freq == 0 || count % vtk_out_freq == 0){
          //updating host memory when necessary
					thrust::copy(d_positions_x.begin(), d_positions_x.end(), h_positions_x.begin());
					thrust::copy(d_positions_y.begin(), d_positions_y.end(), h_positions_y.begin());
					thrust::copy(d_positions_z.begin(), d_positions_z.end(), h_positions_z.begin());
					thrust::copy(d_velocities_x.begin(), d_velocities_x.end(), h_velocities_x.begin());
					thrust::copy(d_velocities_y.begin(), d_velocities_y.end(), h_velocities_y.begin());
					thrust::copy(d_velocities_z.begin(), d_velocities_z.end(), h_velocities_z.begin());
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

  return EXIT_SUCCESS;
}
