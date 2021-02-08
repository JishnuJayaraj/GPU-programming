#include<iostream>
#include<vector>
#include<fstream>
#include<sstream>
#include<string>
#include<iterator>
#include<ctype.h>
#include <iomanip>
#include <math.h>
#include <fstream>
#include<cuda_runtime.h>

using namespace std;

vector<string> parFile;
vector<string> particleConfig;


__global__ void updateVel (double* d_pPos, double delTime, double* d_pVel,double* d_calculatedData, int* d_pMass, double* d_oldForce)
{
	int Idx = blockIdx.x;

	d_pVel[Idx*3+0]+=(d_oldForce[Idx*3+0]+d_calculatedData[Idx*3+0])*delTime/(2.0*d_pMass[Idx]);
	d_pVel[Idx*3+1]+=(d_oldForce[Idx*3+1]+d_calculatedData[Idx*3+1])*delTime/(2.0*d_pMass[Idx]);
	d_pVel[Idx*3+2]+=(d_oldForce[Idx*3+2]+d_calculatedData[Idx*3+2])*delTime/(2.0*d_pMass[Idx]);
//	printf("Old force_kernel: %f \t new FOrce_kernel:%f \tIdx: %d\n", d_oldForce[Idx*3], d_calculatedData[Idx*3], Idx);
//	printf("vel_kernel: %f \nIdx: %d\n", d_pVel[Idx*3], Idx);

}

__global__ void updatePos (double*d_pPos, double delTime, double* d_pVel, double* d_calculatedData, int* d_pMass)
{
	int Idx = blockIdx.x;
	//int Idy = blockIdx.y;
	//if(Idx!=Idy)
	
	{
	d_pPos[Idx*3+0]+=delTime*d_pVel[Idx*3+0]+(d_calculatedData[Idx*3+0]*pow(delTime,2)/(2.0*d_pMass[Idx]));
	d_pPos[Idx*3+1]+=delTime*d_pVel[Idx*3+1]+(d_calculatedData[Idx*3+1]*pow(delTime,2)/(2.0*d_pMass[Idx]));
	d_pPos[Idx*3+2]+=delTime*d_pVel[Idx*3+2]+(d_calculatedData[Idx*3+2]*pow(delTime,2)/(2.0*d_pMass[Idx]));
	}
	
//	printf("force inside_POS_kernel: %f\t\t Idx:%d\n",d_calculatedData[Idx*3], Idx);
//	printf("POS_kernel: %f \nIdx: %d\n", d_pPos[Idx*3], Idx);
}

__global__ void forceTwoParticles(int nParticles, int* d_pMass, double* d_pPos, double* d_pVel, double* d_calculatedData, double* d_eps, double* d_sigma)
{
	int Idx = blockIdx.x ;
	int Idy = blockIdx.y;
	
	//vector<double> initForce = totalForce(Idx, Idy, nParticles, eps, sigma, d_pPos);
	if(Idx!=Idy)
	{
		double separation [3] = {d_pPos[Idx*3+0]-d_pPos[Idy*3+0], d_pPos[Idx*3+1]-d_pPos[Idy*3+1], d_pPos[Idx*3+2]-d_pPos[Idy*3+2]};
	//for (auto i:  separation)
 		//printf("separation vector0: %f \nseparation vector3: %f \nidx: %d \nidy: %d \n", separation[0], separation[3] , Idx, Idy);
 	
	
	double separation_magnitude = sqrt (pow(separation[0],2)+pow(separation[1],2)+pow(separation[2],2));
//	printf("separation: %f  Idx: %d Idy: %d \n", separation_magnitude,Idx, Idy);
	
	double force_LJ = 24*(*d_eps/(pow(separation_magnitude,2)))*pow(*d_sigma/separation_magnitude,6)*(2*pow(*d_sigma/separation_magnitude,6)-1);
	//printf("force: %f  Idx: %d Idy: %d \n", force_LJ,Idx, Idy);
	//printf("s: %f \n", pow(separation_magnitude,2));
	
	d_calculatedData [Idx*3+0] = force_LJ*(d_pPos[Idx*3+0]-d_pPos[Idy*3+0]);
	d_calculatedData [Idx*3+1] = force_LJ*(d_pPos[Idx*3+1]-d_pPos[Idy*3+1]);
	d_calculatedData [Idx*3+2] = force_LJ*(d_pPos[Idx*3+2]-d_pPos[Idy*3+2]);
	
	/*printf("calculated FORCE_kernel %f, %f, %f\n%f, %f, %f\n", d_calculatedData [0],d_calculatedData [1],d_calculatedData [2],d_calculatedData [3],d_calculatedData [4],d_calculatedData [5]);
*/	}
	

}


int main(int argc, char* argv[])
{
	//enableMapHost();
	
	ifstream parFileName (argv[1]);		//reading *.par file
	if (parFileName.is_open())
	{
		string str;
		while(getline(parFileName, str))
		{
			stringstream temp(str);
			string str1;
			while (getline(temp, str1,' '))
			{
				parFile.push_back(str1);
			}
			
		}
	}
	else
	{
		cout<<"*.par file not found"<< endl;
	}

	
	ifstream particleConfigFile (parFile[1]);	//reading *.in file
	
	

	if (particleConfigFile.is_open())
	{
		string str;
		while(getline(particleConfigFile,str))
		{
			stringstream temp(str);
			string str1;
			while (getline(temp, str1,' '))
			{
				
				if(str1[str1.size()-1]=='\0')	//alpha-numeric character check
				{
					str1.pop_back();
				}
				particleConfig.push_back(str1);
			}
		}
	}
	else
	{
		cout<<"*.in file not found"<< endl;
	}
	
	for(int i=0;i<particleConfig.size();++i)
	{
		cout<<"i: "<<i<<" "<<setprecision(10)<<(particleConfig[i])<<"   size: "<<particleConfig[i].size()<<endl;
	
	}

	
	int nParticles = stoi(particleConfig[0]);
	particleConfig.erase(particleConfig.begin());
	
	vector<int> pMass (nParticles);
	vector<double> pPos (nParticles*3);
	vector<double> pVel (nParticles*3);
	
	for(int i=0; i<nParticles; ++i)
	{
		pMass[i]=stoi(particleConfig[i*7]);
	}
	
	int i=0;
	for(int j=0; j<particleConfig.size(); ++j)
	{
		if((i*7+3)<=particleConfig.size())
		{
			pPos[j]=stod(particleConfig[i*7+1]);
			pPos[j+1]=stod(particleConfig[i*7+2]);
			pPos[j+2]=stod(particleConfig[i*7+3]);	
			j+=2;
			++i;
		}
		else
			break;
	}
	i=0;
	for(int j=0; j<particleConfig.size(); ++j)
	{
		if((i*7+6)<=particleConfig.size())
		{

			pVel[j]=stod(particleConfig[i*7+4]);
			pVel[j+1]=stod(particleConfig[i*7+5]);
			pVel[j+2]=stod(particleConfig[i*7+6]);
			j+=2;
			++i;
		}	
		else
			break;
	}
	
	
	
	double initTime=0.005, delTime = stod(parFile[3]), endTime = stod(parFile[5]), eps = stod(parFile[7]), sigma = stod(parFile[9]);
	string baseFile = parFile[13];
	
	int totalTimeSteps = endTime / delTime;
	
	
	vector<double> calculatedData(3*nParticles);
	
	//assigning pointer variables for device memory
	int* d_pMass; double* d_pPos; double* d_pVel; double* d_calculatedData;
	double* d_eps; double* d_sigma; double* d_delTime;
	
	//mem allocation in Device
	cudaMalloc (&d_pMass, nParticles*sizeof(int));
	cudaMalloc (&d_pPos, nParticles*3*sizeof(double));
	cudaMalloc (&d_pVel, nParticles*3*sizeof(double));
	cudaMalloc (&d_calculatedData, 3*nParticles*sizeof(double));
	cudaMalloc (&d_eps, sizeof(double));
	cudaMalloc (&d_sigma, sizeof(double));
	cudaMalloc (&d_delTime, sizeof(double));
	
	//mem copy from HostToDevice
	cudaMemcpy (d_pMass, &pMass[0],nParticles*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy (d_pPos, &pPos[0],nParticles*3*sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy (d_pVel, &pVel[0],nParticles*3*sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy (d_calculatedData, &calculatedData[0],3*nParticles*sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy (d_eps,&eps,sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy (d_sigma,&sigma,sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy (d_delTime,&d_delTime,sizeof(double), cudaMemcpyHostToDevice);
	
	int nThreads = nParticles*(nParticles-1);
	
	vector<double> oldForce;
	oldForce.resize(3*nParticles);
	
	double* d_oldForce;
	cudaMalloc (&d_oldForce, 3*nParticles*sizeof(double));
	
	dim3 grid(nParticles,nParticles,1);
	
	
	//cout<<"hi"<<endl;
	
	//initial force
	forceTwoParticles<<<grid, 1>>>(nParticles, d_pMass, d_pPos, d_pVel, d_calculatedData, d_eps, d_sigma);
	cudaMemcpy (&calculatedData[0], d_calculatedData, 3*nParticles*sizeof(double), cudaMemcpyDeviceToHost);
	
	for(auto i: calculatedData)
	{cout<<"initialForce: "<<i<<'\t';}
	cout<<endl;
	
	
	int n=0;
	int vtk_count = 1;
	int out_count = 1;
	bool write_vtk = true;
	bool write_out = true;
	int temp_vtk = 0;
	int temp_out = 0;
	//calculate intial forces
	vector<double> force_new;
	
	//initial configuration at t=0
	ofstream myfile; 
	myfile.open(baseFile+"0"+".vtk");
			myfile<<"# vtk DataFile Version 4.0\nhesp visualization file\nASCII\nDATASET UNSTRUCTURED_GRID\n";
			myfile<<"POINTS"<<" "<<nParticles<<" "<<"double"<<'\n';
			for(int i=0;i<nParticles;++i)
			{
				myfile<<fixed<<pPos[i*3+0]<<" "<<pPos[i*3+1]<<" "<<pPos[i*3+2];
				myfile<<'\n';
			}
			
			myfile<<"CELLS"<<" "<<0<<" "<<0<<'\n';
			myfile<<"CELL_TYPES"<<" "<<0<<'\n';
			myfile<<"POINT_DATA"<<" "<<nParticles<<'\n';
			myfile<<"SCALARS"<<" "<<"m"<<" "<<"double"<<'\n';
			myfile<<"LOOKUP_TABLE"<<" "<<"default"<<'\n';
			for (int i=0; i<nParticles;++i)
			{
				myfile<<pMass[i]<<'\n';
			}
			myfile<<"VECTORS"<<" "<<"v"<<" "<<"double"<<'\n';
			for(int i=0;i<nParticles;++i)
			{
				myfile<<pVel[i*3+0]<<" "<<pVel[i*3+1]<<" "<<pVel[i*3+2];
				myfile<<'\n';
			}
			myfile.close();
	myfile.open(baseFile+"0"+".out");
			myfile<<nParticles<<'\n';
			for(int i=0; i<nParticles; ++i)
			{
				myfile<<pMass[i]<<" "<<fixed<<pPos[i*3+0]<<" "<<pPos[i*3+1]<<" "<<pPos[i*3+2]<<" "<<pVel[i*3+0]<<" "<<pVel[i*3+1]<<" "<<pVel[i*3+2]<<'\n';			
			}
			myfile.close();
	
	temp_vtk += 1;
	write_vtk=false;
	temp_out += 1; 
	write_out=false;


	
	while (initTime<endTime)
	{
		if (temp_vtk % stoi(parFile[15]) == 0)
	     		//cout<<temp_vtk<<endl;
	     		write_vtk = true;
	     	if (temp_out % stoi(parFile[11]) == 0) 
    	
	     		//cout<<temp_out<<endl;
	     		write_out = true;
		if (temp_vtk % stoi(parFile[15]) != 0)
			write_vtk=false;
		if (temp_out % stoi(parFile[11]) != 0)	
			write_out=false;

		
		
		cout<<"\n####\tIteration:  "<<n<<"\tTime: "<<initTime<<"\t####\n";
		//calling Kernel
		/*for(int i=0; i<pPos.size();++i)
		{
			
			cout<<"old POS_main: "<<pPos[i]<<'\t';
		}
		cout<<endl;*/
		updatePos<<<nParticles , 1>>>(d_pPos, delTime, d_pVel, d_calculatedData,d_pMass);
		cudaMemcpy (&pPos[0], d_pPos, 3*nParticles*sizeof(double), cudaMemcpyDeviceToHost);
		//cudaMemcpy (d_pPos, &pPos[0], 3*nParticles*sizeof(double), cudaMemcpyHostToDevice);
		/*for(int i=0; i<nParticles*3; ++i)
	{
		cout<<"###"<<endl;
		cout<<fixed<<"pos: "<<pPos[i]<<endl;
		cout<<"vel: "<<pVel[i]<<endl;
	}*/
		//cout<<"hello"<<endl;
		cudaMemcpy (&calculatedData[0], d_calculatedData, 3*nParticles*sizeof(double), cudaMemcpyDeviceToHost);
		//oldForce.assign(calculatedData.begin(), calculatedData.end()-1);
		
		/*for (auto i:  calculatedData)
 			 std::cout << i << ' ';*/
		for(int i=0; i<oldForce.size();++i)
		{
			oldForce[i]=calculatedData[i];
			//cout<<"old FORC_main: "<<oldForce[i]<<'\t';
		}
		//cout<<endl;
		
		cout<<"updated POS: ";
		for(int i=0; i<pPos.size();++i)
		{
			
			cout<<fixed<<pPos[i]<<'\t';
		}
		cout<<endl;
		cudaMemcpy (d_oldForce, &oldForce[0], 3*nParticles*sizeof(double), cudaMemcpyHostToDevice);
		
		forceTwoParticles<<<grid , 1>>>(nParticles, d_pMass, d_pPos, d_pVel, d_calculatedData, d_eps, d_sigma);
		cudaMemcpy (&calculatedData[0], d_calculatedData, 3*nParticles*sizeof(double), cudaMemcpyDeviceToHost);
		//cudaMemcpy (d_calculatedData, &calculatedData[0], totalTimeSteps*nParticles*sizeof(double), cudaMemcpyHostToDevice);
			
		/*for(int i=0; i<calculatedData.size();++i)
		{
			
			cout<<"new FORCE_main: "<<calculatedData[i]<<'\t';
		}*/
		
		updateVel<<<nParticles , 1>>>(d_pPos, delTime, d_pVel, d_calculatedData,d_pMass, d_oldForce);
		
		cudaMemcpy (&pPos[0], d_pPos, 3*nParticles*sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy (&pVel[0], d_pVel, 3*nParticles*sizeof(double), cudaMemcpyDeviceToHost);
		
	cout<<"updated VEL: ";
		for(int i=0;i<pVel.size();++i)
		{
			cout<<fixed<<pVel[i]<<'\t';
		}		
		cout<<endl;
		
		//writing *.vtk file

		if(write_vtk == true)
		{
			myfile.open(baseFile+to_string(vtk_count)+".vtk");
			myfile<<"# vtk DataFile Version 4.0\nhesp visualization file\nASCII\nDATASET UNSTRUCTURED_GRID\n";
			myfile<<"POINTS"<<" "<<nParticles<<" "<<"double"<<'\n';
			for(int i=0;i<nParticles;++i)
			{
				myfile<<fixed<<pPos[i*3+0]<<" "<<pPos[i*3+1]<<" "<<pPos[i*3+2];
				myfile<<'\n';
			}
			
			myfile<<"CELLS"<<" "<<0<<" "<<0<<'\n';
			myfile<<"CELL_TYPES"<<" "<<0<<'\n';
			myfile<<"POINT_DATA"<<" "<<nParticles<<'\n';
			myfile<<"SCALARS"<<" "<<"m"<<" "<<"double"<<'\n';
			myfile<<"LOOKUP_TABLE"<<" "<<"default"<<'\n';
			for (int i=0; i<nParticles;++i)
			{
				myfile<<pMass[i]<<'\n';
			}
			myfile<<"VECTORS"<<" "<<"v"<<" "<<"double"<<'\n';
			for(int i=0;i<nParticles;++i)
			{
				myfile<<pVel[i*3+0]<<" "<<pVel[i*3+1]<<" "<<pVel[i*3+2];
				myfile<<'\n';
			}
			myfile.close();
			vtk_count+=1;
		}
		
		if(write_out == true)
		{
			//cout<<"temp_out: "<<temp_out<<"\t";
			myfile.open(baseFile+to_string(out_count)+".out");
			myfile<<nParticles<<'\n';
			for(int i=0; i<nParticles; ++i)
			{
				myfile<<pMass[i]<<" "<<fixed<<pPos[i*3+0]<<" "<<pPos[i*3+1]<<" "<<pPos[i*3+2]<<" "<<pVel[i*3+0]<<" "<<pVel[i*3+1]<<" "<<pVel[i*3+2]<<'\n';			
			}
			myfile.close();
			out_count+=1;		
		}
		
		
		temp_vtk += 1;
		temp_out += 1;

		
		initTime+=delTime;
		++n;
	}
	
	
	
}
