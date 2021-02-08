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

__global__ void updatePos (double*d_pPos, double delTime, double* d_pVel, double* d_calculatedData, int* d_pMass, int x_max, int y_max, int z_max)
{
    int Idx = blockIdx.x;
    //int Idy = blockIdx.y;
    //if(Idx!=Idy)
    d_pPos[Idx*3+0]+=delTime*d_pVel[Idx*3+0]+(d_calculatedData[Idx*3+0]*pow(delTime,2)/(2.0*d_pMass[Idx]));
    d_pPos[Idx*3+1]+=delTime*d_pVel[Idx*3+1]+(d_calculatedData[Idx*3+1]*pow(delTime,2)/(2.0*d_pMass[Idx]));
    d_pPos[Idx*3+2]+=delTime*d_pVel[Idx*3+2]+(d_calculatedData[Idx*3+2]*pow(delTime,2)/(2.0*d_pMass[Idx]));

    //	printf("force inside_POS_kernel: %f\t\t Idx:%d\n",d_calculatedData[Idx*3], Idx);
    //printf("POS_kernel_X: %f \nIdx: %d\n", d_pPos[Idx*3], Idx);

    //PBC condition on particle position
    //tolerance of 0.00995 is defined so that after the reappearance, particles do not remain at the boundary but within it
    if(d_pPos[Idx*3+0]<= 0.0) {d_pPos[Idx*3+0] = x_max-0.00995;}
    else if(d_pPos[Idx*3+0]>= x_max) {d_pPos[Idx*3+0] = 0.00995;}

    if(d_pPos[Idx*3+1]<= 0.0) {d_pPos[Idx*3+1] = y_max-0.00995;}
    else if(d_pPos[Idx*3+1]>= y_max) {d_pPos[Idx*3+1] = 0.00995;}

    if(d_pPos[Idx*3+2]<= 0.0) {d_pPos[Idx*3+2] = z_max-0.00995;}
    else if(d_pPos[Idx*3+2]>= z_max) {d_pPos[Idx*3+2] = 0.00995;}
}


__global__ void updateCells(double* d_pPos, int* d_vec_particles, int* d_vec_cells, double len_x, double len_y, double len_z, double x_n, double y_n)
{
    int Idx = blockIdx.x;
    int x_coord = 0, y_coord = 0, z_coord = 0;
    x_coord = (int)(d_pPos[Idx*3+0]/len_x);
    y_coord = (int)(d_pPos[Idx*3+1]/len_y);
    z_coord = (int)(d_pPos[Idx*3+2]/len_z);
    //printf("x_coord: %d\t len_x: %f\n", x_coord, len_x);
   // __syncthreads();

    //d_vec_particles[Idx] = 0; d_vec_cells[int(x_coord+(y_coord*x_n)+(z_coord*x_n*y_n))] = -1;

    d_vec_particles[Idx] = atomicExch(&d_vec_cells[int(x_coord+(y_coord*x_n)+(z_coord*x_n*y_n))], Idx);

    //particle[current] = atomicExch(&cells[current], particlee[current]); //returns 0
    //printf("tester: %d \t", d_vec_cells[0]);

    //d_vec_particles[Idx] = d_vec_cells[int(x_coord+(y_coord*x_n)+(z_coord*x_n*y_n))];
    //d_vec_cells[int(x_coord+(y_coord*x_n)+(z_coord*x_n*y_n))] = Idx;

}


__global__ void forceTwoParticles(int nParticles, int* d_pMass, double* d_pPos, double* d_pVel, double* d_calculatedData, double* d_eps, double* d_sigma, double d_rCutOff)//, int* d_vec_cells)
{
    int Idx = blockIdx.x;

    int l_sys = 1;
    double separation_magnitude = 0.0;
    //int Idy = blockIdx.y;
    //vector<double> initForce = totalForce(Idx, Idy, nParticles, eps, sigma, d_pPos);
    d_calculatedData [Idx*3+0]=0;d_calculatedData [Idx*3+1]=0;d_calculatedData [Idx*3+2]=0;

    //current cell:
    int x_coord = 0, y_coord = 0, z_coord = 0;
    x_coord = (int)(d_pPos[Idx*3+0]/l_sys);
    y_coord = (int)(d_pPos[Idx*3+1]/l_sys);
    z_coord = (int)(d_pPos[Idx*3+2]/l_sys);
    int x_n = 3, y_n = 3,z_n = 3;
    int currentCell = int(x_coord+(y_coord*x_n)+(z_coord*x_n*y_n));



    for(int i=0; i<nParticles; ++i)
        {
            if(Idx!=i)
                {
                    //force calculation for particles at BOUNDARY (opposite side of the domain) for PBC --> update separation vector
                    if((d_pPos[Idx*3+0]/1==0 && d_pPos[i*3+0]/1==2) || (d_pPos[Idx*3+1]/1==0 && d_pPos[i*3+1]/1==2) || (d_pPos[Idx*3+2]/1==0 && d_pPos[i*3+2]/1==2) || (d_pPos[Idx*3+0]/1==2 && d_pPos[i*3+0]/1==0) || (d_pPos[Idx*3+1]/1==2 && d_pPos[i*3+1]/1==0) || (d_pPos[Idx*3+2]/1==2 && d_pPos[i*3+2]/1==0))
                        {
                            double separation [3] = {d_pPos[Idx*3+0]-d_pPos[i*3+0], d_pPos[Idx*3+1]-d_pPos[i*3+1], d_pPos[Idx*3+2]-d_pPos[i*3+2]};
                            double separation_correction [3] = {separation[0]-l_sys*int(separation[0]/l_sys),separation[1]-l_sys*int(separation[1]/l_sys), separation[2]-l_sys*int(separation[2]/l_sys)};
                            separation_magnitude = sqrt (pow(separation_correction[0],2)+pow(separation_correction[1],2)+pow(separation_correction[2],2));
                        }
                    else    //for particles NOT at the boundary
                        {
                            double separation [3] = {d_pPos[Idx*3+0]-d_pPos[i*3+0], d_pPos[Idx*3+1]-d_pPos[i*3+1], d_pPos[Idx*3+2]-d_pPos[i*3+2]};
                            //for (auto i:  separation)
                            //printf("separation vector0: %f \nseparation vector3: %f \nidx: %d \nidy: %d \n", separation[0], separation[3] , Idx, Idy);
                            separation_magnitude = sqrt (pow(separation[0],2)+pow(separation[1],2)+pow(separation[2],2));
                            //	printf("separation: %f  Idx: %d Idy: %d \n", separation_magnitude,Idx, Idy);
                        }

                    double force_LJ = 0;         // cut off radius

                    if(separation_magnitude <= d_rCutOff)
                        force_LJ = 24*(*d_eps/(pow(separation_magnitude,2)))*pow(*d_sigma/separation_magnitude,6)*(2*pow(*d_sigma/separation_magnitude,6)-1);
                    //printf("force: %f  Idx: %d Idy: %d \n", force_LJ,Idx, Idy);
                    //printf("s: %f \n", pow(separation_magnitude,2));
                    //printf("Idx: %d\nx: %f",Idx, force_LJ*(d_pPos[Idx*3+0]-d_pPos[i*3+0]));
                    d_calculatedData [Idx*3+0] += force_LJ*(d_pPos[Idx*3+0]-d_pPos[i*3+0]);
                    d_calculatedData [Idx*3+1] += force_LJ*(d_pPos[Idx*3+1]-d_pPos[i*3+1]);
                    d_calculatedData [Idx*3+2] += force_LJ*(d_pPos[Idx*3+2]-d_pPos[i*3+2]);
                }
            //printf("calculated FORCE_kernel %f, %f, %f\n%f, %f, %f\n", d_calculatedData [0],d_calculatedData [1],d_calculatedData [2],d_calculatedData [3],d_calculatedData [4],d_calculatedData [5]);
        }


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
                            if(str1.length() != 0)
                                {
                                    if(str1[str1.size()]=='\0' || '\n')	//alpha-numeric character check
                                        {
                                            str1.pop_back();
                                        }
                                    parFile.push_back(str1);
                                }
                        }

                }
        }
    else
        {
            cout<<"*.par file not found"<< endl;
        }

    cout<<"parFile contents: "<<endl;
    //std::copy(parFile.begin(),parFile.end(), ostream_iterator<string>(cout, "\n"));
    for(int i=0; i<parFile.size(); ++i)
        {
            cout<<"i: "<<i<<'\t'<<"strLength: "<<parFile[i].length()<<'\t'<<parFile[i]<<endl;
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



    double delTime = (int)(stod(parFile[5])*1000.0)/1000.0, endTime = stod(parFile[3]), eps = stod(parFile[7]), sigma = stod(parFile[9]);
    string baseFile = parFile[13];
    double initTime = delTime;

    cout<<"@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ "<<"delTime: "<<fixed<<setprecision(35)<<delTime<<", "<<"endTIme: "<<endTime<<", "<<"initTIme: "<<initTime<<endl;

    //defining variables for computational domain
    double x_min = stoi(parFile[27]), x_max = stoi(parFile[29]), y_min = stoi(parFile[31]), y_max = stoi(parFile[33]), z_min = stoi(parFile[35]), z_max = stoi(parFile[37]);
    double x_n = stoi(parFile[39]), y_n = stoi(parFile[41]), z_n = stoi(parFile[43]);    //total number of divisions in x,y and z directions
    double rCutOff = stod(parFile[45]);

    int nCells = (int)(x_n * y_n * z_n);   //total number of cells
    cout<<"celllssss: "<<nCells<<endl;
    double len_x = (x_max - x_min) / x_n, len_y = (y_max - y_min) / y_n, len_z = (z_max - z_min) / z_n;    //length of each cell in x,y,z dir
cout<<"###"<<endl;
    cout<<"x_max: "<<x_max<<"\t"<<"x_min: "<<x_min<<"\t"<<"x_n: "<<x_n<<endl;
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

    vector<int> vec_cells (nCells,-1);  //vector of cells, each initialized to -1
    vector<int> vec_particles (nParticles); //vector of Particles, each initialized to 0

    //        for (int id=0; id<nParticles; ++id )
    //            {
    //                int x_coord = 0, y_coord = 0, z_coord = 0;
    //                        x_coord = pPos[id*3+0]/1;
    //                        y_coord = pPos[id*3+1]/1;
    //                        z_coord = pPos[id*3+2]/1;x_max, y_max, z_ma
    //                        vec_particles[id] = vec_cells[x_coord+(y_coord*3)+(z_coord*9)];
    //                        vec_cells[x_coord+(y_coord*3)+(z_coord*9)] = id;

    //            }

    int* d_vec_cells; int* d_vec_particles;
    cudaMalloc (&d_vec_cells, nCells*sizeof(int));
    cudaMalloc (&d_vec_particles, nParticles*sizeof(int));

// not copied them to device yet

    //initial force
    forceTwoParticles<<<nParticles, 1>>>(nParticles, d_pMass, d_pPos, d_pVel, d_calculatedData, d_eps, d_sigma, rCutOff);
    cudaMemcpy (&calculatedData[0], d_calculatedData, 3*nParticles*sizeof(double), cudaMemcpyDeviceToHost);

//    for(auto i: calculatedData)
//        {cout<<"initialForce: "<<i<<'\t';}
//    cout<<endl;


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



    while (initTime<=endTime)
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



            cout<<"\n####\tIteration:  "<<n<<"\tTime: "<<fixed<<setprecision(20)<<initTime<<"\t####\n";
            //calling Kernel
            /*for(int i=0; i<pPos.size();++i)
                {

                        cout<<"old POS_main: "<<pPos[i]<<'\t';
                }
                cout<<endl;*/

            //cudaMemcpy (d_calculatedData, &calculatedData[0], 3*nParticles*sizeof(double), cudaMemcpyHostToDevice);

            vector<int> vec_cells (nCells,-1);  //vector of cells, each initialized to -1
            vector<int> vec_particles (nParticles); //vector of Particles, each initialized to 0
            cudaMemcpy(d_vec_cells, &vec_cells[0], nCells*sizeof(int), cudaMemcpyHostToDevice);
            cudaMemcpy(d_vec_particles, &vec_particles[0], nParticles*sizeof(int), cudaMemcpyHostToDevice);

            updatePos<<<nParticles , 1>>>(d_pPos, delTime, d_pVel, d_calculatedData,d_pMass, x_max, y_max, z_max);
            cudaMemcpy (&pPos[0], d_pPos, 3*nParticles*sizeof(double), cudaMemcpyDeviceToHost);
            cudaMemcpy (d_pPos, &pPos[0], 3*nParticles*sizeof(double), cudaMemcpyHostToDevice);

            updateCells<<<nParticles , 1>>> (d_pPos, d_vec_particles, d_vec_cells, len_x, len_y, len_z, x_n, y_n);
            cudaMemcpy (&vec_cells[0], d_vec_cells, nCells*sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy (d_vec_cells, &vec_cells[0], nCells*sizeof(int), cudaMemcpyHostToDevice);
            cudaMemcpy (&vec_particles[0], d_vec_particles, nParticles*sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy (d_vec_particles, &vec_particles[0], nParticles*sizeof(int), cudaMemcpyHostToDevice);
            for(int i=0; i<nCells; ++i)
                {
                    cout<<vec_cells[i]<<", ";
                }
            cout<<endl;
            for(int i=0; i<nParticles; ++i)
                {
                    cout<<vec_particles[i]<<", ";
                }
            cout<<endl;
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

            //                cout<<"updated POS: ";
            //                for(int i=0; i<pPos.size();++i)
            //                {

            //                        cout<<fixed<<pPos[i]<<'\t';
            //                }
            //                cout<<endl;
            cudaMemcpy (d_oldForce, &oldForce[0], 3*nParticles*sizeof(double), cudaMemcpyHostToDevice);

            forceTwoParticles<<<nParticles , 1>>>(nParticles, d_pMass, d_pPos, d_pVel, d_calculatedData, d_eps, d_sigma, rCutOff);
            cudaMemcpy (&calculatedData[0], d_calculatedData, 3*nParticles*sizeof(double), cudaMemcpyDeviceToHost);
            //cudaMemcpy (d_calculatedData, &calculatedData[0], totalTimeSteps*nParticles*sizeof(double), cudaMemcpyHostToDevice);

            //                for(int i=0; i<calculatedData.size();++i)
            //                {

            //                        cout<<"new FORCE_main: "<<calculatedData[i]<<'\t';
            //                }

            updateVel<<<nParticles , 1>>>(d_pPos, delTime, d_pVel, d_calculatedData,d_pMass, d_oldForce);

            cudaMemcpy (&pPos[0], d_pPos, 3*nParticles*sizeof(double), cudaMemcpyDeviceToHost);
            cudaMemcpy (&pVel[0], d_pVel, 3*nParticles*sizeof(double), cudaMemcpyDeviceToHost);

            //        cout<<"updated VEL: ";
            //                for(int i=0;i<pVel.size();++i)
            //                {
            //                        cout<<fixed<<pVel[i]<<'\t';
            //                }
            //                cout<<endl;

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
