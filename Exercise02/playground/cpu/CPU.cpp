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

using namespace std;

vector<string> parFile;
vector<string> particleConfig;
	




vector<double> force (int particleA, int particleB, double eps, double sigma, vector<double> pPos)	//force is exerted by particleB on particleA
{
	
	vector<double> separation = {pPos[particleA*3+0]-pPos[particleB*3+0], pPos[particleA*3+1]-pPos[particleB*3+1], pPos[particleA*3+2]-pPos[particleB*3+2]};
	
	/*cout<<"separation vector"<<endl;
	std::copy(separation.begin(), separation.end(), std::ostream_iterator<double>(std::cout, "\t")); cout<<endl;
	cout<<"######"<<endl;*/
	
	
	
	double separation_magnitude = sqrt (pow(separation[0],2)+pow(separation[1],2)+pow(separation[2],2));
	//cout<<"separation magnitude: "<<separation_magnitude<<endl;
	
	double force_LJ = 24*(eps/pow(separation_magnitude,2))*pow(sigma/separation_magnitude,6)*(2*pow(sigma/separation_magnitude,6)-1);
	
	//cout<<"force_LJ: "<<force_LJ<<endl;
	
	
	vector<double> vec_force = {force_LJ*(pPos[particleA*3+0]-pPos[particleB*3+0]), force_LJ*(pPos[particleA*3+1]-pPos[particleB*3+1]), force_LJ*(pPos[particleA*3+2]-pPos[particleB*3+2])};
	
	/*cout<<"vec_force"<<endl;
	std::copy(vec_force.begin(), vec_force.end(), std::ostream_iterator<double>(std::cout, "\t")); cout<<endl;
	cout<<"######"<<endl;*/
	
	
	
	return vec_force;

}




vector<double> totalForce(int nParticles, double eps, double sigma,vector<double> pPos )
{
	
	vector<double> total_force;
	for (int i=0; i<nParticles; ++i)
		{
			vector<double> sum= {0,0,0};
			for(int j=0; j<nParticles; ++j)
			{
				if(i!=j)
				{
					vector<double> force_two_particles = force(i, j, eps, sigma, pPos);
					sum[0]+=force_two_particles[0];
					sum[1]+=force_two_particles[1];
					sum[2]+=force_two_particles[2];	
				}
				
			}
			total_force.push_back(sum[0]);
			total_force.push_back(sum[1]);
			total_force.push_back(sum[2]);
			
		}

	return total_force;

}


void velocityVerlet(int nParticles, vector<int> pMass, vector<double> pPos ,vector<double> pVel)
{
	
	double initTime=0, delTime = stod(parFile[3]), endTime = stod(parFile[5]), eps = stod(parFile[7]), sigma = stod(parFile[9]);
	string baseFile = parFile[13];
	
	vector<double> initForce = totalForce(nParticles, eps, sigma, pPos );
	
	//std::copy(initForce.begin(), initForce.end(), std::ostream_iterator<double>(std::cout, "\t"));
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
	     		cout<<temp_vtk<<endl;
	     		write_vtk = true;
	     	if (temp_out % stoi(parFile[11]) == 0)	     	
	     		cout<<temp_out<<endl;
	     		write_out = true;
		if (temp_vtk % stoi(parFile[15]) != 0)
			write_vtk=false;
		if (temp_out % stoi(parFile[11]) != 0)	
			write_out=false;
			
			
			
		for(int i=0; i<nParticles; ++i)
		{
			pPos[i*3+0]+=delTime*pVel[i*3+0]+(initForce[i*3+0]*pow(delTime,2)/(2.0*pMass[i]));
			pPos[i*3+1]+=delTime*pVel[i*3+1]+(initForce[i*3+1]*pow(delTime,2)/(2.0*pMass[i]));
			pPos[i*3+2]+=delTime*pVel[i*3+2]+(initForce[i*3+2]*pow(delTime,2)/(2.0*pMass[i]));
		}
		
		/*cout<<"@@@@"<<endl;
	std::copy(pPos.begin(), pPos.end(), std::ostream_iterator<double>(std::cout, "\t"));
	cout<<endl;*/
		
		force_new = totalForce(nParticles, eps, sigma, pPos );
		for(int i=0; i<nParticles; ++i)
		{
			pVel[i*3+0]+=(initForce[i*3+0]+force_new[i*3+0])*delTime/(2.0*pMass[i]);
			pVel[i*3+1]+=(initForce[i*3+1]+force_new[i*3+1])*delTime/(2.0*pMass[i]);
			pVel[i*3+2]+=(initForce[i*3+2]+force_new[i*3+2])*delTime/(2.0*pMass[i]);
		}
		initForce = force_new;
		
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
		
	}
	
	
	/*cout<<"@@@@"<<endl;
	std::copy(pPos.begin(), pPos.end(), std::ostream_iterator<double>(std::cout, "\t"));
	cout<<endl;*/

}










int main(int argc, char* argv[])
{
	
	
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

	//std::copy(particleConfig.begin(), particleConfig.end(), std::ostream_iterator<string>(std::cout, "\n"));
	
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
	
	//to generate a BaseFile name for VTK output
	/*string fileName = parFile[1];
	string trunc = ".in";
	string::size_type k = fileName.find(trunc);
	if(k!=string::npos)
	{ fileName.erase(k, trunc.length());	}*/
	
	//calling algo
	velocityVerlet(nParticles, pMass, pPos, pVel);
	
	//pMass[0]=stod(particleConfig[1]);
		
		//cout<<"particleConfig_size:"<<particleConfig.size()<<endl;
	/*cout<<"#####"<<endl;
	std::copy(pMass.begin(), pMass.end(), std::ostream_iterator<double>(std::cout, "\t"));
	cout<<"#####"<<endl;*/
	
	//std::copy(pPos.begin(), pPos.end(), std::ostream_iterator<double>(std::cout, "\t"));
	
	/*std::copy(pVel.begin(), pVel.end(), std::ostream_iterator<double>(std::cout, "\t"));
	cout<<"#####"<<endl;*/
		
}
