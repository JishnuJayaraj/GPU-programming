#include<iostream>
#include<vector>
#include"lodepng.h"
//#include"julia.cuh"
#include <cstdlib>
#include<cuda_runtime.h>

using namespace std;

void checkError(cudaError_t err)
{
    if (err != cudaSuccess)
    {
        //print a human readable error message
        std::cout << cudaGetErrorString(err) << std::endl;
        exit(-1);
    }
}

void encodeOneStep(const char* filename, std::vector<unsigned char>& image, unsigned width, unsigned height)
{
  //Encode the image
  unsigned error = lodepng::encode(filename, image, width, height);

  //if there's an error, display it
  if(error) std::cout << "encoder error " << error << ": "<< lodepng_error_text(error) << std::endl;
}

class ComplexNr
{
	private:
	float r, i;
	
	public:
	__device__ ComplexNr(float m, float n) {r=m; i=n; }
	
	__device__ ComplexNr operator* (const ComplexNr& t)
	{
		return ComplexNr(r*t.r - i*t.i, r*t.i+i*t.r);
	} 
	__device__ ComplexNr operator+ (const ComplexNr& t)
	{
		return ComplexNr(r+t.r, i+t.i);
	}
	
	__device__ int magnitudeSq()
	{
		return (r*r+i*i); 
	}


};




__device__ int julia(int x, int y, unsigned d_width, unsigned d_height )
{
	float rx = 2.0*((d_width/2.0)-x)/(d_width/2.0);
	float iy = 2.0*((d_height/2.0)-y)/(d_height/2.0);
	
	ComplexNr a(rx, iy);
	ComplexNr c(-0.8, 0.2);
	
	int nItr=0;
	while (a.magnitudeSq()<100)
	{
		a = (a * a) + c;
		nItr++;		
	}
	
	return nItr;
}

/*__global__ void dummyVec(int* dummy)
{
	int Idx =blockIdx.x;
	
	dummy[Idx]=blockIdx.x;
}*/

__global__ void fillImage(unsigned char* d_image, unsigned* d_width, unsigned* d_height)
{
	//int Idx = threadIdx.x+blockDim.x*blockIdx.x;
	//int Idy = threadIdx.y+blockDim.y*blockIdx.y;
	int Idx =blockIdx.x;
	int Idy =blockIdx.y;
	
		   // int nItr = 0;
			int juliaVal= julia(Idx,Idy, *d_width, *d_height);
			//cout<<juliaVal<<"\t";
			d_image[(Idx+Idy*(*d_height))*4+0]=juliaVal*255;
			d_image[(Idx+Idy*(*d_height))*4+1]=0;
			d_image[(Idx+Idy*(*d_height))*4+2]=0;
			d_image[(Idx+Idy*(*d_height))*4+3]=255;    //opacity
			
}




int main(int argc,char *argv[])
{
	unsigned width=1024, height=1024;
	const char* filename = argc > 1 ? argv[1] : "juliaGPU.png";
	vector<unsigned char> image (width*height*4, 0);
	
	dim3 grid(width,height,1);
	
	unsigned* d_width;
	unsigned* d_height;
	unsigned char* d_image;
	
	checkError(cudaMalloc(&d_width, width*sizeof(unsigned)));
	checkError(cudaMalloc(&d_height, height*sizeof(unsigned)));
	checkError(cudaMalloc(&d_image, width*height*4*sizeof(unsigned)));
	
	checkError(cudaMemcpy(d_width, &width, width*sizeof(unsigned), cudaMemcpyHostToDevice));
	//cout<<"asdfsdf0"<<endl;
	checkError(cudaMemcpy(d_height, &height, height*sizeof(unsigned), cudaMemcpyHostToDevice));
	//cout<<"asdfsdf1"<<endl;
	checkError(cudaMemcpy(d_image, &image[0], width*height*4*sizeof(unsigned char), cudaMemcpyHostToDevice));
	//cout<<"asdfsdf2"<<endl;
	
	/*vector<int> dummy (width*height,0);
	int* d_dummy;
	
	checkError(cudaMalloc(&d_dummy, width*height*sizeof(int)));
	checkError(cudaMemcpy(d_dummy, &dummy[0], width*height*sizeof(int), cudaMemcpyHostToDevice));
	
	dummyVec<<<grid,1>>>(d_dummy);
	
	checkError(cudaMemcpy(&dummy[0], d_dummy, width*height*sizeof(int), cudaMemcpyDeviceToHost));
	for(int i=0;i<width*height;++i)
	{
		cout<<dummy[i]<<"\t";
	}
	cout<<endl;*/
	fillImage<<<grid,1>>>(d_image, d_width, d_height);
	
	checkError(cudaMemcpy(&image[0], d_image, width*height*4*sizeof(unsigned char), cudaMemcpyDeviceToHost));
	
	/*for(int i=0;i<32;++i)
	{
		for(int j=0;j<32;j+=1)
	{
		cout<<int(image[i*32*4+j*4])<<",";		
	}		
	cout<<endl;	
	}*/
	cout<<endl;
	checkError(cudaFree(d_width));
	checkError(cudaFree(d_height));
	checkError(cudaFree(d_image));
	
	encodeOneStep(filename, image, width, height);
		
	return 0;


}
