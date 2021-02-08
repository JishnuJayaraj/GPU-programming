/**
Name: Schmid, Sebastian
Exercise: 1 - Julia Set, OpenCL, v2
Date: 16.05.2018
**/

#define CL_HPP_TARGET_OPENCL_VERSION 200

#include <cstdlib>
#include <iostream>
#include <assert.h>
#include <complex>

#include "lodepng.h"
#include <CL/cl2.hpp>

using namespace std;

void check4Error(cl_int stat)
{
    if(stat != CL_SUCCESS)
    {
        std::cout << "error in openCL caught" << std::endl;
        exit(0);
    }
}

//Taken from lodepng examples
//Example 1
//Encode from raw pixels to disk with a single function call
void encodeOneStep(const char* filename, std::vector<unsigned char>& image, unsigned width, unsigned height)
{
  //Encode the image
  unsigned error = lodepng::encode(filename, image, width, height);

  //if there's an error, display it
  if(error) std::cout << "encoder error " << error << ": "<< lodepng_error_text(error) << std::endl;
}

string kernel_code =
"void kernel simple_iterate(global float* grid, global unsigned int* dim, global float* real, global float* imag, global int *iter, global float *c_re, global float *c_im){"
"int i = 0; int thresh = 10; int iter_thresh = 100; float re_buffer = 0; float im_buffer = 0;"
" int index = get_global_id(0)+get_global_id(1)* (*dim);                          "
"       while(i < iter_thresh && (real[index]*real[index]+imag[index]*imag[index])<= (thresh*thresh)){"
"           re_buffer = real[index]*real[index]-imag[index]*imag[index]+ *c_re;"
"           im_buffer = 2*real[index]*imag[index] + *c_im;"
"                                                       "
"           real[index] = re_buffer; imag[index] = im_buffer;"
"           i++;}"
"iter[index]=i;                "
"                  }";

std::string picture_code=
            "   void kernel picture(global unsigned char* image, global int* it, global int* dim){       "
            "       int index = get_global_id(1) * (*dim)+ get_global_id(0);                                                     "
            "       image[index*4+0]=255*(it[index]/(float)100);"
            "       image[index*4+1]=0;            "
            "       image[index*4+2]=0;            "
            "       image[index*4+3]=255;"
            "   }";

int main(int argc, char* argv[])
{
    unsigned int threadsX = 0;
    unsigned int threadsY = 0;

    if(argc > 2)
    {
        threadsX = atoi(argv[1]);
        threadsY = atoi(argv[2]);

    }
    else
    {
        threadsX = 1;
        threadsY = 1;
    }
    cout << threadsX << " "<<threadsY<<endl;
    assert(threadsX > 0 && threadsY > 0);

    //CL_DEVICE_MAX_WORK_GROUP_SIZE

    std::vector<cl::Platform> all_platforms;
    check4Error(cl::Platform::get(&all_platforms));
    if(all_platforms.size()==0)
    {
        std::cout << "error platform";
        exit(1);
    }
    cl::Platform default_platform = all_platforms[0];
    std::cout<<"using " << all_platforms[0].getInfo<CL_PLATFORM_NAME>()<<std::endl;

    //get Device from platforms
    std::vector<cl::Device> all_devices;
    default_platform.getDevices(CL_DEVICE_TYPE_ALL, &all_devices);
    if(all_devices.size()==0)
    {
        std::cout << "error devices";
        exit(1);
    }
    cl::Device default_device = all_devices[0];
    std::cout<<"using " << all_devices[0].getInfo<CL_DEVICE_NAME>()<<std::endl;
    std::cout<<"Maximum Workgroup Size = " << all_devices[0].getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>()<<std::endl;
    std::cout<<"Ordered Workgroup Size (threadsX*threadsY) = " << threadsX*threadsY << endl;
    assert(threadsX*threadsY <= all_devices[0].getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>());

    //Context, meaning we link context to device (like a "runtime link")
    cl::Context context({default_device});
    //create the program to be executed
    cl::Program::Sources sources;

    sources.push_back({kernel_code.c_str(), kernel_code.length()});
    sources.push_back({picture_code.c_str(), picture_code.length()});

    cl::Program program(context, sources);
    if(program.build({default_device})!=CL_SUCCESS)
    {
        std::cout<<" Error building: "<<program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(default_device)<<"\n";
        exit(1);
    }
    cl::CommandQueue queue(context,default_device);

    /***************************ITERATION DATA for Kernel*********************/

    unsigned int dim = 2048;
    std::complex<float> c = std::complex<float>(-0.8, 0.2);


    float* grid = new float[dim*dim];
    float* real = new float[dim*dim];
    float* imag = new float[dim*dim];
    int* iterations = new int[dim*dim];
    float c_re = c.real();
    float c_im = c.imag();

    int iter = 0;
    //initialize z_0 from -2 to 2 in x&y
    for(unsigned int y = 0; y < dim; y++)
    {
        for(unsigned int x = 0; x < dim; x++)
        {
            real[iter] = -2.0 + x*(4.0/dim);
            imag[iter] = 2.0 - y*(4.0/dim);
            iter++;
        }
    }

    cl::Buffer buffer_grid(context,CL_MEM_READ_WRITE,sizeof(float)*dim*dim);
    cl::Buffer buffer_dim(context,CL_MEM_READ_WRITE,sizeof(int));

    cl::Buffer buffer_real(context,CL_MEM_READ_WRITE,sizeof(float)*dim*dim);
    cl::Buffer buffer_imag(context,CL_MEM_READ_WRITE,sizeof(float)*dim*dim);

    cl::Buffer buffer_iterations(context,CL_MEM_READ_WRITE,sizeof(int)*dim*dim);
    cl::Buffer buffer_c_re(context,CL_MEM_READ_WRITE,sizeof(float));
    cl::Buffer buffer_c_im(context,CL_MEM_READ_WRITE,sizeof(float));

    queue.enqueueWriteBuffer(buffer_grid,CL_TRUE,0,sizeof(float)*(dim)*(dim),grid);
    queue.enqueueWriteBuffer(buffer_dim,CL_TRUE,0,sizeof(int),&dim);

    queue.enqueueWriteBuffer(buffer_real,CL_TRUE,0,sizeof(float)*(dim)*(dim),real);
    queue.enqueueWriteBuffer(buffer_imag,CL_TRUE,0,sizeof(float)*(dim)*(dim),imag);

    queue.enqueueWriteBuffer(buffer_iterations,CL_TRUE,0,sizeof(int)*(dim)*(dim),iterations);
    queue.enqueueWriteBuffer(buffer_c_re,CL_TRUE,0,sizeof(float),&c_re);
    queue.enqueueWriteBuffer(buffer_c_im,CL_TRUE,0,sizeof(float),&c_im);
    /*****************************************************************/

    cl::Kernel simple_iterate(program, "simple_iterate");
    simple_iterate.setArg(0, buffer_grid);
    simple_iterate.setArg(1, buffer_dim);
    simple_iterate.setArg(2, buffer_real);
    simple_iterate.setArg(3, buffer_imag);
    simple_iterate.setArg(4, buffer_iterations);
    simple_iterate.setArg(5, buffer_c_re);
    simple_iterate.setArg(6, buffer_c_im);
    queue.enqueueNDRangeKernel(simple_iterate,cl::NullRange,cl::NDRange({dim, dim}),cl::NDRange({threadsX, threadsY}));
    queue.finish();

    queue.enqueueReadBuffer(buffer_grid,CL_TRUE,0,sizeof(float)*dim*dim,grid);
    queue.enqueueReadBuffer(buffer_iterations,CL_TRUE,0,sizeof(int)*dim*dim,iterations);

//    cout << endl<<endl<<"iter:"<<endl;
//    unsigned int little_couter = 0;
//    for(unsigned int i = 0; i < dim*dim; i++)
//    {
//            cout << iterations[i]<< "\t";
//            little_couter++;
//            if(little_couter == dim)
//            {
//                cout<<endl;
//                little_couter = 0;
//            }
//    }
//        cout << endl;

    /***************************PICTURE STUFF for Kernel**********************/

    const char* filename = "julia.png"; //create file
    unsigned char* image = new unsigned char[dim*dim*4];
    for(unsigned int i = 0; i < (dim*dim*4); i++ )
    {
        image[i] = 0; // initialize
    }

    cl::Buffer buffer_image(context,CL_MEM_READ_WRITE,sizeof(unsigned char)*dim*dim*4);
    queue.enqueueWriteBuffer(buffer_image,CL_TRUE,0,sizeof(unsigned char)*dim*dim*4,image);

    cl::Kernel image_calc(program, "picture");
    image_calc.setArg(0, buffer_image);
    image_calc.setArg(1, buffer_iterations);
    image_calc.setArg(2, buffer_dim);
    queue.enqueueNDRangeKernel(image_calc,cl::NullRange,cl::NDRange({dim, dim}),cl::NDRange({threadsX, threadsY}));
    queue.finish();

    queue.enqueueReadBuffer(buffer_image,CL_TRUE,0,sizeof(unsigned char)*dim*dim*4,image);

    std::vector<unsigned char> vec_image; //will hold the RBGA values
    vec_image.resize(dim*dim* 4);
    vec_image.assign(image, image + (dim * dim * 4));
    encodeOneStep(filename, vec_image, (unsigned)dim, (unsigned) dim);

    /*************************************************************************/

    delete[] grid;
    delete[] real;
    delete[] imag;
    delete[] iterations;
    return 0;
}

/*
string kernel_code =
"void kernel simple_iterate(global int* grid, global unsigned int* dim){"
"         printf(\"\%d \", get_global_id(0));           "
"         printf(\"\%d\\n\", get_global_id(1));  "
"   grid[get_global_id(0)+get_global_id(1)* (*dim)]=get_global_id(0);                "
"
*/
