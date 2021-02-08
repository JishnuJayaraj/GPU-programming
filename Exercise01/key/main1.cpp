/**
Name: Schmid, Sebastian
Exercise: 1 - Julia Set, OpenCL
Date: 10.05.2018
**/

#define CL_HPP_TARGET_OPENCL_VERSION 200


#include <cstdlib>
#include <iostream>
#include <vector>
#include <complex>
#include <ctime>
#include <assert.h>
#include "lodepng.h"
#include <CL/cl2.hpp>

using namespace std;

struct Pixel
{
    public:
    int x;
    int y;
    int iterations = 0; //holds the used no. of iterations
    std::complex<float> value; //holds the value
};

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


void check4Error(cl_int stat)
{
    if(stat != CL_SUCCESS)
    {
        std::cout << "error in openCL caught" << std::endl;
        exit(0);
    }
}

int main(int argc, char* argv[])
{
    const int dim = 3; //also change dim in if-check of picture function!!
    std::complex<float> c = std::complex<float>(-0.8, 0.2);
    int threadsX = 0;
    int threadsY = 0;

    if(argc > 1)
    {
//        threadsX = (int)argv[1];
  //      threadsY = (int)argv[2];
    }
    else
    {
        threadsX = (dim+1)*(dim+1);
        threadsY = 1;
    }

    assert(threadsX > 0 && threadsY > 0);

    //get all Platforms
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


    //Context, meaning we link context to device (like a "runtime link")
    cl::Context context({default_device});

    //create the program to be executed
    cl::Program::Sources sources;

    //create program(kernel) code
    //Id's can go from 0 to get_global_size(0) - 1
    // [get_global_id(0)+get_global_id(1)*dim]]
    std::string kernel_code=
            "   void kernel simple_iterate(global float* re,global float* im, global int* it, global const float *c_re, global const float *c_im){       "
            "   int i = 0; int thresh = 10; int iter = 100; float re_buffer = 0; float im_buffer = 0; int dim = 3;"
            "   while(i <= iter && (re[get_global_id(0)+get_global_id(1)*dim]*re[get_global_id(0)+get_global_id(1)*dim]+im[get_global_id(0)+get_global_id(1)*dim]*im[get_global_id(0)+get_global_id(1)*dim])<= (thresh*thresh))      "
            "   {                                                       "
            "     re_buffer = re[get_global_id(0)+get_global_id(1)*dim]*re[get_global_id(0)+get_global_id(1)*dim]-im[get_global_id(0)+get_global_id(1)*dim]*im[get_global_id(0)+get_global_id(1)*dim]+ *c_re;"
            "                                                       "
            "     im_buffer = 2*re[get_global_id(0)+get_global_id(1)*dim]*im[get_global_id(0)+get_global_id(1)*dim] + *c_im;"
            "           i += 1; re[get_global_id(0)+get_global_id(1)*dim] = re_buffer; im[get_global_id(0)+get_global_id(1)*dim] = im_buffer;}                                    "
            "         it[get_global_id(0)+get_global_id(1)*dim]=i;          "
            "   }";

    //DIMENSIONS have to be checked as fixed number up to now....;
    std::string picture_code=
            "   void kernel picture(global unsigned char* image, global int* it){       "
            "                                                           "
            "       image[get_global_id(0)*4+0]=255*(it[get_global_id(0)]/(float)100);"
            "       image[get_global_id(0)*4+1]=0;            "
            "       image[get_global_id(0)*4+2]=0;            "
            "       image[get_global_id(0)*4+3]=255;"
            "   }";

    //build kernel code, push in sources
    sources.push_back({kernel_code.c_str(), kernel_code.length()});
    sources.push_back({picture_code.c_str(), picture_code.length()});

    //create Program from context and source(with kernel)
    cl::Program program(context, sources);

    if(program.build({default_device})!=CL_SUCCESS)
    {
        std::cout<<" Error building: "<<program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(default_device)<<"\n";
        exit(1);
    }

    ///prepare data for program ///////////////////////////////////////////////////////////////////////////////////////////////////
    //const int dim = 2048; //also change dim in if-check of picture function!!
    //std::complex<float> c = std::complex<float>(-0.8, 0.2);
    //int iteration_limit = 100; //limit for most iteration possible; also used to scale color in picture
    //float threshold = 10;
    //unsigned nBytes = sizeof(Pixel)*(dim+1)*(dim+1);

    //cl::Buffer buffer_px(context,CL_MEM_READ_WRITE,nBytes);
    cl::Buffer buffer_re(context,CL_MEM_READ_WRITE,sizeof(float)*(dim+1)*(dim+1));
    cl::Buffer buffer_im(context,CL_MEM_READ_WRITE,sizeof(float)*(dim+1)*(dim+1));
    cl::Buffer buffer_it(context,CL_MEM_READ_WRITE,sizeof(int)*(dim+1)*(dim+1));
    cl::Buffer buffer_c_re(context,CL_MEM_READ_WRITE,sizeof(float));
    cl::Buffer buffer_c_im(context,CL_MEM_READ_WRITE,sizeof(float));
    cl::Buffer buffer_pic(context,CL_MEM_READ_WRITE,sizeof(unsigned char)*(dim+1)*(dim+1)*4);
    //cl::Buffer buffer_C(context,CL_MEM_READ_WRITE,sizeof(int)*10);

//    std::cout << "Buffer worked"<<std::endl;

    //Pixel array holds all values
    Pixel** px = new Pixel*[dim+1];
    for(int i = 0; i <= dim; ++i)
        px[i] = new Pixel[dim+1];

    //bunch of arrays to hold the values, bc OpenCL does not like classes in the kernel (?)
    float* px_real = new float[(dim+1)*(dim+1)];
    float* px_imag = new float[(dim+1)*(dim+1)];
    int* px_iter = new int[(dim+1)*(dim+1)];
    float c_re = c.real();
    float c_im = c.imag();

    int iter = 0;

    //initialize z_0 from -2 to 2 in x&y
    for(int y = 0; y <= dim; y++)
    {
        for(int x = 0; x <= dim; x++)
        {
            //acc. to dimension, the values are filled w/ by each step from -2 to 2
            px[x][y].value.real(-2.0 + x*(4.0/dim));
            px[x][y].value.imag( 2.0 - y*(4.0/dim));
            px_real[iter] = px[x][y].value.real();
            px_imag[iter] = px[x][y].value.imag();
            iter++;
        }
    }

    for(int i = 0; i < ((dim+1)*(dim+1)); i++ )
    {
        px_iter[i] = -1; // initialize
    }

/////create queue to which we will push commands for the device.
    cl::CommandQueue queue(context,default_device);

    //copy into Buffers and write on memory device
    queue.enqueueWriteBuffer(buffer_re,CL_TRUE,0,sizeof(float)*(dim+1)*(dim+1),px_real);
    queue.enqueueWriteBuffer(buffer_im,CL_TRUE,0,sizeof(float)*(dim+1)*(dim+1),px_imag);
    queue.enqueueWriteBuffer(buffer_it,CL_TRUE,0,sizeof(int)*(dim+1)*(dim+1),px_iter);

    queue.enqueueWriteBuffer(buffer_c_re,CL_TRUE,0,sizeof(float),&c_re);
    queue.enqueueWriteBuffer(buffer_c_im,CL_TRUE,0,sizeof(float),&c_im);

//////execute kernel; 10 represents the no. of threads we want to run
    cl::Kernel simple_iterate(program, "simple_iterate"); //create Kernel
    //set args
    simple_iterate.setArg(0, buffer_re);
    simple_iterate.setArg(1, buffer_im);
    simple_iterate.setArg(2, buffer_it);
    simple_iterate.setArg(3, buffer_c_re);
    simple_iterate.setArg(4, buffer_c_im);
    queue.enqueueNDRangeKernel(simple_iterate,cl::NullRange,cl::NDRange(threadsX),cl::NDRange(threadsY));
    queue.finish();

    //read result the device
    queue.enqueueReadBuffer(buffer_re,CL_TRUE,0,sizeof(float)*(dim+1)*(dim+1),px_real);
    queue.enqueueReadBuffer(buffer_im,CL_TRUE,0,sizeof(float)*(dim+1)*(dim+1),px_imag);
    queue.enqueueReadBuffer(buffer_it,CL_TRUE,0,sizeof(int)*(dim+1)*(dim+1),px_iter);


    iter = 0;
    //initialize z_0 from -2 to 2 in x&y
    for(int y = 0; y <= dim; y++)
    {
        for(int x = 0; x <= dim; x++)
        {
            //acc. to dimension, the values are filled w/ by each step from -2 to 2
            px[x][y].value.real(px_real[iter]);
            px[x][y].value.imag(px_imag[iter]);
            px[x][y].iterations = px_iter[iter];
            iter++;
//            std::cout << x << " " << y << ": " << px[x][y].value.real() << " | " << px[x][y].value.imag() << std::endl;

cout<< "iterations"<< x << " " << y << ": " <<px[x][y].iterations << endl;
        }
    }
    

    /**Drawing the picture*/
    const char* filename = "julia.png"; //create file
    //std::vector<unsigned char> image; //will hold the RBGA values
    //image.resize((dim+1) * (dim+1) * 4);
    unsigned char* image = new unsigned char[(dim+1) * (dim+1) * 4];
    for(int i = 0; i < ((dim+1)*(dim+1)*4); i++ )
    {
        image[i] = 0; // initialize
    }

    cl::Kernel picture(program, "picture"); //create Kernel

    queue.enqueueWriteBuffer(buffer_pic,CL_TRUE,0,sizeof(unsigned char)*(dim+1)*(dim+1)*4,image);

    picture.setArg(0, buffer_pic);
    picture.setArg(1, buffer_it);
    queue.enqueueNDRangeKernel(picture,cl::NullRange,cl::NDRange(threadsX),cl::NDRange(threadsY));
    queue.finish();

    queue.enqueueReadBuffer(buffer_pic,CL_TRUE,0,sizeof(unsigned char)*(dim+1)*(dim+1)*4,image);

    std::vector<unsigned char> vec_image; //will hold the RBGA values
    vec_image.resize((dim+1) * (dim+1) * 4);
    vec_image.assign(image, image + ((dim+1) * (dim+1) * 4));
    cout<<"##############"<<endl;
    for(int i=0;i<vec_image.size();++i)
    {
    	cout<<int(vec_image[i])<<"\t";
    }
    cout<<endl;
    encodeOneStep(filename, vec_image, (unsigned)dim, (unsigned) dim);

//////////////////////////////////////////////////////////////////////////////////////
//    cleaning array up
    for(int i = 0; i <= dim; ++i)
        delete px[i];

    delete[] px;
    delete[] px_real;
    delete[] px_imag;
    delete[] px_iter;
    delete[] image;

    return 0;
}
