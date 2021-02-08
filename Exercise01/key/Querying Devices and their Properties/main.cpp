/**
Name: Schmid, Sebastian
Exercise: 1 - Querying Devices and their Properties
Date: 30.04.2018
**/

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <vector>
#include <CL/cl.h>

void checkError(cl_int succ)
{
    if(succ != CL_SUCCESS)
    {
        std::cout << "Error " << succ << std::endl;
        exit(-1);
    }
}

int main(void)
{
    cl_uint platformIdCount = 0;
    checkError(clGetPlatformIDs(0, nullptr, &platformIdCount)); //get no. of platforms
    std::vector<cl_platform_id> platformIds(platformIdCount);
    checkError(clGetPlatformIDs(platformIdCount, platformIds.data(), nullptr)); //get all platforms

    cl_uint deviceIdCount = 0;
    //get number of Devices of first platform
    checkError(clGetDeviceIDs(platformIds[0], CL_DEVICE_TYPE_ALL, 0,nullptr, &deviceIdCount));
    std::vector<cl_device_id> deviceIds (deviceIdCount);
    //get all Devices of first platform
    checkError(clGetDeviceIDs(platformIds[0], CL_DEVICE_TYPE_ALL, deviceIdCount, deviceIds.data(),nullptr));

    char data[1024];
    checkError(clGetDeviceInfo(deviceIds[0], CL_DEVICE_VENDOR, sizeof(data), data, NULL));
        std::cout << "CL_DEVICE_VENDOR: "<< data << std::endl<< std::endl;
    checkError(clGetDeviceInfo(deviceIds[0],CL_DEVICE_EXTENSIONS , sizeof(data), data, NULL));
        std::cout << "CL_DEVICE_EXTENSIONS: "<<data << std::endl<< std::endl;

    unsigned long long mem_size;
    checkError(clGetDeviceInfo(deviceIds[0],CL_DEVICE_GLOBAL_MEM_SIZE , sizeof(cl_ulong), &mem_size, NULL));
        std::cout << "CL_DEVICE_GLOBAL_MEM_SIZE (Bytes): "<<mem_size << std::endl<< std::endl;
   checkError(clGetDeviceInfo(deviceIds[0],CL_DEVICE_LOCAL_MEM_SIZE , sizeof(cl_ulong), &mem_size, NULL));
        std::cout << "CL_DEVICE_LOCAL_MEM_SIZE (Bytes): "<<mem_size << std::endl<< std::endl;

    unsigned int freq;
    checkError(clGetDeviceInfo(deviceIds[0],CL_DEVICE_MAX_CLOCK_FREQUENCY , sizeof(cl_uint), &freq, NULL));
        std::cout << "CL_DEVICE_MAX_CLOCK_FREQUENCY (MHz): " << freq << std::endl;

    unsigned int comp_units;
    checkError(clGetDeviceInfo(deviceIds[0],CL_DEVICE_MAX_COMPUTE_UNITS , sizeof(cl_uint), &comp_units, NULL));
        std::cout << "CL_DEVICE_MAX_COMPUTE_UNITS (parallel compute cores): " << comp_units << std::endl;

    checkError(clGetDeviceInfo(deviceIds[0],CL_DEVICE_MAX_CONSTANT_ARGS , sizeof(cl_uint), &comp_units, NULL));
        std::cout << "CL_DEVICE_MAX_CONSTANT_ARGS (max. no. of arguments with constant qualifier): " << comp_units << std::endl;

    checkError(clGetDeviceInfo(deviceIds[0],CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE , sizeof(cl_ulong), &mem_size, NULL));
        std::cout << "CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE (Bytes): "<<mem_size << std::endl<< std::endl;

    checkError(clGetDeviceInfo(deviceIds[0],CL_DEVICE_MAX_MEM_ALLOC_SIZE , sizeof(cl_ulong), &mem_size, NULL));
        std::cout << "CL_DEVICE_MAX_MEM_ALLOC_SIZE (Bytes): "<<mem_size << std::endl<< std::endl;

    size_t param;
    checkError(clGetDeviceInfo(deviceIds[0],CL_DEVICE_MAX_PARAMETER_SIZE , sizeof(size_t), &param, NULL));
        std::cout << "CL_DEVICE_MAX_PARAMETER_SIZE (Bytes): "<<param << std::endl<< std::endl;

    checkError(clGetDeviceInfo(deviceIds[0],CL_DEVICE_MAX_WORK_GROUP_SIZE , sizeof(size_t), &param, NULL));
        std::cout << "CL_DEVICE_MAX_WORK_GROUP_SIZE (no. of samplers in a kernel): "<<param<< std::endl<< std::endl;

    checkError(clGetDeviceInfo(deviceIds[0],CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS , sizeof(cl_uint), &comp_units, NULL));
        std::cout << "CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS (work item IDs in parallel exec. model): "<<comp_units<< std::endl<< std::endl;

    size_t work_sizes[comp_units];
    checkError(clGetDeviceInfo(deviceIds[0],CL_DEVICE_MAX_WORK_ITEM_SIZES , sizeof(size_t[comp_units]), work_sizes, NULL));
        std::cout << "CL_DEVICE_MAX_WORK_ITEM_SIZES: [";
        // << std::string(work_sizes) << std::endl;

        for(size_t e : work_sizes)
        {
            std::cout<<e<<" ";
        }

        std::cout<<"]"<<std::endl;

    checkError(clGetDeviceInfo(deviceIds[0],CL_DEVICE_NAME , sizeof(data), data, NULL));
        std::cout <<"CL_DEVICE_NAME: " << data << std::endl;

    checkError(clGetDeviceInfo(deviceIds[0],CL_DEVICE_VENDOR , sizeof(data), data, NULL));
        std::cout <<"CL_DEVICE_VENDOR: " << data << std::endl;

    checkError(clGetDeviceInfo(deviceIds[0],CL_DEVICE_VERSION , sizeof(data), data, NULL));
        std::cout << "CL_DEVICE_VERSION: " << data << std::endl;

    checkError(clGetDeviceInfo(deviceIds[0],CL_DRIVER_VERSION , sizeof(data), data, NULL));
        std::cout << "CL_DRIVER_VERSION: " <<  data << std::endl << std::endl;

    checkError(clGetPlatformInfo(platformIds[0],CL_PLATFORM_VENDOR , sizeof(data), data, NULL));
        std::cout << "CL_PLATFORM_VEDOR: " << data << std::endl;

    checkError(clGetPlatformInfo(platformIds[0],CL_PLATFORM_EXTENSIONS , sizeof(data), data, NULL));
        std::cout << "CL_PLATFORM_EXTENSIONS: " << data << std::endl;

	 return 0;
}
