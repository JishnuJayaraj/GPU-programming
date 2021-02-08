/**
Name: Schmid, Sebastian
Exercise: 1 - Julia Set, CPU
Date: 01.05.2018
**/

#include <cstdlib>
#include <iostream>
#include <vector>
#include <complex>
#include <ctime>
#include "lodepng.h"

//#define DEBUG

class Pixel
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

int main()
{
    const clock_t begin_time = clock(); // start timer

    //picture shall have dimension 2048 * 2048
    const int dim = 2048;
    //Pixel array holds all values
    Pixel** px = new Pixel*[dim+1];
    for(int i = 0; i <= dim; ++i)
        px[i] = new Pixel[dim+1];

    #ifdef DEBUG
    std::cout << "Initializing px"<<std::endl;
    #endif

    //initialize z_0 from -2 to 2 in x&y
    for(int y = 0; y <= dim; y++)
    {
        for(int x = 0; x <= dim; x++)
        {
            //acc. to dimension, the values are filled w/ by each step from -2 to 2
            px[x][y].value.real(-2.0 + x*(4.0/dim));
            px[x][y].value.imag( 2.0 - y*(4.0/dim));
        }
    }

    std::complex<float> z_n1; //this will hold the new value
    std::complex<float> c = std::complex<float>(-1.15, 0.29);//(-1.0, 0.3) //(-0.7, 0.3)
    //std::complex<float> c = std::complex<float>(-0.8,0.2);
    int iteration_limit = 100; //limit for most iteration possible; also used to scale color in picture
    float threshold = 10; //maximum value for z_n1

    #ifdef DEBUG
    std::cout << "Iterating over all px" <<std::endl;
    #endif // DEBUG

    //iterate over each px until threshold or iteration maximum is hit
    for(int y = 0; y <= dim; y++)
    {
        for(int x = 0; x <= dim; x++)
        {
            //iterate over every px only until limit && absolute value not over threshold
            //check abs before starting
            float abs = sqrt(px[x][y].value.real() * px[x][y].value.real() + px[x][y].value.imag() * px[x][y].value.imag());

            while(px[x][y].iterations < iteration_limit && abs <= threshold)
            {
                //iteration rule acc. to ex. sheet
                z_n1 = (px[x][y].value * px[x][y].value) + c;

                //save z_n1 in px_value
                px[x][y].value = z_n1;

                //Calculate new absolute value and increase iteration counter
                abs = sqrt(px[x][y].value.real() * px[x][y].value.real() + px[x][y].value.imag() * px[x][y].value.imag());
                px[x][y].iterations++; //increase iteration counter
            }
            #ifdef DEBUG
            std::cout << "Finished px "<< x<<" "<< y<< " iter: "<< px[x][y].iterations <<std::endl;
            #endif // DEBUG
        }
    }

    #ifdef DEBUG
    std::cout << "Finished px iteration"<<std::endl;
    std::cout << "Starting picture drawing"<<std::endl;
    #endif // DEBUG

    const char* filename = "julia.png"; //create file
    std::vector<unsigned char> image; //will hold the RBGA values
    image.resize((dim+1) * (dim+1) * 4);

    //create color values for img
    for(int y = 0; y <= dim; y++)
    {
        for(int x = 0; x <= dim; x++)
        {
            //relative coloring w.r.t. maximum iteration; channels are RGBA
            image[4 * dim * y + 4 * x + 2] = (unsigned char)(255 * (px[x][y].iterations/(float)iteration_limit));
            image[4 * dim * y + 4 * x + 1] = 10;
            image[4 * dim * y + 4 * x + 0] = 0;
            image[4 * dim * y + 4 * x + 3] = 255;
        }
    }

    //create img; taken from lodepng
    encodeOneStep(filename, image, (unsigned)dim, (unsigned) dim);

    float end_time = float( clock () - begin_time ); //stop time
    std::cout << "Took process " <<end_time/  CLOCKS_PER_SEC << " ms"<<std::endl;

    #ifdef DEBUG
    std::cout << "Finished picture drawing"<<std::endl;
    #endif // DEBUG

    //cleaning array up
    for(int i = 0; i <= dim; ++i)
        delete px[i];

    delete[] px;

    return 0;
}
