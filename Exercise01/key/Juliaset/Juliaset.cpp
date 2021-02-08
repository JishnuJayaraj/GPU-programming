//============================================================================
// Name        : Juliaset.cpp
// Author      : 
// Version     :
// Copyright   : Your copyright notice
// Description : Hello World in C, Ansi-style
//============================================================================
#include <iostream>
#include <fstream>
#include <math.h>

using namespace std;

//Size of image
const int imageWidth = 512, imageHeight = 512;

//Shape of Julia Set

//const float ca = -0.70176 , cb = -0.3842;
//const float ca = 0.285 , cb = 0.01;
//const float ca = 0.70176 , cb=-0.3842;
const float ca = -0.2 , cb=0.8;


int findJulia(double cr, double ci, int max_iterations) {
	int i = 0;
	double zr =cr, zi = ci;
	while (i < max_iterations && zr * zr + zi * zi < 16.0) {
		double temp = zr * zr - zi * zi + ca;
		zi = 2.0 * zr * zi + cb;
		zr = temp ;
		i++;
	}
		return i;
}

double mapToReal(int x, int imageWidth, double minR, double maxR) {
	double range = maxR - minR;
	return x * (range / imageWidth) + minR;

}

double mapToImaginary(int y, int imageHeight, double minI, double maxI) {
	double range = maxI - minI;
	return y * (range / imageHeight) + minI;
}

int main() {
	int maxN = 255; // Max of N in z_n Julia set formula
	double minR = -2.0, maxR = 2.0;
	double minI = -2.0, maxI = 2.0;
	ofstream fout("image.ppm");
	fout << "P3" << endl; //see PPM file
	fout << imageWidth << " " << imageHeight << endl; // dimensions
	fout << "256" << endl; // max value of a pixel R,G,B

	for (int y = 0; y < imageHeight; y++) { // rows
		for (int x = 0; x < imageWidth; x++) { // Pixels in row
			double cr = mapToReal(x, imageWidth, minR, maxR);
			double ci = mapToImaginary(y, imageHeight, minI, maxI);
			cout << "" << cr << endl;
			cout << "" << ci << endl;

			int n = findJulia(cr, ci, maxN);

			int r = ((n*n) % 255); // change for colors
			int g = ((n*n) % 255); // change for colors
			int b = (n % 255); // change for colors
//			int r = ((int)(n * sinf(n)) % 256); // change for colors
//			int g = ((n * n) % 256); // change for colors
//			int b = ((n * n) % 256); // change for colors
			fout << r << " " << g << " " << b << " ";
		}
		fout << endl;
	}
	fout.close();
	return 0;
}
