#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/core/utils/filesystem.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <vector>
#include "Utilities.h"
#include <cstdlib>
#include <GL/glut.h>
using namespace cv;
using namespace std;

int diameter = 15;
int sigmaColor = 1;
int sigmaSpace = 1;
const char* window_name = "Filtered Image";
 Mat img, img_gray, out;

static void bilateral(int, void*)
{
    bilateralFilter(img, out, diameter, sigmaColor*0.1, sigmaSpace*0.01);
    imshow( window_name, out);
}



int main(void) 
{
    img = imread("./out/food0_0.jpg");
    if (img.empty()) {
        cout << "ERROR: image not found, wrong file name or format" << endl;
        return -1;
    }
    namedWindow("Input", WINDOW_NORMAL);
    imshow("Input", img);
    waitKey(0);
    out.create( img.size(), img.type() );
    
    //cvtColor( img, img_gray, COLOR_BGR2GRAY );
    namedWindow( window_name, WINDOW_AUTOSIZE );
    createTrackbar( "sigmaColor:", window_name, &sigmaColor, 2000, bilateral );
    setTrackbarMin("sigmaColor:", window_name, sigmaColor);
    createTrackbar( "sigmaSpace:", window_name, &sigmaSpace, 2000, bilateral );
    setTrackbarMin("sigmaSpace:", window_name, sigmaSpace);

    bilateralFilter(img, out, diameter, sigmaColor, sigmaSpace);
    waitKey(0);

    Mat test;
    bilateralFilter(img, test, diameter, 1000*0.1, 800*0.01);
    namedWindow("output test", WINDOW_AUTOSIZE);
    imshow("output test", test);
    waitKey(0);
    return 0;
}
