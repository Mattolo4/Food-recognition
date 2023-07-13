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

Mat img, img_gray;
Mat out, out_edges, blurred;
int lowThreshold = 0;
int highThreshold = 0;
const int max_lowThreshold = 100;
const int max_highThreshold = 200;
const int ratio_ = 3;
const int kernel_size = 3;
const char* window_name = "Edge Map";

static void CannyThreshold(int, void*)
{
    blur( img_gray, blurred, Size(3,3) );
    Canny( blurred, out_edges, lowThreshold, highThreshold+max_lowThreshold, kernel_size );
    imshow( window_name, out_edges);
}



int main(void) 
{
    Mat img = imread("./out/food0_1.jpg");
    if (img.empty()) {
        cout << "ERROR: image not found, wrong file name or format" << endl;
        return -1;
    }
    namedWindow("Input", WINDOW_NORMAL);
    imshow("Input", img);
    waitKey(0);
    out.create( img.size(), img.type() );
    cvtColor( img, img_gray, COLOR_BGR2GRAY );
    namedWindow( window_name, WINDOW_AUTOSIZE );
    createTrackbar( "Min Threshold:", window_name, &lowThreshold, max_lowThreshold, CannyThreshold );
    createTrackbar( "Max Threshold:", window_name, &highThreshold, max_highThreshold, CannyThreshold );
    CannyThreshold(0, 0);
    waitKey(0);
    return 0;
}
