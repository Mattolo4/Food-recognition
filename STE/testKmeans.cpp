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

 Mat img, out;
int diameter = 15;
int sigmaColor = 100;
int sigmaSpace = 8;


int main(void) 
{
    img = imread("./out/food0_1.jpg");
    if (img.empty()) {
        cout << "ERROR: image not found, wrong file name or format" << endl;
        return -1;
    }
    Mat biFilt, imgHSV;
    bilateralFilter(img, biFilt, diameter, sigmaColor, sigmaSpace);
    /* namedWindow("output test", WINDOW_AUTOSIZE);
    imshow("output test", biFilt);
    waitKey(0); */
    //cvtColor(biFilt, imgHSV, COLOR_BGR2HSV);
    cout << "biFilt shaped: " << biFilt.size()  << endl;
    Mat features = biFilt.reshape(1, biFilt.rows * biFilt.cols);
    features.convertTo(features, CV_32F);
    Mat coords(img.rows*img.cols,2, CV_32F);
    cout << "coords shaped: " << coords.size()  << endl;
    int ind =0 ;
    for (int y = 0; y < img.rows; y++){
		for (int x = 0; x < img.cols; x++){
            if(biFilt.at<Vec3b>(y,x) == Vec3b(0,0,0)){
                coords.at<float>(0,ind) = static_cast<float>(0);
                coords.at<float>(1,ind) = static_cast<float>(0);
            }
            else{
                coords.at<float>(0,ind) = static_cast<float>(y);
                coords.at<float>(1,ind) = static_cast<float>(x);
            }
            ind++;
        }
    }
    
    //REMOVE ALL ROWS OF BLACK PIXEL BUT KEEP EITHER TRACK OF THE ROW OR USE THE COORDINATES

    Mat featureVec;
    hconcat(features,coords,featureVec);
    /* cout << "biFilt shape: " << features.size()  << endl;
    cout << "First line shape: " << features.row(0)  << endl;
    cout << "First colors of pixel: " << biFilt.at<Vec3b>(0,0)  << endl; */
    Mat centers;
    Mat labels;
    int k = 3;
    int attempts = 10;

    TermCriteria criteria(TermCriteria::EPS + TermCriteria::COUNT, 10, 1.0);
	double compactness = kmeans(featureVec, k, labels, criteria, attempts, KMEANS_PP_CENTERS, centers);

    cout << "Centers shape: " << centers.row(1)  << endl;
    cout << "Centers shape: " << centers.size()  << endl;
    /*
    cout << "labels shape: " << labels.size()  << endl;
    cout << "cast first row: " << static_cast<Vec3b>(centers.at<Vec3f>(2))  << endl; */
    Mat segments = Mat::zeros(img.size(), CV_8UC3);
    Mat centerColors = centers(cv::Rect(0, 0, centers.cols - 2, centers.rows)).clone();
    cout << "centerColors shape: " << centerColors.row(2)  << endl;
    cout << "centerColors shape: " << centerColors.size()  << endl;
    
    for(int i = 0; i < features.rows; i++){
        
        int label = static_cast<int>(labels.at<int>(0,i));

        if(label==1){
            segments.at<Vec3b>(i) = static_cast<Vec3b>(centerColors.at<Vec3f>(1));
        }
        if(label==2){
            segments.at<Vec3b>(i) = static_cast<Vec3b>(centerColors.at<Vec3f>(2));
        }
    }
    segments = segments.reshape(3, biFilt.rows);
    cout << "non zeros" << countNonZero(labels)<< endl;
    cout << "type labels" << labels.type() << endl;
    cout << "features reshaped: " << segments.size()  << endl;

    namedWindow("output test", WINDOW_AUTOSIZE);
    imshow("output test", segments);
    waitKey(0);
    return 0;
}

