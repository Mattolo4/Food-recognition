#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/core/utils/filesystem.hpp>
#include <opencv2/calib3d.hpp>
#include <vector>
#include "Utilities.h"
#include <cstdlib>
#include <GL/glut.h>
using namespace cv;
using namespace std;

Mat HSVimg;
void onTrackbarChange(int thresholdValue, void* userData);

int main(void) 
{
    Mat img = imread("./out/food0_1.jpg");
    if (img.empty()) {
        cout << "ERROR: image not found, wrong file name or format" << endl;
        return -1;
    }
    //strongly blur the image
    Size kernel = Size(33, 33);
    GaussianBlur(img,img, kernel, 0);
    namedWindow("Blurred", WINDOW_NORMAL);
    imshow("Blurred", img);
    waitKey(0);
    
    cvtColor(img, HSVimg, COLOR_BGR2HSV);

    //Extract the Hue channel
    vector<Mat> hsvChannels;
    split(HSVimg, hsvChannels);
    //hsvChannels[1] = Scalar(200);

    Mat hueChannel = hsvChannels[0];
    //cout << "hueChannel type : " << hueChannel.depth() << endl;
    int histSize = 30;
    float range[] = { 0, 180 };
    const float* histRange = { range };

    Mat HISTimg;
    calcHist(&hueChannel, 1, nullptr, Mat(), HISTimg, 1, &histSize, &histRange);

    // Order the histogram
    vector<pair<int, float>> histogramData;

    // Populate the vector of pairs of bin index and bin value
    for (int i = 0; i < histSize; i++) {
        float binValue = HISTimg.at<float>(i);
        histogramData.push_back(make_pair(i, binValue));
        //cout << "Bin " << histogramData[i].first << ": " << binValue << endl;
    }

    // Sort the vector of pairs in descending order based on bin values
    sort(histogramData.begin(), histogramData.end(), [](const pair<int, float>& a, const pair<int, float>& b) {
        return a.second > b.second;
    });

    //remove the black pixels in the histogram
    histogramData.erase(histogramData.begin());
    //prints out the values for the histogram
    vector<pair<int, float>> newHist;
    newHist.push_back(histogramData[0]);
    int ind = 0;
    cout << "Bin " << histogramData[0].first*180/histSize << ": " << histogramData[0].second << endl;
    for (int i = 1; i < 10; i++) {
        int Bin = newHist[ind].first*180/histSize;
        int tempBin = histogramData[i].first*180/histSize;
        cout << "ye olde Bin " << histogramData[i].first*180/histSize << ": " << histogramData[i].second << endl;
        if(abs(Bin -tempBin) > 7 && histogramData[i].second > 2000){
            newHist.push_back(histogramData[i]);
            cout << "Bin " << tempBin << ": " << histogramData[i].second << endl;
            ind++;
        }
    } 

    
    // Define the threshold range for the desired hue value
    
    Mat mask1, mask2, mask3;
    Mat HSVimgMod, BGRimgMod;
   
     merge(hsvChannels, HSVimgMod);
    /*cvtColor(HSVimgMod, BGRimgMod, COLOR_HSV2BGR);
    namedWindow("BGRimgMod", WINDOW_NORMAL);
    imshow("BGRimgMod", BGRimgMod);
    waitKey(0); */



    float delta = 5;
    for(int i = 0; i<newHist.size(); i++){  
            namedWindow("mask 1", WINDOW_NORMAL);
            namedWindow("mask 2", WINDOW_NORMAL);
            namedWindow("mask 3", WINDOW_NORMAL);
            float targetHue = static_cast<float>(newHist[i].first)*180/histSize;
            cout << "Target hue : " << targetHue << endl;
            Scalar hsv_lower_l(targetHue-delta, 100, 0);
            Scalar hsv_lower_h(targetHue+delta, 255, 85);
            inRange(HSVimgMod, hsv_lower_l, hsv_lower_h, mask1);
            Scalar hsv_lower_l2(targetHue-delta, 100, 85);
            Scalar hsv_lower_h2(targetHue+delta, 255, 170);
            inRange(HSVimgMod, hsv_lower_l2, hsv_lower_h2, mask2);
            Scalar hsv_lower_l3(targetHue-delta, 100, 170);
            Scalar hsv_lower_h3(targetHue+delta, 255, 255);
            inRange(HSVimgMod, hsv_lower_l3, hsv_lower_h3, mask3);
            imshow("mask 1", mask1);
            imshow("mask 2", mask2);
            imshow("mask 3", mask3);
            waitKey(0);
    }
    

/* 
    int initialThreshold = 0;  // Initial threshold value
    int maxThreshold = 255;   // Maximum threshold value
    namedWindow("Threshold Trackbar");
    createTrackbar("Threshold", "Threshold Trackbar", &initialThreshold, maxThreshold, onTrackbarChange);
    imshow("Original Image", img);
    waitKey(0);  */
/* 
    namedWindow("Plate", WINDOW_NORMAL);
    imshow("Plate", test);
    waitKey(0);  */
 








}
/* 
void onTrackbarChange(int thresholdValue, void* userData)
{
    Mat thresholded;
    inRange(HSVimg, Scalar(thresholdValue, 0, 0), Scalar(thresholdValue, 255, 255), thresholded);

    // Perform further processing or display the thresholded image
    imshow("Thresholded Image", thresholded);
}  */