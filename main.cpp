#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/core/utils/filesystem.hpp>
#include <opencv2/calib3d.hpp>
#include "Utilities/Classes.h"
#include "Utilities/Utilities.h"
#include "Utilities/mAP.h"
#include <vector>
#include <string>
using namespace cv;
using namespace std;

int main(){
    
    string food_try;
    cout << "please insert the path of interessed food_image from build folder";
    cin >> food_try;
    cout << "The path selected is: " << food_try << endl;
    Mat inputImage;
    try
    {
        inputImage = imread(food_try);
        if (inputImage.empty())
        {
            throw invalid_argument("The img is empty!");
        }
    }
    catch (invalid_argument &e)
    {
        cout << "Exception: " << e.what() << endl;
        return 1;
    }
    
    tray initialTray = generateTray(inputImage);
    vector<Mat> predictedMasks = generateFoodMasks(initialTray);
    Mat allMasks = mergeMasks(predictedMasks);
    
    for(int i = 0; i<allMasks.rows; i++){
            for(int j = 0; j<allMasks.cols; j++){
                int pix = allMasks.at<uchar>(i,j);
                if(pix == 0){
                    allMasks.at<uchar>(i,j) = 125;
                }
            }
        }
    
    imshow("allMasksWhiteBack ", allMasks);
    waitKey(0);

    string lo_path;
    cout << "please insert the path of interessed leftover from build folder";
    cin >> lo_path;
    cout << "The path selected is: " << lo_path << endl;
    Mat leftoverImage;
    try
    {
        leftoverImage = imread(food_try);
        if (leftoverImage.empty())
        {
            throw invalid_argument("The img is empty!");
        }
    }
    catch (invalid_argument &e)
    {
        cout << "Exception: " << e.what() << endl;
        return 1;
    }
    tray leftoverTray = generateLeftoverTray(leftoverImage,initialTray);

    vector<Mat> predictedMasks_l = generateFoodMasks(leftoverImage);
    Mat allMasks_l = mergeMasks(predictedMasks_l);
    
    for(int i = 0; i<allMasks_l.rows; i++){
            for(int j = 0; j<allMasks_l.cols; j++){
                int pix = allMasks_l.at<uchar>(i,j);
                if(pix == 0){
                    allMasks_l.at<uchar>(i,j) = 125;
                }
            }
        }
    
    imshow("allMasksWhiteBack ", allMasks_l);
    waitKey(0);

    string lo_path;
    cout << "please insert the path of interessed mask asset folder";
    cin >> lo_path;

    Mat maskGT = imread(lo_path  + "masks/food_image.png");
    // computes the IoU 
    for(const auto& mask: predictedMasks){
        float ratio = getIoU_fromMasks(maskGT, mask, false);
        cout << ratio;
    }

    return 0;
}