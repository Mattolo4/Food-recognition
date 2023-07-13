#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/core/utils/filesystem.hpp>
#include <opencv2/calib3d.hpp>
#include "Utilities.h"
using namespace cv;
using namespace std;


int main(void) 
{
    String rootFolder = "../Food_leftover_dataset/tray";  
    vector<String> subFolders = {"1", "2", "3", "4", "5", "6", "7", "8"};
    
    vector<Mat> foods;
    vector<vector<Mat>> Plates;
    vector<String> imgPaths;
    
    for(int i = 0; i < subFolders.size(); i++){
        vector<String> imgPaths;
        String tempPath = rootFolder + subFolders[i];
        utils::fs::glob(tempPath, "food_image.jpg",imgPaths);
        foods.push_back(imread(imgPaths[0]));
        Plates.push_back(getPlateMask(foods[i]));
    }
    /*
    vector<Mat> hist = getHist(Plates[1]);
    Mat histogram = printHist(hist);
    imshow("Histogram", histogram);
    waitKey(0);
    */
    string pathInit = "./out/food"; 
    string pathName = to_string(0);
    Mat removed0, removed1;
    for(int i = 0; i < foods.size(); i++){
        namedWindow("Plate 1", WINDOW_NORMAL);
        namedWindow("Plate 2", WINDOW_NORMAL);
        moveWindow("Plate 1", 0, 0);
        moveWindow("Plate 2", foods[0].cols/2, 0);
        removed0 = GRemove(Plates[i][0], 20);
        removed1 = GRemove(Plates[i][1], 20);
        imshow("Plate 1", removed0);
        imshow("Plate 2", removed1);
        pathName = pathInit + to_string(i) +"_0.jpg";
        cout << pathName;
        waitKey(0);
        
        imwrite(pathName, removed0);
        pathName = pathInit + to_string(i) +"_1.jpg";
        imwrite(pathName, removed1);
    }
    

    return 0;
}
