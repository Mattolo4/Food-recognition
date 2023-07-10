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

Mat GRemove(Mat &image, int delta){
    Mat img = image.clone();
    for(int i=0; i<img.rows; i++){
        for(int j=0; j<img.cols; j++){
            Vec3b pix = img.at<Vec3b>(i,j);
            int avg = round((pix[0] + pix[1] + pix[2])/3);
            if(pix[0]-avg < delta && pix[1]-avg < delta && pix[2]-avg < delta){
                img.at<Vec3b>(i,j) = Vec3b(0,0,0);
            }
        }
    }
    return img;
}

vector<Mat> getPlateMask(const Mat& image){
    Mat img;
    cvtColor(image, img, COLOR_BGR2GRAY);
    vector<Mat> masks;

    medianBlur(img, img, 9);
    vector<Vec3f> circles;
    HoughCircles(img, circles, HOUGH_GRADIENT, 1,220,100, 20, 260, 280);

    vector<Mat> Plates;
    for( size_t i = 0; i < circles.size(); i++ )
    {
        Mat tempMask = Mat::zeros(image.rows, image.cols, CV_8UC3); 
        Vec3i c = circles[i];
        Point center = Point(c[0], c[1]); // circle center
        int radius = c[2];
        circle(tempMask, center, radius, Scalar(255,255,255), -1);
        masks.push_back(tempMask);
        Mat plate;
        image.copyTo(plate, masks[i]);
        Plates.push_back(plate);
    }

    return Plates;
}

int main(void) 
{   
    //assets/tray1/food_image.jpg
    String rootFolder = "../../assets/tray";  
    vector<String> subFolders = {"1", "2", "3", "4", "5", "6", "7", "8"};
    
    vector<Mat> foods;
    vector<vector<Mat>> Plates;
    vector<String> imgPaths;
    
    for(int i = 0; i < subFolders.size(); i++){ //does for all trays
        vector<String> imgPaths;
        String tempPath = rootFolder + subFolders[i]; //save temporary path for current tray
        utils::fs::glob(tempPath, "food_image.jpg",imgPaths); //grabs image name+path
        foods.push_back(imread(imgPaths[0])); //reads images and saves them in foods[]
        /* namedWindow("input test", WINDOW_AUTOSIZE);
        imshow("input test", foods[i]);
        waitKey(0); */
        Plates.push_back(getPlateMask(foods[i])); //for every tray saves the vector containing the plates 
        //Plates[i] contains two images with only one plate per image
    }
    
    string pathInit = "./out/food"; 
    string pathName = to_string(0);
    vector<vector<Mat>> foodsSeg;
    vector<Mat> tempFood;
    Mat temp;
    for(int i = 0; i < foods.size(); i++){
        for(int j =0; j< Plates[i].size(); j++){
            temp = GRemove(Plates[i][j], 20);
            tempFood.push_back(temp);
            pathName = pathInit + to_string(i) +"_"+ to_string(j) + ".jpg";
            imwrite(pathName, tempFood[j]);
            /* namedWindow("output test", WINDOW_AUTOSIZE);
            imshow("output test", temp);
            waitKey(0); */
        }
        foodsSeg.push_back(tempFood);
    }

    return 0;
}
