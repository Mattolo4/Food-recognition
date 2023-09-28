#ifndef UTILITIES
#define UTILITIES

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/core/utils/filesystem.hpp>
#include <opencv2/calib3d.hpp>
#include "Classes.h"
#include <map>

using namespace cv;
using namespace std;

Mat GRemove(Mat &image, int delta);

//Histogram equalization for three channel image
Mat equalize(Mat img);

//PLATE SEGMENTATION
vector<Mat> getPlateMask(const Mat& unfiltered, const Mat& image);
Mat blobDelete(Mat mask, int size);

//SALAD SEGMENTATION
Mat getSaladMask(const Mat& unfiltered);
Mat cutSalad(const Mat& salad);

//BEAN SEGMENTATION

Mat beanSelector(Mat beanPlate);
Mat eqbBeanSelector(Mat beanPlate);
Mat treshBeans(Mat beans, int tresh);
Mat segBeans( Mat beanPlate, Mat trayImg );

//POTATOES SEGMENTATION

Mat segPotatoes(Mat plateImg);

//MAIN COURSE SEGMENTATION
Mat segMain(Mat &plate, int &ID);
Mat segSides(Mat &plateImg, int &ID, Mat trayImg);
Mat segMainSecond(Mat PlateNOSides);
Mat segSidesFirst(plate main, Mat& trayImg);

//FIRST COURSE SEGMENTATION
Mat segFirst(Mat &first);

// BREAD SEGMENTATION
Mat getBreadMask(const Mat& img);
Mat growBread(Mat Mask, const Mat& img);
Mat segBread(Mat &trayImage);
Mat BreadQuantize(Mat img);
Mat breadCleanup(Mat quantizedBread);
map <int, vector<int>> resizeBBox(Mat mask, map <int, vector<int>> oldBBox, int ID);

//BOUNDING BOXES CONSTRUCTION
map <int, vector<int>> getBBox(Mat mask, int ID);
void printBBox(Mat img, map <int, vector<int>> bbox, int ID);
Mat squareBreadImage(food bread,const Mat& img);

//create square mask for food recognition
Mat squareMask(Mat colorMask);

//FOOD recognition
plate recognizeFood(Mat plateMask);

//creates a mask with the ID as color
Mat colorID(Mat mask, int ID);
vector<Mat> generateFoodMasks(tray inputTray);
Mat mergeMasks(vector<Mat> Masks);

//Generate the complete tray
tray generateTray(const Mat& image);
tray generateLeftoverTray(const Mat& image, tray initialTray);

//temporary davide functions
bool isThereSalad(Mat saladImage);
bool isThereBread(Mat breadImage);
plate recognizeFood(Mat squarePlate, int plateN);

#endif // UTILITIES