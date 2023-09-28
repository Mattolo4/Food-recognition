#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/core/utils/filesystem.hpp>
#include <opencv2/calib3d.hpp>
#include "Classes.h"
#include "Plate_rec.hpp"
#include <map>
using namespace cv;
using namespace std;

// temporary DAVIDE functions

plate recognizeFood(Mat squarePlate, int plateN)
{

    vector<food> foods;
    if (plateN == 0)
    {
        food main(7), side(11);
        foods.push_back(main);
        foods.push_back(side);
    }
    else
    {
        food primo(5);
        foods.push_back(primo);
    }
    plate tePlate;
    tePlate.foods = foods;
    return tePlate;
}

bool isThereBread(Mat breadImage)
{
    return true;
}

bool isThereSalad(Mat saladImage)
{
    return true;
}

// Histogram equalization for three channel image
Mat equalize(Mat img)
{
    Mat channels[3];
    split(img, channels);

    for (int i = 0; i < 3; ++i)
    {
        equalizeHist(channels[i], channels[i]);
    }
    Mat out;
    merge(channels, 3, out);
    return out;
}

// BOUNDING BOXES CONSTRUCTION
map<int, vector<int>> getBBox(Mat mask, int ID)
{
    Mat bwmask;
    if (mask.type() == 16)
    {
        cvtColor(mask, bwmask, COLOR_BGR2GRAY);
    }
    else
    {
        bwmask = mask;
    }
    map<int, vector<int>> bboxes;
    int firstRow, lastRow, firstCol, lastCol;
    Mat Points;
    findNonZero(bwmask, Points);
    Rect bbox = boundingRect(Points);
    int x = bbox.x;
    int y = bbox.y;
    int width = bbox.width;
    int height = bbox.height;
    bboxes.insert({ID, {x, y, width, height}});
    return bboxes;
}

void printBBox(Mat img, map<int, vector<int>> bbox, int ID)
{
    vector<int> tempBBox = bbox[ID];
    Point tl(tempBBox[0], tempBBox[1]);
    Point br(tempBBox[0] + tempBBox[2] - 1, tempBBox[1] + tempBBox[3] - 1);
    Scalar color;
    if (ID < 6)
    {
        color = Scalar(255, 0, 0);
    } // pasta/rice
    else if (ID < 10)
    {
        color = Scalar(0, 255, 0);
    } // meat/fish
    else if (ID == 10)
    {
        color = Scalar(255, 255, 0);
    } // beans
    else if (ID == 11)
    {
        color = Scalar(255, 0, 255);
    } // potatoes
    else if (ID == 12)
    {
        color = Scalar(0, 255, 255);
    } // salad
    else
    {
        color = Scalar(0, 0, 255);
    } // bread

    rectangle(img, tl, br, color, 3);
}

// remove white from plates
Mat GRemove(Mat &image, int delta)
{
    Mat img = image.clone();
    for (int i = 0; i < img.rows; i++)
    {
        for (int j = 0; j < img.cols; j++)
        {
            Vec3b pix = img.at<Vec3b>(i, j);
            int avg = round((pix[0] + pix[1] + pix[2]) / 3);
            if (pix[0] - avg < delta && pix[1] - avg < delta && pix[2] - avg < delta)
            {
                img.at<Vec3b>(i, j) = Vec3b(0, 0, 0);
            }
        }
    }
    return img;
}
// PLATE SEGMENTATION
vector<Mat> getPlateMask(const Mat &unfiltered, const Mat &image)
{
    Mat img;
    cvtColor(image, img, COLOR_BGR2GRAY);

    vector<Mat> masks;

    medianBlur(img, img, 9);
    vector<Vec3f> circles;
    HoughCircles(img, circles, HOUGH_GRADIENT, 1, 220, 100, 20, 260, 280);

    vector<Mat> Plates;
    for (size_t i = 0; i < circles.size(); i++)
    {
        Mat tempMask = Mat::zeros(image.rows, image.cols, CV_8UC3);
        Vec3i c = circles[i];
        Point center = Point(c[0], c[1]); // circle center
        int radius = c[2];
        circle(tempMask, center, radius, Scalar(255, 255, 255), -1);
        masks.push_back(tempMask);
        Mat plate;
        unfiltered.copyTo(plate, masks[i]);
        int x = max(c[0] - c[2], 0);
        int y = max(c[1] - c[2], 0);
        int width = min(c[0] + c[2], plate.cols) - x;
        int height = min(c[1] + c[2], plate.rows) - y;
        // Plates.push_back(plate(Rect(x, y, width, height)));
        Plates.push_back(plate);
    }

    return Plates;
}

Mat blobDelete(Mat mask, int size)
{
    Mat labels, stats, centroids;
    int compNum = connectedComponentsWithStats(mask, labels, stats, centroids);
    for (int i = 0; i < compNum; i++)
    {
        int compSize = stats.at<int>(i, CC_STAT_AREA);
        if (compSize < size * size)
        {
            Mat delMask;
            delMask = labels == i;
            mask.setTo(0, delMask);
        }
    }
    // imwrite("bboxMask.jpg", mask);
    return mask;
}

// SALAD SEGMENTATION
Mat getSaladMask(const Mat &unfiltered)
{
    Mat filt;
    bilateralFilter(unfiltered, filt, 15, 100, 8);
    Mat img;
    cvtColor(filt, img, COLOR_BGR2GRAY);
    medianBlur(img, img, 9);
    vector<Vec3f> circles;
    HoughCircles(img, circles, HOUGH_GRADIENT, 1, 220, 100, 20, 180, 210);
    Mat mask = Mat::zeros(unfiltered.rows, unfiltered.cols, CV_8UC3);
    Vec3i c = circles[0];
    Point center = Point(c[0], c[1]); // circle center
    int radius = c[2];
    circle(mask, center, radius, Scalar(255, 255, 255), -1);

    Mat salad;
    unfiltered.copyTo(salad, mask);
    /*
    imshow("test salad square mask",salad(Rect(x, y, width, height)));
    waitKey(0);*/

    return salad;
}

Mat cutSalad(const Mat &salad)
{
    Mat saladHSV;
    cvtColor(salad, saladHSV, COLOR_BGR2HSV);
    // Extract the Hue, Saturation, Value channels
    vector<Mat> ChannelsHSV;
    split(saladHSV, ChannelsHSV);
    Mat satChannel = ChannelsHSV[1];
    Mat mask;
    inRange(satChannel, 150, 255, mask);
    /* imshow("Dirty mask",mask);
    waitKey(0);  */

    Mat element = getStructuringElement(MORPH_RECT, Size(7, 7));
    for (int i = 0; i < 20; i++)
    {
        dilate(mask, mask, element);
        erode(mask, mask, element);
    }
    floodFill(mask, Point(0, 0), Scalar(125));
    inRange(mask, 123, 125, mask);
    Mat invMask;
    bitwise_not(mask, invMask);

    Mat fmask = blobDelete(invMask, 15);

    Mat cut;
    salad.copyTo(cut, fmask);
    /* imshow("Obtained mask",fmask);
    waitKey(0);
    imshow("only salad",cut);
    waitKey(0);  */
    return fmask;
}

// BEAN SEGMENTATION

Mat beanSelector(Mat beanPlate)
{
    Mat imgHSV, out;
    cvtColor(beanPlate, imgHSV, COLOR_BGR2HSV);
    vector<Mat> ChannelsHSV;
    split(imgHSV, ChannelsHSV);
    Mat hueChannel = ChannelsHSV[0];
    Mat satChannel = ChannelsHSV[1];
    Mat valChannel = ChannelsHSV[2];
    Mat mask, hsmask, hueMask, satMask, valMask;
    // 5,13 50,255 7,247
    // 0,180 58,127 40,255 equalized
    int lHue = 5;
    int hHue = 13;
    int lSat = 50;
    int hSat = 255;
    int lVal = 7;
    int hVal = 247;

    inRange(hueChannel, lHue, hHue, hueMask);
    inRange(satChannel, lSat, hSat, satMask);
    inRange(valChannel, lVal, hVal, valMask);
    bitwise_and(hueMask, satMask, hsmask);
    bitwise_and(valMask, hsmask, mask);
    beanPlate.copyTo(out, mask);
    return out;
}

Mat eqbBeanSelector(Mat beanPlate)
{
    Mat imgHSV, out;
    cvtColor(beanPlate, imgHSV, COLOR_BGR2HSV);
    vector<Mat> ChannelsHSV;
    split(imgHSV, ChannelsHSV);
    Mat hueChannel = ChannelsHSV[0];
    Mat satChannel = ChannelsHSV[1];
    Mat valChannel = ChannelsHSV[2];
    Mat mask, hsmask, hueMask, satMask, valMask;
    // 0,180 51,130 54,255
    int lHue = 0;
    int hHue = 180;
    int lSat = 51;
    int hSat = 130;
    int lVal = 54;
    int hVal = 255;

    inRange(hueChannel, lHue, hHue, hueMask);
    inRange(satChannel, lSat, hSat, satMask);
    inRange(valChannel, lVal, hVal, valMask);
    bitwise_and(hueMask, satMask, hsmask);
    bitwise_and(valMask, hsmask, mask);
    beanPlate.copyTo(out, mask);
    return out;
}

Mat treshBeans(Mat beans, int tresh)
{

    Mat treshMask;
    threshold(beans, treshMask, tresh, 255, THRESH_BINARY);
    Mat out;
    beans.copyTo(out, treshMask);
    return out;
}

Mat segBeans(Mat beanPlate, Mat trayImg)
{
    Mat onlyBeans = beanSelector(beanPlate);
    Mat onlyBeansMask, eqOnlyBeans;
    cvtColor(onlyBeans, onlyBeansMask, COLOR_BGR2GRAY);
    inRange(onlyBeansMask, 2, 255, onlyBeansMask);
    equalize(trayImg).copyTo(eqOnlyBeans, onlyBeansMask);
    Mat equalizedBeans = eqbBeanSelector(eqOnlyBeans);
    Mat grayBeans;
    cvtColor(equalizedBeans, grayBeans, COLOR_BGR2GRAY);
    Mat beanMask;
    inRange(grayBeans, 10, 255, beanMask);
    Mat blobbedBeans = blobDelete(beanMask, 10);
    Mat element = getStructuringElement(MORPH_RECT, Size(3, 3));
    Mat dilatedBeans = blobDelete(blobbedBeans, 10);
    Mat labels, stats, centroids;
    vector<Mat> componentMasks;
    int compNum = connectedComponentsWithStats(blobbedBeans, labels, stats, centroids);
    for (int i = 0; i < compNum; i++)
    {
        int compSize = stats.at<int>(i, CC_STAT_AREA);
        if (compSize < 100 * 100)
        {
            Mat componentMask = (labels == i);
            componentMasks.push_back(componentMask);
        }
    }
    Mat finalBeanMask = Mat::zeros(beanPlate.rows, beanPlate.cols, CV_8U);
    ;
    for (int i = 1; i < componentMasks.size(); i++)
    {

        Scalar meanValue = mean(onlyBeans, componentMasks[i]);
        int meanValueB = static_cast<int>(meanValue[0]);
        int meanValueG = static_cast<int>(meanValue[1]);
        int meanValueR = static_cast<int>(meanValue[2]);
        int deltaB = abs(meanValueB - 90);
        int deltaG = abs(meanValueG - 100);
        int deltaR = abs(meanValueR - 150);
        // if color correspond to beans
        if (deltaB < 25 && deltaB < 25 && deltaR < 25)
        {
            bitwise_or(finalBeanMask, componentMasks[i], finalBeanMask);
        }
    }
    element = getStructuringElement(MORPH_RECT, Size(7, 7));
    dilate(finalBeanMask, finalBeanMask, element);
    finalBeanMask = blobDelete(finalBeanMask, 20);
    Mat BBBBEAAANS;
    beanPlate.copyTo(BBBBEAAANS, finalBeanMask);

    return (finalBeanMask);
}

// POTATOES SEGMENTATION

Mat segPotatoes(Mat plateImg)
{
    Mat img;
    bilateralFilter(plateImg, img, 15, 100, 8);
    Mat imgHSV;
    cvtColor(img, imgHSV, COLOR_BGR2HSV);
    // Extract the Hue, Saturation, Value channels
    vector<Mat> ChannelsHSV;
    split(imgHSV, ChannelsHSV);
    Mat hueChannel = ChannelsHSV[0];
    Mat satChannel = ChannelsHSV[1];
    Mat valChannel = ChannelsHSV[2];
    int sizeBlobs = 15;
    Mat hueMask, satMask, valMask, food, mask;

    inRange(hueChannel, 18, 28, hueMask);
    inRange(satChannel, 43, 255, satMask);
    inRange(valChannel, 10, 255, valMask);
    Mat tempMask;
    bitwise_and(hueMask, satMask, tempMask);
    bitwise_and(valMask, tempMask, mask);
    sizeBlobs = 50;

    Mat element = getStructuringElement(MORPH_RECT, Size(7, 7));
    for (int i = 0; i < 20; i++)
    {
        dilate(mask, mask, element);
        erode(mask, mask, element);
    }
    floodFill(mask, Point(0, 0), Scalar(125));
    inRange(mask, 123, 125, mask);
    Mat invMask;
    bitwise_not(mask, invMask);

    Mat fmask = blobDelete(invMask, sizeBlobs);
    img.copyTo(food, fmask);
    // imshow("Only Main Dish",food);
    // waitKey(0);
    return fmask;
}

// MAIN COURSE SEGMENTATION

Mat segMain(Mat &plate, int &ID)
{
    Mat img;
    bilateralFilter(plate, img, 15, 100, 8);
    Mat imgHSV;
    cvtColor(img, imgHSV, COLOR_BGR2HSV);
    // Extract the Hue, Saturation, Value channels
    vector<Mat> ChannelsHSV;
    split(imgHSV, ChannelsHSV);
    Mat hueChannel = ChannelsHSV[0];
    Mat satChannel = ChannelsHSV[1];
    Mat valChannel = ChannelsHSV[2];
    int sizeBlobs = 15;
    Mat hueMask, satMask, valMask, food, mask;

    if (ID == 6)
    { // pork cutlet
        Mat tempMask;
        inRange(hueChannel, 9, 18, hueMask);
        inRange(satChannel, 45, 150, satMask);
        inRange(valChannel, 0, 205, valMask);
        bitwise_and(hueMask, satMask, tempMask);
        bitwise_and(valMask, tempMask, mask);
        sizeBlobs = 100;
    }
    if (ID == 7)
    { // fish cutlet
        Mat tempMask;
        inRange(hueChannel, 12, 17, hueMask);
        inRange(satChannel, 110, 206, satMask);
        inRange(valChannel, 140, 255, valMask);
        bitwise_and(hueMask, satMask, tempMask);
        bitwise_and(valMask, tempMask, mask);
        sizeBlobs = 100;
    }
    if (ID == 8)
    { // rabbit w potatoes
        Mat tempMask;
        inRange(hueChannel, 6, 34, hueMask);
        inRange(satChannel, 78, 255, satMask);
        inRange(valChannel, 46, 220, valMask);
        bitwise_and(hueMask, satMask, tempMask);
        bitwise_and(valMask, tempMask, mask);
        sizeBlobs = 100;
    }
    if (ID == 9)
    { // seafood salad
        Mat tempMask;
        inRange(hueChannel, 0, 24, hueMask);
        inRange(satChannel, 30, 255, satMask);
        inRange(valChannel, 80, 220, valMask);
        bitwise_and(hueMask, satMask, tempMask);
        bitwise_and(valMask, tempMask, mask);
        sizeBlobs = 50;
    }

    Mat element = getStructuringElement(MORPH_RECT, Size(7, 7));
    for (int i = 0; i < 20; i++)
    {
        dilate(mask, mask, element);
        erode(mask, mask, element);
    }
    floodFill(mask, Point(0, 0), Scalar(125));
    inRange(mask, 123, 125, mask);
    Mat invMask;
    bitwise_not(mask, invMask);

    Mat fmask = blobDelete(invMask, sizeBlobs);
    img.copyTo(food, fmask);
    // imshow("Only Main Dish",food);
    // waitKey(0);
    return fmask;
}

Mat segMainSecond(Mat PlateNOSides)
{
    Mat PlateNOSidesHSV;
    cvtColor(PlateNOSides, PlateNOSidesHSV, COLOR_BGR2HSV);
    vector<Mat> ChannelsHSV;
    split(PlateNOSidesHSV, ChannelsHSV);
    Mat satChannel = ChannelsHSV[1];
    Mat mask;
    inRange(satChannel, 150, 255, mask);
    Mat element = getStructuringElement(MORPH_RECT, Size(7, 7));
    for (int i = 0; i < 20; i++)
    {
        dilate(mask, mask, element);
        erode(mask, mask, element);
    }
    floodFill(mask, Point(0, 0), Scalar(125));
    inRange(mask, 123, 125, mask);
    Mat invMask;
    bitwise_not(mask, invMask);
    Mat fmask = blobDelete(invMask, 20);
    Mat cut;
    PlateNOSides.copyTo(cut, fmask);
    imshow("meat", cut);
    waitKey(0);
    return fmask;
}

Mat segSides(Mat &plateImg, int &ID, Mat trayImg)
{
    if (ID == 10)
    { // beans
        return segBeans(plateImg, trayImg);
    }
    return segPotatoes(plateImg);
}

Mat segSidesFirst(plate main, Mat &trayImg)
{
    food mainFood;
    vector<food> mainSides;
    vector<Mat> mainSidesMasks;

    for (int i = 0; i < main.foods.size(); i++)
    {
        if (main.foods[i].ID < 10)
        {
            mainFood = main.foods[i];
        }
        else
        { // it is a side so we obtain the masks for these sides so that we can subtract them later
            mainSides.push_back(main.foods[i]);
            mainSidesMasks.push_back(segSides(main.plateImage, main.foods[i].ID, trayImg));
        }
    }
    // now we have the mainFood and the masks for all sides
    Mat allSidesMask;
    if (mainSidesMasks.size() > 1)
    {
        bitwise_or(mainSidesMasks[0], mainSidesMasks[1], allSidesMask);
    }
    else if (mainSidesMasks.size() == 1)
    {
        allSidesMask = mainSidesMasks[0];
    }
    else
    {
        allSidesMask = Mat::zeros(trayImg.rows, trayImg.cols, CV_8U);
    }
    // Plate image without any of the dishes masks in it
    Mat plateImageNOSides, invertedAllSidesMask;
    bitwise_not(allSidesMask, invertedAllSidesMask);

    main.plateImage.copyTo(plateImageNOSides, invertedAllSidesMask);
    /* imshow("plateImageNOSides",plateImageNOSides);
    waitKey(0);
    imwrite("plateImageNOSides.png",plateImageNOSides);
    */
    // Segment all food that`s left in the plates with no sides
    Mat mainFoodMask = segMain(plateImageNOSides, mainFood.ID);
    // imshow("mainFoodMask",mainFoodMask);
    // waitKey(0);
    return mainFoodMask;
}

// FIRST COURSE SEGMENTATION
Mat segFirst(Mat &first)
{
    Mat firstHSV;
    cvtColor(first, firstHSV, COLOR_BGR2HSV);
    // Extract the Hue, Saturation, Value channels
    vector<Mat> ChannelsHSV;
    split(firstHSV, ChannelsHSV);
    Mat satChannel = ChannelsHSV[1];
    Mat mask;
    inRange(satChannel, 150, 255, mask);
    /* imshow("Dirty mask",mask);
    waitKey(0);  */

    Mat element = getStructuringElement(MORPH_RECT, Size(7, 7));
    for (int i = 0; i < 20; i++)
    {
        dilate(mask, mask, element);
        erode(mask, mask, element);
    }
    floodFill(mask, Point(0, 0), Scalar(125));
    inRange(mask, 123, 125, mask);
    Mat invMask;
    bitwise_not(mask, invMask);

    Mat fmask = blobDelete(invMask, 15);

    Mat cut;
    // first.copyTo(cut, fmask);
    /* imshow("Obtained mask",fmask);
    waitKey(0); */
    // imshow("Only First Dish",cut);
    // waitKey(0);
    return fmask;
}

// BREAD SEGMENTATION
Mat getBreadMask(const Mat &img)
{
    Mat imgHSV;
    cvtColor(img, imgHSV, COLOR_BGR2HSV);
    vector<Mat> ChannelsHSV;
    split(imgHSV, ChannelsHSV);
    Mat satChannel = ChannelsHSV[1];
    Mat valChannel = ChannelsHSV[2];
    Mat satMask, valMask, mask;

    inRange(valChannel, 9, 216, valMask);
    inRange(satChannel, 0, 63, satMask);
    bitwise_and(valMask, satMask, mask);
    Mat invMask;
    bitwise_not(mask, invMask);

    Mat element = getStructuringElement(MORPH_RECT, Size(3, 3));
    for (int i = 0; i < 1; i++)
    {
        dilate(invMask, invMask, element);
        erode(invMask, invMask, element);
    }
    bitwise_not(invMask, mask);
    return mask;
}

Mat growBread(Mat Mask, const Mat &img)
{

    vector<Mat> channels;
    split(img, channels);
    Mat gray = channels[0];
    Mat Points;
    findNonZero(Mask, Points);
    Rect bbox = boundingRect(Points);
    int x = bbox.x + bbox.width / 2;
    int y = bbox.y + bbox.height / 2;
    Point seed(x, y);

    Mat blurred, edgeMask, filt;
    blur(gray, blurred, Size(5, 5));
    Canny(blurred, edgeMask, 69, 100, 3);
    Mat breadMask = getBreadMask(img);
    Mat floodMask = Mat::zeros(edgeMask.rows + 2, edgeMask.cols + 2, edgeMask.type());
    copyMakeBorder(breadMask, floodMask, 1, 1, 1, 1, BORDER_CONSTANT, Scalar(0));
    Mat flodMask;
    int tresh = 9;
    Mat gray254, grayMask;
    inRange(gray, 0, 253, grayMask);
    bitwise_and(gray, grayMask, gray254);
    floodFill(gray254, floodMask, seed, Scalar(255),
              nullptr, Scalar(tresh), Scalar(tresh), 8 | (255 << 8));

    return gray254;
}

Mat segBread(Mat &trayImage)
{
    int thresholdB = 133;
    int thresholdG = 175;
    int thresholdR = 200;
    Mat filteredImage;
    bilateralFilter(trayImage, filteredImage, 15, 1000 * 0.1, 800 * 0.01);
    GaussianBlur(filteredImage, filteredImage, Size(9, 9), 3, 3);

    Mat mask;
    Scalar lower(thresholdB - 10, thresholdG - 10, thresholdR - 10);
    Scalar upper(thresholdB + 10, thresholdG + 10, thresholdR + 10);

    inRange(filteredImage, lower, upper, mask);

    Mat labels, stats, centroids;
    int largestArea = 0;
    int compLab = 0;
    int compNum = connectedComponentsWithStats(mask, labels, stats, centroids);
    for (int i = 0; i < compNum; i++)
    {
        int compSize = stats.at<int>(i, CC_STAT_AREA);
        if (compSize > largestArea && compSize < 1000 * 1000)
        {
            largestArea = compSize;
            compLab = i;
        }
    }
    Mat largestCompMask = (labels == compLab);

    Mat BbreadMask = growBread(largestCompMask, trayImage);
    Mat finalMask;
    inRange(BbreadMask, 254, 255, finalMask);
    return finalMask;
}

Mat BreadQuantize(Mat img)
{
    Mat biFilt;
    bilateralFilter(img, biFilt, 15, 100, 8);
    Mat features = biFilt.reshape(1, biFilt.rows * biFilt.cols);
    features.convertTo(features, CV_32F);
    vector<Vec3f> centers;
    centers.push_back(Vec3f(208.82082, 198.13742, 192.04411));
    centers.push_back(Vec3f(54.500652, 82.904533, 121.03458));
    centers.push_back(Vec3f(101.18235, 118.24827, 137.45587));
    centers.push_back(Vec3f(104.02261, 153.43279, 193.86192));
    centers.push_back(Vec3f(123.22646, 45.818489, 8.9983034));
    centers.push_back(Vec3f(149.03354, 143.06297, 141.08955));
    centers.push_back(Vec3f(28.609711, 44.660385, 78.54126));
    centers.push_back(Vec3f(164.8829, 73.303627, 5.9191713));
    centers.push_back(Vec3f(156.85526, 167.06149, 182.94035));
    centers.push_back(Vec3f(71.656349, 111.18534, 159.13615));
    centers.push_back(Vec3f(38.356865, 63.541718, 105.89803));
    centers.push_back(Vec3f(123.6679, 141.03932, 162.92538));
    centers.push_back(Vec3f(183.27263, 177.41438, 173.18538));
    centers.push_back(Vec3f(65.999352, 96.925385, 134.49191));
    centers.push_back(Vec3f(158.16553, 183.29399, 213.67415));
    centers.push_back(Vec3f(88.654297, 106.66511, 127.43037));
    centers.push_back(Vec3f(142.86543, 156.56624, 175.18062));
    centers.push_back(Vec3f(40.142361, 70.509705, 149.22398));
    centers.push_back(Vec3f(77.85479, 94.475746, 115.1222));
    centers.push_back(Vec3f(90.756279, 132.59978, 172.33955));
    centers.push_back(Vec3f(50.315228, 92.813583, 179.50658));
    centers.push_back(Vec3f(65.181053, 80.939491, 104.3078));
    centers.push_back(Vec3f(83.009201, 18.093664, 6.2365999));
    centers.push_back(Vec3f(118.77245, 128.33493, 139.68806));
    centers.push_back(Vec3f(188.29384, 116.09152, 16.27264));
    centers.push_back(Vec3f(197.13034, 187.50661, 181.54794));
    centers.push_back(Vec3f(50.582424, 65.525246, 89.527916));
    centers.push_back(Vec3f(164.85812, 159.35196, 156.76054));
    centers.push_back(Vec3f(127.78929, 168.21423, 209.29895));
    centers.push_back(Vec3f(193.83231, 163.68724, 144.79321));
    centers.push_back(Vec3f(40, 195, 187));  // greenish
    centers.push_back(Vec3f(10, 75, 91));    // greenish
    centers.push_back(Vec3f(10, 135, 135));  // greenish
    centers.push_back(Vec3f(100, 40, 20));   // blueish
    centers.push_back(Vec3f(80, 80, 255));   // reddish
    centers.push_back(Vec3f(220, 220, 220)); // whiteish
    centers.push_back(Vec3f(200, 200, 200)); // whiteish
    centers.push_back(Vec3f(150, 150, 150)); // grayish
    centers.push_back(Vec3f(80, 70, 70));    // grayish
    centers.push_back(Vec3f(75, 50, 30));    // dark brown
    centers.push_back(Vec3f(155, 175, 180)); // tan
    centers.push_back(Vec3f(160, 165, 180)); // tan
    centers.push_back(Vec3f(135, 156, 177)); // tan
    centers.push_back(Vec3f(45, 100, 190));  // orange
    centers.push_back(Vec3f(41, 58, 85));    // orange
    centers.push_back(Vec3f(87, 128, 160));
    centers.push_back(Vec3f(154, 163, 177));
    centers.push_back(Vec3f(40, 61, 88));
    centers.push_back(Vec3f(52, 83, 108));

    for (int i = 0; i < features.rows; i++)
    {
        int label = 0;
        float diff = 1000000000000000;
        for (int j = 0; j < centers.size(); j++)
        {
            Vec3f pix = features.at<Vec3f>(i);
            Vec3f cent = centers[j];
            // float tempDiff = (sqrt(pix[0]- cent[0]) +sqrt(pix[1]- cent[1]) +sqrt(pix[3]- cent[3]) );
            float tempDiff = norm(pix - cent, NORM_L2);
            if (tempDiff < diff)
            {
                label = j;
                diff = tempDiff;
            }
        }
        features.at<Vec3f>(i) = centers[label];
    }
    features = features.reshape(3, img.rows);
    Mat out;
    features.convertTo(out, CV_8UC3);
    vector<Vec3b> colors;
    colors.push_back(Vec3b(158, 183, 214));
    colors.push_back(Vec3b(128, 168, 209));
    colors.push_back(Vec3b(38, 64, 106));
    colors.push_back(Vec3b(104, 153, 194));
    colors.push_back(Vec3b(159, 111, 72));
    colors.push_back(Vec3b(91, 133, 172));
    colors.push_back(Vec3b(66, 97, 134));
    colors.push_back(Vec3b(55, 83, 121));
    colors.push_back(Vec3b(72, 111, 159));
    colors.push_back(Vec3b(41, 58, 85));

    // colors.push_back(Vec3b(29,45,79));
    // colors.push_back(Vec3b(124,141,163));
    // colors.push_back(Vec3b(51,66,90));
    // imshow("quantized breadbodx image",out);
    // waitKey(0);
    for (int y = 0; y < img.rows; y++)
    {
        for (int x = 0; x < img.cols; x++)
        {
            bool breadColor = false;
            for (int k = 0; k < colors.size(); k++)
            {
                if (out.at<Vec3b>(y, x) == colors[k])
                {
                    breadColor = true;
                }
            }
            if (breadColor == false)
            {
                out.at<Vec3b>(y, x) = Vec3b(0, 0, 0);
            }
        }
    }
    return out;
}

Mat breadCleanup(Mat quantizedBread)
{

    Mat quantizedHSV, grayBread, mask;
    cvtColor(quantizedBread, grayBread, COLOR_BGR2GRAY);
    inRange(grayBread, 2, 255, mask);
    int sizeBlobs = 30;
    Mat element = getStructuringElement(MORPH_RECT, Size(3, 3));
    erode(mask, mask, element);
    dilate(mask, mask, element);
    dilate(mask, mask, element);
    floodFill(mask, Point(0, 0), Scalar(125));
    floodFill(mask, Point(0, mask.rows - 1), Scalar(125));
    floodFill(mask, Point(0, mask.rows - 1), Scalar(125));
    floodFill(mask, Point(mask.cols - 1, mask.rows - 1), Scalar(125));
    inRange(mask, 123, 125, mask);
    Mat invMask;
    bitwise_not(mask, invMask);
    Mat fmask = blobDelete(invMask, sizeBlobs);
    return fmask;
}

map<int, vector<int>> resizeBBox(Mat mask, map<int, vector<int>> oldBBox, int ID)
{

    map<int, vector<int>> maskBBox = getBBox(mask, ID);
    map<int, vector<int>> newBBox;
    vector<int> newVal, maskBBoxVal;
    maskBBoxVal = maskBBox[ID];
    vector<int> oldBBoxVal = oldBBox[ID];
    newVal.push_back(maskBBoxVal[0]);
    newVal.push_back(maskBBoxVal[1]);
    newVal.push_back(maskBBoxVal[2]);
    newVal.push_back(maskBBoxVal[3]);
    newBBox.insert({ID, newVal});
    return newBBox;
}

// BOUNDING BOXES CONSTRUCTION

Mat squareBreadImage(food bread, const Mat &img)
{
    map<int, vector<int>> bbox = getBBox(bread.foodMask, bread.ID);
    vector<int> tempBBox = bbox[13];
    Mat breadBox = img(Rect(tempBBox[0], tempBBox[1], tempBBox[2], tempBBox[3]));
    return breadBox;
}

// create square mask for food recognition

Mat squareMask(Mat colorMask)
{
    Mat grayMask;
    if (colorMask.type() == 16)
    {
        cvtColor(colorMask, grayMask, COLOR_BGR2GRAY);
    }
    else
    {
        grayMask = colorMask;
    }
    map<int, vector<int>> GrayMaskBBox = getBBox(grayMask, 0);
    vector<int> tempBBox = GrayMaskBBox[0];
    int x = tempBBox[0];
    int y = tempBBox[1];
    int width = tempBBox[2];
    int height = tempBBox[3];

    Mat squaredMask = colorMask(Rect(x, y, width, height));
    return squaredMask;
}

// creates a mask with the ID as color
Mat colorID(Mat mask, int ID)
{
    Mat IDColor(mask.rows, mask.cols, CV_8U, Scalar(ID));
    Mat out;
    IDColor.copyTo(out, mask);
    return out;
}

vector<Mat> generateFoodMasks(tray inputTray)
{
    vector<Mat> predictedMasks;

    Mat firstCourseMaskID = colorID(inputTray.firstCourse.foods[0].foodMask, inputTray.firstCourse.foods[0].ID);
    predictedMasks.push_back(firstCourseMaskID);
    for (int i = 0; i < inputTray.mainCourse.foods.size(); i++)
    {
        Mat mainCourseMaskID = colorID(inputTray.mainCourse.foods[i].foodMask, inputTray.mainCourse.foods[i].ID);
        predictedMasks.push_back(mainCourseMaskID);
    }

    Mat breadMaskID = colorID(inputTray.bread.foodMask, inputTray.bread.ID);
    Mat saladMaskID = colorID(inputTray.salad.foodMask, inputTray.salad.ID);
    predictedMasks.push_back(breadMaskID);
    predictedMasks.push_back(saladMaskID);

    return predictedMasks;
}

// Generate the complete tray
tray generateTray(const Mat &image)
{
    // Create tray object "inTray" and initialize inTray.image as input image
    tray inTray(image);
    Mat boxedImage = image;
    Mat filt;
    bilateralFilter(inTray.trayImage, filt, 15, 100, 8);
    /*
    imshow("report-BilateralFilter.png", filt);
    waitKey(0);
    imwrite("report-BilateralFilter.png", filt);
    */
    vector<Mat> vecPlates = getPlateMask(inTray.trayImage, filt);
    //------------------

    // PASS THE VECTOR OF 2 PLATES TO DAVIDE'S SCRIPT THAT RECOGNIZE THEM AND RETURN ID'S
    for (int i = 0; i < 2; i++)
    {
        Mat squaredPlate = squareMask(vecPlates[i]);
        /*
        imshow("report-squarePlate.png", filt);
        waitKey(0);
        String num = std::to_string(i).c_str();
        imwrite("report-squarePlate"+num+".png", filt);
        */
        // plate tempPlate = recognizeFood(squaredPlate, i); /// DAVIDE
        plate tempPlate = processImages(squaredPlate);
        food tempFood = tempPlate.foods[0];

        // if it is main Course := tempFood.ID > 5
        if (tempFood.ID > 5)
        {
            inTray.mainCourse = tempPlate;
            inTray.mainCourse.plateImage = vecPlates[i];
            inTray.mainCourse.isEmpty = false;
        }
        // plate is empty
        else if (tempFood.ID == 0)
        {
            if (inTray.firstCourse.isEmpty == false)
            {
                inTray.mainCourse.isEmpty = true;
                Mat empty;
                inTray.mainCourse.plateMask = empty;
            }
            else if (inTray.mainCourse.isEmpty == false)
            {
                inTray.firstCourse.isEmpty = true;
                Mat empty;
                inTray.firstCourse.plateMask = empty;
            }
            else
            {
                inTray.mainCourse.isEmpty = true;
                inTray.firstCourse.isEmpty = true;
                Mat empty;
                inTray.firstCourse.plateMask = empty;
                inTray.mainCourse.plateMask = empty;
            }
        }
        else
        { // it is first Course
            inTray.firstCourse = tempPlate;
            inTray.firstCourse.plateImage = vecPlates[i];
            inTray.firstCourse.isEmpty = false;
        }
    }

    //--------Segment first and main course's foods-------
    for (int i = 0; i < inTray.firstCourse.foods.size(); i++)
    {

        Mat foodCut = segFirst(inTray.firstCourse.plateImage);

        inTray.firstCourse.foods[i].foodMask = foodCut;
        inTray.firstCourse.foods[i].bbox = getBBox(foodCut, inTray.firstCourse.foods[i].ID);

        // printBBox(boxedImage, inTray.firstCourse.foods[i].bbox, inTray.firstCourse.foods[i].ID);
    }

    for (int i = 0; i < inTray.mainCourse.foods.size(); i++)
    {
        Mat foodCut;
        if (inTray.mainCourse.foods[i].ID < 10)
        { // meat
            foodCut = segSidesFirst(inTray.mainCourse, inTray.trayImage);
        }
        else // either basil potatoes or beans
        {
            foodCut = segSides(inTray.mainCourse.plateImage, inTray.mainCourse.foods[i].ID, inTray.trayImage);
        }

        inTray.mainCourse.foods[i].foodMask = foodCut;
        inTray.mainCourse.foods[i].bbox = getBBox(foodCut, inTray.mainCourse.foods[i].ID);

        // printBBox(boxedImage, inTray.mainCourse.foods[i].bbox, inTray.mainCourse.foods[i].ID);
    }

    //--------------------finds bread--------------------
    food Bread(13);
    // finds preliminary mask
    Bread.foodMask = segBread(inTray.trayImage);

    // create rectangular image for recognition
    Mat squaredBread = squareBreadImage(Bread, inTray.trayImage);
    // recognize if there is bread
    // bool breadPresence = isThereBread(squaredBread);
    bool breadPresence = is_this_bread(squaredBread);

    if (breadPresence == true)
    {
        Bread.bbox = getBBox(Bread.foodMask, Bread.ID);
        Bread.foodMask = breadCleanup(BreadQuantize(squaredBread));
        // fixing the small mask to one of size as the input image
        int tl_x = Bread.bbox[Bread.ID][0]; // X-coordinate of the top-left point
        int tl_y = Bread.bbox[Bread.ID][1]; // Y-coordinate of the top-left point
        Mat fixedMask(inTray.trayImage.rows, inTray.trayImage.cols, CV_8U);
        Rect roi_rect(tl_x, tl_y, Bread.foodMask.cols, Bread.foodMask.rows);
        Mat roi = fixedMask(roi_rect);
        Bread.foodMask.copyTo(roi);
        Bread.foodMask = fixedMask;

        Bread.bbox = resizeBBox(Bread.foodMask, Bread.bbox, Bread.ID);
        inTray.bread.bbox = Bread.bbox;
        inTray.hasBread = true;
        inTray.bread = Bread;
        // printBBox(boxedImage, inTray.bread.bbox, inTray.bread.ID);
        // cout << inTray.bread.bbox[13][0] << ", " << inTray.bread.bbox[13][1] << ", " << inTray.bread.bbox[13][2]<< ", " << inTray.bread.bbox[13][3];
    }
    else
    {
        inTray.hasBread = false;
        Mat empty;
        inTray.bread.foodMask = empty;
    }

    //--------------------finds salad--------------------
    food salad(12);
    // finds salad bowl
    Mat saladMask = getSaladMask(inTray.trayImage);
    // create rectangular image for recognition
    Mat squaredSalad = squareMask(saladMask);
    // recognize if there is salad
    // bool saladPresence = isThereSalad(squaredSalad);
    bool saladPresence = is_this_salad(squaredSalad);
    if (saladPresence == true)
    {
        salad.foodMask = cutSalad(saladMask);
        inTray.hasSalad = true;
        inTray.salad = salad;
        inTray.salad.bbox = getBBox(inTray.salad.foodMask, inTray.salad.ID);
        // printBBox(boxedImage, inTray.salad.bbox, inTray.salad.ID);
    }
    else
    {
        inTray.hasSalad = false;
        Mat empty;
        inTray.salad.foodMask = empty;
    } /*
     namedWindow("Tray with final bounding boxes");
     imshow("Tray with final bounding boxes", boxedImage);
     waitKey(0); */
    // imwrite("Tray4_BBox.png", boxedImage);

    return inTray;
}

// Generate the complete tray
tray generateLeftoverTray(const Mat &image, tray initialTray)
{
    // Create tray object "inTray" and initialize inTray.image as input image
    tray inTray = initialTray;
    // make the new tray as the old so to have all ID's
    inTray.trayImage = image;

    Mat boxedImage = image;
    Mat filt;
    bilateralFilter(inTray.trayImage, filt, 15, 100, 8);
    vector<Mat> vecPlates = getPlateMask(inTray.trayImage, filt);
    //------------------
    vector<Mat> squaredPlates;
    if (vecPlates.size() < 1)
    {
        cout << "no plates found" << endl;
        tray empty;
        return empty;
    }
    else if ((vecPlates.size() == 1))
    {
        squaredPlates.push_back(squareMask(vecPlates[0]));
        
    }
    else
    {
        for (int i = 0; i < 2; i++)
        {
            squaredPlates.push_back(squareMask(vecPlates[i]));
            
        }
    }

    // this id is the corresponding first place in the plate vector
    int ind = process_leftover(squaredPlates, initialTray.firstCourse); /// DAVIDE
    if (ind > 1)
    {
        cout << "wrong food detected" << endl;
        tray empty;
        return empty;
    }

    bool singlePlatefound = false;
    if (vecPlates.size() > 1)
    { // two plates found
        if (ind == 0)
        { // pasta is first index
            inTray.firstCourse.plateImage = vecPlates[0];
            inTray.mainCourse.plateImage = vecPlates[1];
        }
        else if (ind == 1)
        { //(ind == 1)
            inTray.firstCourse.plateImage = vecPlates[1];
            inTray.mainCourse.plateImage = vecPlates[0];
        }
        else
        {
            cout << "wrong food detected" << endl;
            tray empty;
            return empty;
        }
    }
    else if (vecPlates.size() == 1)
    { // only one plate found
        singlePlatefound = true;
        if (ind == -1)
        { // no first plate
            // set found plate on main course
            inTray.mainCourse.plateImage = vecPlates[0];
            // reset foods and plate image for first course
            Mat black;
            inTray.firstCourse.plateImage = black;
            food emptyFood(-1);
            vector<food> emptyFoods;
            emptyFoods.push_back(emptyFood);
            inTray.firstCourse.foods = emptyFoods;
        }

        else
        { //(ind == 0) is pasta
            // set found plate to first course image
            inTray.firstCourse.plateImage = vecPlates[0];
            // reset foods and plate image for main course
            food emptyFood(-1);
            vector<food> emptyFoods;
            Mat black;
            inTray.mainCourse.plateImage = black;
            emptyFoods.push_back(emptyFood);
            inTray.mainCourse.foods = emptyFoods;
        }
    }
    else
    {
        cout << "wrong food detected" << endl;
        tray empty;
        return empty;
    }

    // RESETS ALL MASKS AND BBOXES
    // Maybe this?
    /*
    for(int j = 0; j< inTray.firstCourse.foods.size(); j++){
        Mat empty;
        inTray.firstCourse.foods[j].foodMask = empty;
        map <int, vector<int>> emptyMap;
        inTray.firstCourse.foods[j].bbox = emptyMap;
    }
    for(int j = 0; j< inTray.mainCourse.foods.size(); j++){
        Mat empty;
        inTray.mainCourse.foods[j].foodMask = empty;
        map <int, vector<int>> emptyMap;
        inTray.mainCourse.foods[j].bbox = emptyMap;
    }
    */

    //--------Segment first and main course's foods-------
    if (singlePlatefound == false || ind == 0)
    {
        for (int i = 0; i < inTray.firstCourse.foods.size(); i++)
        {

            Mat foodCut = segFirst(inTray.firstCourse.plateImage);

            inTray.firstCourse.foods[i].foodMask = foodCut;
            inTray.firstCourse.foods[i].bbox = getBBox(foodCut, inTray.firstCourse.foods[i].ID);

            printBBox(boxedImage, inTray.firstCourse.foods[i].bbox, inTray.firstCourse.foods[i].ID);
        }
    }

    if (singlePlatefound == false || ind == -1)
    {
        for (int i = 0; i < inTray.mainCourse.foods.size(); i++)
        {
            Mat foodCut;
            if (inTray.mainCourse.foods[i].ID < 10)
            { // meat
                foodCut = segSidesFirst(inTray.mainCourse, inTray.trayImage);
            }
            else // either basil potatoes or beans
            {
                foodCut = segSides(inTray.mainCourse.plateImage, inTray.mainCourse.foods[i].ID, inTray.trayImage);
            }

            inTray.mainCourse.foods[i].foodMask = foodCut;
            inTray.mainCourse.foods[i].bbox = getBBox(foodCut, inTray.mainCourse.foods[i].ID);

            printBBox(boxedImage, inTray.mainCourse.foods[i].bbox, inTray.mainCourse.foods[i].ID);
        }
    }

    //--------------------finds bread--------------------
    food Bread(13);
    // finds preliminary mask
    Bread.foodMask = segBread(inTray.trayImage);

    // create rectangular image for recognition
    Mat squaredBread = squareBreadImage(Bread, inTray.trayImage);
    // recognize if there is bread
    bool breadPresence = isThereBread(squaredBread);
    if (breadPresence == true)
    {
        Bread.bbox = getBBox(Bread.foodMask, Bread.ID);
        Bread.foodMask = breadCleanup(BreadQuantize(squaredBread));
        // fixing the small mask to one of size as the input image
        int tl_x = Bread.bbox[Bread.ID][0]; // X-coordinate of the top-left point
        int tl_y = Bread.bbox[Bread.ID][1]; // Y-coordinate of the top-left point
        Mat fixedMask(inTray.trayImage.rows, inTray.trayImage.cols, CV_8U);
        Rect roi_rect(tl_x, tl_y, Bread.foodMask.cols, Bread.foodMask.rows);
        Mat roi = fixedMask(roi_rect);
        Bread.foodMask.copyTo(roi);
        Bread.foodMask = fixedMask;

        Bread.bbox = resizeBBox(Bread.foodMask, Bread.bbox, Bread.ID);
        inTray.bread.bbox = Bread.bbox;
        inTray.hasBread = true;
        inTray.bread = Bread;
        printBBox(boxedImage, inTray.bread.bbox, inTray.bread.ID);
        // cout << inTray.bread.bbox[13][0] << ", " << inTray.bread.bbox[13][1] << ", " << inTray.bread.bbox[13][2]<< ", " << inTray.bread.bbox[13][3];
    }
    else
    {
        inTray.hasBread = false;
        Mat empty;
        inTray.bread.foodMask = empty;
    }

    //--------------------finds salad--------------------
    food salad(12);
    // finds salad bowl
    Mat saladMask = getSaladMask(inTray.trayImage);
    // create rectangular image for recognition
    Mat squaredSalad = squareMask(saladMask);
    // recognize if there is salad
    bool saladPresence = isThereSalad(squaredSalad);
    if (saladPresence == true)
    {
        salad.foodMask = cutSalad(saladMask);
        inTray.hasSalad = true;
        inTray.salad = salad;
        inTray.salad.bbox = getBBox(inTray.salad.foodMask, inTray.salad.ID);
        printBBox(boxedImage, inTray.salad.bbox, inTray.salad.ID);
    }
    else
    {
        inTray.hasSalad = false;
        Mat empty;
        inTray.salad.foodMask = empty;
    } /*
     namedWindow("Tray with final bounding boxes");
     imshow("Tray with final bounding boxes", boxedImage);
     waitKey(0);
     imwrite("Tray4_BBox.png", boxedImage); */

    return inTray;
}

Mat mergeMasks(vector<Mat> Masks)
{
    Mat mergedMask = Mat::zeros(Masks[0].rows, Masks[0].cols, CV_8U);
    for (int k = 0; k < Masks.size(); k++)
    {
        Mat tempMask = Masks[k];
        for (int i = 0; i < tempMask.rows; i++)
        {
            for (int j = 0; j < tempMask.cols; j++)
            {
                int pix = tempMask.at<uchar>(i, j);
                if (pix != 0)
                {
                    mergedMask.at<uchar>(i, j) = pix;
                }
            }
        }
    }
    return mergedMask;
}
