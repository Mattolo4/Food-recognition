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
Mat element = getStructuringElement(MORPH_RECT, Size(3, 3));
Mat HSVimg;
Mat colQuantize(Mat img);
Point getCenter(const Mat& mask);
Mat myErode(Mat& mask);
int myLabel(Mat featureRow, vector<Mat> centroids);
Mat enContrast(Mat& img, int& contrast, int& brightness);
vector<Mat> getClusters(Mat& img, vector<Mat> centroids);
Mat createClusters(Mat img, Mat labels, Mat features, int k);

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
    int contrast = 64;
    int brightness = 32;
    Mat contrasted =  enContrast(img, contrast, brightness);
    /* namedWindow("contrasted", WINDOW_NORMAL);
    imshow("contrasted", contrasted);
    waitKey(0); */
    
    //strongly blur the image
    Size kernel = Size(99, 99);
    GaussianBlur(contrasted,contrasted, kernel, 0);
    /* namedWindow("Blurred", WINDOW_NORMAL);
    imshow("Blurred", img);
    waitKey(0); */
    
    Mat quantImg = colQuantize(contrasted);
     namedWindow("quantized", WINDOW_NORMAL);
    imshow("quantized", quantImg);
    waitKey(0); 

    cvtColor(quantImg, HSVimg, COLOR_BGR2HSV);

    //Extract the Hue channel
    vector<Mat> hsvChannels;
    split(HSVimg, hsvChannels);
    //hsvChannels[1] = Scalar(200);

    Mat hueChannel = hsvChannels[0];
    //cout << "hueChannel type : " << hueChannel.depth() << endl;
    int histSize = 180;
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
    vector<Mat> masks;
    vector<int> Bins;
    for (int i = 0; i < 10; i++) {
        int tempBin = histogramData[i].first*180/histSize;
        if(histogramData[i].second > 0){
            newHist.push_back(histogramData[i]);
            cout << "Bin " << tempBin << ": " << histogramData[i].second << endl;
            Scalar hsv_lower_l(tempBin, 0, 0);
            Scalar hsv_lower_h(tempBin, 255, 255);
            Mat tempMask;
            inRange(HSVimg, hsv_lower_l, hsv_lower_h, tempMask);
            masks.push_back(tempMask);
            /* namedWindow("Mask", WINDOW_NORMAL);
            imshow("Mask", tempMask);
            waitKey(0); */
        }
    } 
    Mat newMask = Mat::zeros(img.size(), CV_8UC1);
    vector<Mat> centroids; // has the centroids coordinates and color [B,G,R,x,y]
    Mat tempCentroid = Mat::zeros(1,5, CV_32F);

    // Set the white pixel at the specified location
    for(int i = 0; i< masks.size() ; i++){
        Point tempPos = getCenter(masks[i]);
        //cout << "position of centroid: " << tempPos << endl;
        Vec3b tempColor = contrasted.at<Vec3b>(tempPos.y,tempPos.x);
        //cout << "color of centroid: " << tempColor << endl;
        tempCentroid.at<float>(0,0) = static_cast<float>(tempColor[0]);
        tempCentroid.at<float>(0,1) = static_cast<float>(tempColor[1]);
        tempCentroid.at<float>(0,2) = static_cast<float>(tempColor[2]);
        tempCentroid.at<float>(0,3) = static_cast<float>(tempPos.x);
        tempCentroid.at<float>(0,4) = static_cast<float>(tempPos.y);
        centroids.push_back(tempCentroid);
        newMask.at<uchar>(tempPos) = 255;
    }
    /* namedWindow("Mask points", WINDOW_NORMAL);
    imshow("Mask points", newMask);
    waitKey(0);  */

    vector<Mat> clusters = getClusters(img, centroids);
}

Mat colQuantize(Mat img){
    int K = 4;
    Mat samples(img.rows * img.cols, img.channels(), CV_32F);
	for (int y = 0; y < img.rows; y++)
		for (int x = 0; x < img.cols; x++)
			for (int z = 0; z < img.channels(); z++)
				if (img.channels() == 3) {
					samples.at<float>(y + x * img.rows, z) = img.at<Vec3b>(y, x)[z];
				}
				else {
					samples.at<float>(y + x * img.rows, z) = img.at<uchar>(y, x);
				}

	Mat labels;
	int attempts = 5;
	Mat centers;
    TermCriteria criteria(TermCriteria::EPS + TermCriteria::COUNT, 10, 1.0);
	kmeans(samples, K, labels, criteria, attempts, KMEANS_PP_CENTERS, centers);


	Mat new_image(img.size(), img.type());
	for (int y = 0; y < img.rows; y++)
		for (int x = 0; x < img.cols; x++)
		{
			int cluster_idx = labels.at<int>(y + x * img.rows, 0);
			if (img.channels()==3) {
				for (int i = 0; i < img.channels(); i++) {
					new_image.at<Vec3b>(y, x)[i] = centers.at<float>(cluster_idx, i);
				}
			}
			else {
				new_image.at<uchar>(y, x) = centers.at<float>(cluster_idx, 0);
			}
		}
	//imshow("clustered image", new_image);
	return new_image;
}

Point getCenter(const Mat& mask){
    Mat element = getStructuringElement(MORPH_RECT, Size(3, 3));

    // Erode the mask
    Mat labels, stats, centroids;
    int numComponents = connectedComponentsWithStats(mask, labels, stats, centroids);

    // Find the index of the largest component (excluding the background)
    int largestComponentIndex = 0;
    int largestComponentSize = 0;
    for (int label = 1; label < numComponents; ++label) {
        int componentSize = stats.at<int>(label, CC_STAT_AREA);
        if (componentSize > largestComponentSize) {
            largestComponentSize = componentSize;
            largestComponentIndex = label;
        }
    }

    // Create a binary mask for the largest component
    Mat largestComponentMask = (labels == largestComponentIndex);
    /* namedWindow("largestComponentMask", WINDOW_NORMAL);
    imshow("largestComponentMask", largestComponentMask);
    waitKey(0); */
    // Erode the largest component mask
    Mat eroded = myErode(largestComponentMask);
    int whiPix = countNonZero(eroded);
    //cout << "white points temp: " << whiPix << endl;
    // Find the coordinates of the single white point
    Point point(-1, -1);
    Mat nonzeroLocs;
    findNonZero(eroded, nonzeroLocs);
    if (!nonzeroLocs.empty()) {
        point = nonzeroLocs.at<Point>(0);
    }
    cout << "point coords: " << point << endl;
    return point;
}

Mat myErode(Mat& mask){
    Mat erodedMask, tempMask;
    tempMask = mask;
    erode(mask, erodedMask, element);
    int numWhitePixels = countNonZero(erodedMask);
    //cout << "white points: " << numWhitePixels << endl;
    if(numWhitePixels != 0){
        tempMask = myErode(erodedMask);
    }
    return tempMask;
}

Mat enContrast(Mat& img, int& contrast, int& brightness){
    Mat out;
    
    if (brightness != 0)
    {
        double shadow, highlight;
        if (brightness > 0)
        {
            shadow = brightness;
            highlight = 255;
        }
        else
        {
            shadow = 0;
            highlight = 255 + brightness;
        }
        double alpha_b = (highlight - shadow) / 255.0;
        double gamma_b = shadow;

        addWeighted(img, alpha_b, img, 0, gamma_b, out);
    }
    else
    {
       out = img.clone();
    }
    if (contrast != 0)
    {
        double f = 131 * (contrast + 127) / (127.0 * (131 - contrast));
        double alpha_c = f;
        double gamma_c = 127 * (1 - f);

        addWeighted(out, alpha_c, out, 0, gamma_c, out);
    }

    return out;
} 

vector<Mat> getClusters(Mat& img, vector<Mat> centroids){
    Mat features(img.rows * img.cols, img.channels()+2, CV_32F); //feature vector [B,G,R,x,y] 
    int index = 0;
    float initialLabel;
    Mat tempRow = Mat::zeros(1,5, CV_32F);
    Mat initialLabels(img.rows * img.cols, 1, CV_32F);
    for(int y=0; y<img.rows ; y++){
        for(int x=0; x<  img.cols ; x++){
            Vec3b pix = img.at<Vec3b>(y,x);
            if(pix[0]!=0 && pix[1]!=0 && pix[2]!=0){
                features.at<float>(index,0) = static_cast<float>(pix[0]);
                features.at<float>(index,1) = static_cast<float>(pix[1]);
                features.at<float>(index,2) = static_cast<float>(pix[2]);
                features.at<float>(index,3) = static_cast<float>(x);
                features.at<float>(index,4) = static_cast<float>(y);
                initialLabel = myLabel(features.row(index), centroids);
                initialLabels.at<float>(index) = initialLabel;
                index++;
            }
        }
    }
    //hconcat(featuresCol,featuresPos,features);
    //int id = (img.cols)*centroid.y  + centroid.x; 
    //cout << "Centroid x: " << centroid.x << " centroid y : " <<centroid.y << endl;
    //cout << "centroid row of feature Mat: " << features.row(id) << endl;
    //cout << "dimensions of features Mat: " << features.size() << endl;
    /*cout << "dimensions of input Image: " << img.size() << endl;*/
    //cout << "index: " << index << endl; 

    features = features.rowRange(0, index); // deletes all rows of black pixels
    cout << "dimensions of features Mat: " << features.size() << endl;
    initialLabels = initialLabels.rowRange(0, index); // deletes all rows of black pixels
    cout << "dimensions of initialLabels Mat: " << initialLabels.size() << endl;

    int k = centroids.size();

    Mat centers;
    Mat labels = initialLabels;
    int attempts = 10;
    Range rowRange(700, 720);
    //cout << "Initial labels: \n" << labels.rowRange(rowRange) << endl;

    TermCriteria criteria(TermCriteria::EPS + TermCriteria::COUNT, 10, 1.0);
	double compactness = kmeans(features, k, labels, criteria, attempts, KMEANS_USE_INITIAL_LABELS, centers);

    //cout << "Final labels: \n" << labels.rowRange(rowRange) << endl;
    Mat tempCluster = createClusters(img, labels, features, k);
    namedWindow("Final Clusters", WINDOW_NORMAL);
    imshow("Final Clusters", tempCluster);
    waitKey(0);
    vector<Mat> clusters;
    return clusters;
}

int myLabel(Mat featureRow, vector<Mat> centroids){
    float tempVal = 0;
    float minVal = numeric_limits<float>::infinity();
    float minInd;
    for(int i = 0; i < centroids.size(); i++){
        Mat centroid = centroids[i];
        for(int j = 0; j < featureRow.cols; j++){
            float diff = featureRow.at<float>(j)- centroids[i].at<float>(j);
            tempVal = tempVal + diff*diff;//least square value with label i
        }
        if(tempVal < minVal){ 
            minVal = tempVal;
            minInd = static_cast<float>(i);
        }
    }
    return minInd;
}

Mat createClusters(Mat img, Mat labels, Mat features, int k){
    Mat outClusters = Mat::zeros(img.rows, img.cols, CV_8U);
    int x,y,labelVal,color;
    for(int i = 0; i < features.rows ; i++){
        x  = static_cast<int>(features.at<float>(i,3));
        y  = static_cast<int>(features.at<float>(i,4));
        labelVal = static_cast<int>(labels.at<float>(0,i))+1;
        color = int(255/k*labelVal);
        cout << "color " << color << endl;
        outClusters.at<float>(y,x) = color;
    }
    return outClusters;
}