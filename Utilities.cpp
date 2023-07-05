#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/core/utils/filesystem.hpp>
#include <opencv2/calib3d.hpp>
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

Mat printHist(vector<Mat> hist){
    int histWidth = 512;
    int histHeight = 255;
    int binWidth = cvRound((double)histWidth / 256);
    Mat histImage(histHeight, histWidth, CV_8UC3, Scalar(0, 0, 0));
    vector<Scalar> color;
    color.push_back(Scalar(255, 0, 0));
    color.push_back(Scalar(0, 255, 0));
    color.push_back(Scalar(0, 0, 255));
    for(int j = 0; j < hist.size(); j++){
        normalize(hist[j], hist[j], 0, histImage.rows, NORM_MINMAX, -1, Mat());
        for (int i = 1; i < 256; i++) {
            line(histImage, Point(binWidth * (i - 1), histHeight - cvRound(hist[j].at<float>(i - 1))), Point(binWidth * (i), histHeight - cvRound(hist[j].at<float>(i))), color[j], 2, LINE_AA);
        }
    }
    return histImage; 
}

vector<Mat> getHist(Mat img){
    vector<Mat> channels;
    split(img, channels);
    int histSize = 256;
    float range[] = {1, 256}; 
    const float* histRange = {range};
    bool uniform = true; 
    bool accumulate = false;
    vector<Mat> hist;
    for(int j = 0; j < channels.size(); j++){
        Mat tempHist;
        calcHist(&channels[j], 1, 0, Mat(), tempHist, 1, &histSize, &histRange, uniform, accumulate);
        hist.push_back(tempHist);
    }
    return hist; 
}