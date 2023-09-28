#ifndef PLATE_REC_H
#define PLATE_REC_H
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <vector>
#include "Detector_lib.hpp"
#include <algorithm>
#include "Classes.h"


std::vector<cv::Mat> load_images_from_folder_l(const std::string& folderPath);
cv::Scalar average_color_5x5(const cv::Mat& img, int x, int y);
cv::Scalar average_color_35x35(const cv::Mat& img, int x, int y);
std::vector<cv::Scalar> get_colors(const cv::Mat& img, const std::vector<cv::Point>& points);
cv::Scalar get_center_color(const cv::Mat& img);
plate processImages(const cv::Mat& img);
bool is_this_salad(const cv::Mat& img);
bool is_this_bread(const cv::Mat& img);
int process_leftover(const std::vector<cv::Mat>& images,const plate& plate_l);

#endif 