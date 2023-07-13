#ifndef UTILITIES
#define UTILITIES

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/core/utils/filesystem.hpp>
#include <opencv2/calib3d.hpp>

std::vector<cv::Mat> getPlateMask(const cv::Mat& image);
cv::Mat printHist(std::vector<cv::Mat> hist);
std::vector<cv::Mat> getHist(cv::Mat hist);
cv::Mat getHistSingle(cv::Mat img);
cv::Mat GRemove(cv::Mat &image, int delta);
	

#endif // UTILITIES