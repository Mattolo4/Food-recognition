#ifndef DETECTOR_LIB_H
#define DETECTOR_LIB_H

#include <opencv2/opencv.hpp>
#include <vector>

int map_label(char letter);
std::pair<std::vector<int>, std::vector<cv::Mat>> load_images_from_folder(const std::string& folderPath);
std::pair<std::vector<int>, std::vector<int>> sift_object_detection(const cv::Mat& image1, const std::string& folder_database, float dist, int p);
bool containsValue(const std::vector<int>& search_l, int value);
std::pair<std::vector<int>, std::vector<int>> sift_object_detection_label_selected(const cv::Mat& image1, const std::vector<int>& search_l, const std::string& folder_database, float dist, int p);
std::pair<int, int> sift_object_detection_reference_img(const cv::Mat& image1, const std::vector<cv::Mat>& images, float dist, int p);
std::pair<int, float> search_for_salad_bread(const cv::Mat& img, const std::vector<cv::Mat>& images, int p);
std::pair<std::vector<int>, std::vector<float>> search_for(const cv::Mat& img, const std::vector<int>& labels, int k, int p);

#endif  // DETECTOR_LIB_H