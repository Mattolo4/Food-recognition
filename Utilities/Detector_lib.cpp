#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <filesystem>
#include <map>
#include <vector>


int map_label(char letter) {
    if (letter == 'p') // pasta
        return 0;
    else if (letter == 'i') // insalata
        return 1;
    else if (letter == 'f') // fagioli
        return 2;
    else if (letter == 'o') // cotoletta
        return 3;
    else if (letter == 'r') // riso
        return 4;
    else if (letter == 'a') // patate
        return 5;
    else if (letter == 'l') // pollo
        return 6;
    else if (letter == 'v') // polpo
        return 7;
    else if (letter == 'm') // lonza
        return 8;
    else if (letter == 'c') // cozze
        return 9;
    else
        return -1;
}
//this function provide the database of reference images and labels
// std::pair<std::vector<int>,std::vector<cv::Mat>> load_images_from_folder(const std::string& folderPath) {
//     std::vector<cv::Mat> images;
//     std::vector<int> Labels;
//     WIN32_FIND_DATA fd;
//     HANDLE hFind = FindFirstFile((folderPath + "/*.*").c_str(), &fd);
//     if (hFind != INVALID_HANDLE_VALUE) {
//         do {
//             if (!(fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)) {
//                 std::string fileName = fd.cFileName;
//                 if (fileName != "." && fileName != "..") {
//                     std::string filePath = folderPath + "/" + fd.cFileName;
//                     cv::Mat image = cv::imread(filePath);
//                     int label = map_label(fd.cFileName[0]);
                    
//                     if (!image.empty() && label!=-1) { //if the label is valid
//                         images.push_back(image);
//                         Labels.push_back(label);
//                     }
//                 }
//             }
//         } while (FindNextFile(hFind, &fd)); //for each file found in the folder
//         FindClose(hFind);
//     }
//     return {Labels,images};
// }
std::pair<std::vector<int>, std::vector<cv::Mat>> load_images_from_folder(const std::string& folderPath) {
    std::vector<cv::Mat> images;
    std::vector<int> Labels;

    for (const auto& entry : std::filesystem::directory_iterator(folderPath)) {
        if (entry.is_regular_file()) {
            std::string fileName = entry.path().filename().string();
            if (fileName != "." && fileName != "..") {
                std::string filePath = entry.path().string();
                cv::Mat image = cv::imread(filePath);
                int label = map_label(fileName[0]);
                
                if (!image.empty() && label != -1) { //if the label is valid
                    images.push_back(image);
                    Labels.push_back(label);
                }
            }
        }
    }

    return {Labels, images};
}


std::pair<std::vector<int>, std::vector<int>> sift_object_detection(const cv::Mat& image1, const std::string& folder_database, float dist, int p) {
    
    //load reference images
    std::pair<std::vector<int>,std::vector<cv::Mat>> images = load_images_from_folder(folder_database);
    //create sift object
    cv::Ptr<cv::SIFT> sift = cv::SIFT::create();
    //compute descriptors
    std::vector<cv::KeyPoint> kp1;
    cv::Mat des1;
    sift->detectAndCompute(image1, cv::noArray(), kp1, des1);

    std::vector<int> found(10, 0);
    std::vector<int> gm(10, 0);

    for (size_t i = 0; i < images.first.size(); ++i) {
        int label = images.first[i];
        cv::Mat img = images.second[i];
        std::vector<cv::KeyPoint> kp2;
        cv::Mat des2;
        //compute descriptors for reference image img
        sift->detectAndCompute(img, cv::noArray(), kp2, des2);

        cv::BFMatcher bf;
        std::vector<std::vector<cv::DMatch>> matches;
        bf.knnMatch(des1, des2, matches, 2);
        //check if there are good matches based on the provided distance
        std::vector<cv::DMatch> good_matches;
        for (size_t i = 0; i < matches.size(); ++i) {
            if (matches[i][0].distance < dist * matches[i][1].distance) {
                good_matches.push_back(matches[i][0]);
            }
        }
        //checkign if this is valid by the number of good matches found
        if (good_matches.size() > p - 1) {
            found[label]++;
            gm[label] = gm[label] + static_cast<int>(good_matches.size());
        }
    }
    

    return std::make_pair(found, gm);
}

bool containsValue(const std::vector<int>& search_l, int value) {
    return std::find(search_l.begin(), search_l.end(), value) != search_l.end();
}

//same function as before but with the possibility of searching for a specific label
std::pair<std::vector<int>, std::vector<int>> sift_object_detection_label_selected(const cv::Mat& image1, const std::vector<int>& search_l, const std::string& folder_database, float dist, int p) {
    std::pair<std::vector<int>,std::vector<cv::Mat>> images = load_images_from_folder(folder_database);
    cv::Ptr<cv::SIFT> sift = cv::SIFT::create();

    std::vector<cv::KeyPoint> kp1;
    cv::Mat des1;
    sift->detectAndCompute(image1, cv::noArray(), kp1, des1);

    std::vector<int> found(10, 0);
    std::vector<int> gm(10, 0);

    for (size_t i = 0; i < images.first.size(); ++i) {
        int label = images.first[i];
        cv::Mat img = images.second[i];
        if (containsValue(search_l, label)) {
            std::vector<cv::KeyPoint> kp2;
            cv::Mat des2;
            sift->detectAndCompute(img, cv::noArray(), kp2, des2);

            cv::BFMatcher bf;
            std::vector<std::vector<cv::DMatch>> matches;
            bf.knnMatch(des1, des2, matches, 2);

            std::vector<cv::DMatch> good_matches;
            for (size_t i = 0; i < matches.size(); ++i) {
                if (matches[i][0].distance < dist * matches[i][1].distance) {
                    good_matches.push_back(matches[i][0]);
                }
            }

            if (good_matches.size() > p - 1) {
                found[label]++;
                gm[label] = gm[label] + static_cast<int>(good_matches.size());

            }
        }
        
    }
    return std::make_pair(found, gm);
}

//same function as before but dont use the standard reference images but the provided images
std::pair<int, int> sift_object_detection_reference_img(const cv::Mat& image1, const std::vector<cv::Mat>& images, float dist, int p) {
    
    cv::Ptr<cv::SIFT> sift = cv::SIFT::create();

    std::vector<cv::KeyPoint> kp1;
    cv::Mat des1;
    sift->detectAndCompute(image1, cv::noArray(), kp1, des1);

    int found=0;
    int gm=0;

    for (size_t i = 0; i < images.size(); ++i) {
        cv::Mat img = images[i];
        std::vector<cv::KeyPoint> kp2;
        cv::Mat des2;
        sift->detectAndCompute(img, cv::noArray(), kp2, des2);

        cv::BFMatcher bf;
        std::vector<std::vector<cv::DMatch>> matches;
        bf.knnMatch(des1, des2, matches, 2);

        std::vector<cv::DMatch> good_matches;
        for (size_t i = 0; i < matches.size(); ++i) {
            if (matches[i][0].distance < dist * matches[i][1].distance) {
                good_matches.push_back(matches[i][0]);
            }
        }
        if (good_matches.size() > p - 1) {
            found ++;
            gm = gm + static_cast<int>(good_matches.size());
        }
    }
    
    return std::make_pair(found, gm);
}

std::pair<int, float> search_for_salad_bread(const cv::Mat& img, const std::vector<cv::Mat>& images, int p) {
    float dist = 0.45f;
    int found=0;
    cv::Mat blurredImg;
    cv::GaussianBlur(img, blurredImg, cv::Size(5, 5), 0);
    bool f = false;
    while (!f) {
            if (dist > 0.8f) break;
            auto is_in = sift_object_detection_reference_img(blurredImg, images, dist, p);
            if (is_in.first!=0) {
                found = is_in.first;
                }
            dist += 0.05f;
            
        }
    return {found,dist};
}

std::pair<std::vector<int>, std::vector<float>> search_for(const cv::Mat& img, const std::vector<int>& labels, int k, int p) {//image where i want to recognize food,labels that i'm searching for, type of research, number of minimum good matches (for single image of reference db)
    std::string folder_sift = "../../../Utilities/Sift_search";
    cv::Mat blurredImg;
    std::vector<int> foundLabels;
    std::vector<float> distances;
    if (k == 0) { //standard search
        cv::GaussianBlur(img, blurredImg, cv::Size(5, 5), 0);
        bool f = false;
        float dist = 0.5f;
        while (!f) {
            if (dist > 0.8f) break;
            auto is_in = sift_object_detection_label_selected(blurredImg, labels, folder_sift, dist, p);
            for (int label : labels) {
                if (is_in.first[label]!=0) {
                    if(std::find(foundLabels.begin(), foundLabels.end(), label) == foundLabels.end()){
                        foundLabels.push_back(label);
                        distances.push_back(dist);
                    }
                    if(labels.size()==foundLabels.size())f = true;
                    }
            }

            dist += 0.05f;
            
        }
    } else if(k=1){ //if k = 1 i want a less restrictive search
        bool f = false;
        float dist = 0.7f;
        while (!f) {
            if (dist > 1.05f) break;

            auto is_in = sift_object_detection_label_selected(img, labels, folder_sift, dist, p);
            for (int label : labels) {
                if (is_in.first[label]!=0) {
                    if(std::find(foundLabels.begin(), foundLabels.end(), label) == foundLabels.end()){
                        foundLabels.push_back(label);
                        distances.push_back(dist);
                    }
                    if(labels.size()==foundLabels.size())f = true;
                    }
            }
            
            dist += 0.05f;
            
        }
    }else{ //search without labels
        auto is_in = sift_object_detection(img, folder_sift, 0.75f, p);
        for (int i : is_in.first) {
            foundLabels.push_back(i);
            distances.push_back(0.75f);
        }
    }
    return {foundLabels, distances};
}