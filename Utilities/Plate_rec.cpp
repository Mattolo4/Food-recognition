#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <vector>
#include "Detector_lib.hpp"
#include <filesystem>
#include <algorithm>
#include "Classes.h"

// std::vector<cv::Mat> load_images_from_folder_l(const std::string& folderPath) {
//     std::vector<cv::Mat> images;
//     std::vector<int> firstLetters;
//     WIN32_FIND_DATA findData;
//     HANDLE hFind = FindFirstFile((folderPath + "/*.*").c_str(), &findData);
//     if (hFind != INVALID_HANDLE_VALUE) {
//         do {
//             std::string fileName = findData.cFileName;
//                 if (fileName != "." && fileName != "..") {
//                     std::string filePath = folderPath + "/" + findData.cFileName;
//             cv::Mat image = cv::imread(filePath);
//             if (!image.empty()) {
//                 images.push_back(image);
//                 /////////////////
//                 //cv::imshow("out", image);
//                 //cv::waitKey(0);
//                 }
//             }
            
//         } while (FindNextFile(hFind, &findData));
//         FindClose(hFind);
//     }
//     return {images};
// }
std::vector<cv::Mat> load_images_from_folder_l(const std::string& folderPath) {
    std::vector<cv::Mat> images;

    for (const auto& entry : std::filesystem::directory_iterator(folderPath)) {
        if (entry.is_regular_file()) {
            std::string fileName = entry.path().filename().string();
            if (fileName != "." && fileName != "..") {
                std::string filePath = entry.path().string();
                cv::Mat image = cv::imread(filePath);
                
                if (!image.empty()) { //if the label is valid
                    images.push_back(image);
                }
            }
        }
    }
    return {images};
}

cv::Scalar average_color_5x5(const cv::Mat& img, int x, int y) {
    int x_start = (std::max)(0, x - 2);
    int y_start = (std::max)(0, y - 2);
    int x_end = (std::min)(img.cols, x + 3);
    int y_end = (std::min)(img.rows, y + 3);

    cv::Mat patch = img(cv::Range(y_start, y_end), cv::Range(x_start, x_end));

    cv::Scalar avg_color = cv::mean(patch);

    return avg_color;
}

cv::Scalar average_color_35x35(const cv::Mat& img, int x, int y) {
    int x_start = (std::max)(0, x - 17);
    int y_start = (std::max)(0, y - 17);
    int x_end = (std::min)(img.cols, x + 18);
    int y_end = (std::min)(img.rows, y + 18);

    cv::Mat patch = img(cv::Range(y_start, y_end), cv::Range(x_start, x_end));

    cv::Scalar avg_color = cv::mean(patch);

    return avg_color;
}

std::vector<cv::Scalar> get_colors(const cv::Mat& img, const std::vector<cv::Point>& points) {
    std::vector<cv::Scalar> data;
    for (const cv::Point& point : points) {
        cv::Scalar color = average_color_5x5(img, point.x, point.y);
        data.push_back(color);
    }
    return data;
}

cv::Scalar get_center_color(const cv::Mat& img) {
    int center_x = img.cols / 2;
    int center_y = img.rows / 2;
    cv::Scalar color = average_color_35x35(img, center_x, center_y);
    return color;
}

plate processImages(const cv::Mat& img) {

            cv::Scalar c = get_center_color(img);
            // std::cout << "average center color" << std::endl;
            // std::cout << c << std::endl;
            std::vector<int> search = {};
            int search_m=0; // int used for remember what we were searching for
            if (c[2] > 110 && c[2] > c[1] && c[2] > c[0]) { // lot of red
                if (std::abs(c[2] - c[1]) < 50) { // possible yellow
                    if (c[0] > 75) { //(low blue) possible cotoletta patate or ragu or riso (bad case)
                         search = {0, 3, 4, 5};
                         search_m=10;
                    } else { // possible cotoletta w patate or pollo fagioli or riso
                         search = {2, 3, 4, 5, 6};
                         search_m=11;
                    }
                } else { // more red than green, possible strong red (sugo or cozze or center salad)
                    if (c[0] < 40) {
                         search = {0, 9};
                         search_m=12;
                    } else {
                         search = {4}; // possible riso
                         search_m=13;
                    }
                }
            } else if (c[1] > 80 && c[1] > c[0] && c[1] > c[2]) { // prevalence of green
                 search = {0, 4}; // possible pesto or rice (bad case)
                 search_m=20;
            } else { // non prevalence of red or green
                 search = {2, 5, 6, 7, 8}; // possible pollo, fagioli_lonza, fagioli_polpo_patate
                search_m=21;
            }

            int p=7; //number of good mathces that have to be found
            std::pair<std::vector<int>, std::vector<float>> found = search_for(img, search, 0,p);
            while(found.first.empty() && p>1){
                p--;
                std::pair<std::vector<int>, std::vector<float>> found = search_for(img, search, 1,p);
            }
            
            int out[3] = {-1,-1,-1};

            switch (search_m)
            {
            case 10:{

                bool contains_0 = std::find(found.first.begin(), found.first.end(), 0) != found.first.end();
                bool contains_3 = std::find(found.first.begin(), found.first.end(), 3) != found.first.end();
                bool contains_5 = std::find(found.first.begin(), found.first.end(), 5) != found.first.end();
                
                if(contains_0){
                  out[0] = 3;//"pasta al ragu"
                }else if(contains_3 || contains_5){
                    out[0] = 7;//"cotoletta con patate";
                    out[1] = 11;  
                }else out[0] = 5;// "riso"; 

                break;
                }
                
            case 11:{

                bool contains_2 = std::find(found.first.begin(), found.first.end(), 2) != found.first.end();
                bool contains_4 = std::find(found.first.begin(), found.first.end(), 4) != found.first.end();
                bool contains_6 = std::find(found.first.begin(), found.first.end(), 6) != found.first.end();
                 
                if(contains_4) out[0] = 5;// "riso";
                else if(contains_6)
                    if(contains_2){
                        out[0] = 8;
                        out[1] = 10;  
                    } //"pollo fagioli";
                    else out[0] = 8;//"pollo";
                else {
                  out[0] = 7;
                  out[1] = 11;  
                }//"cotoletta con patate";
                // 2 e 3 cotoletta
                // 2 3 4 5 riso (forse conviene controlarlo per primo)
                // 2 3 5 6 pollo con fagioli
                break;
                }

            case 12:{
                // 9 cozze
                bool contains_9 = std::find(found.first.begin(), found.first.end(), 9) != found.first.end();
                bool contains_0 = std::find(found.first.begin(), found.first.end(), 0) != found.first.end();
                
                if(contains_9) out[0] = 4;//"cozze";
                else if(contains_0) out[0] = 2;//"sugo";
                else out[0] = 4;//"cozze";

                break;
                }
                

            case 13:
                //riso
                out[0] = 5;//"riso";
                break;

            case 20:{
                bool contains_0 = std::find(found.first.begin(), found.first.end(), 0) != found.first.end();
                if(contains_0) out[0] = 1;//"pasta al pesto";
                else out[0] = 5;//"riso";      
                // 0 pasta pesto
                // 4 riso
                break;
                }

            case 21:{
                bool contains_2 = std::find(found.first.begin(), found.first.end(), 2) != found.first.end();
                bool contains_5 = std::find(found.first.begin(), found.first.end(), 5) != found.first.end();
                bool contains_6 = std::find(found.first.begin(), found.first.end(), 6) != found.first.end();
                bool contains_7 = std::find(found.first.begin(), found.first.end(), 7) != found.first.end();
                bool contains_8 = std::find(found.first.begin(), found.first.end(), 8) != found.first.end();
                //con 2 6 8 || 5 8 fagioli e lonza
                // con 2 5 6 8 pollo (quindi prima conviene verificare questo)
                // 2 5 7 polpo con patate e fagioli
                if(contains_2 && contains_5 && contains_7) {
                    out[0] = 9;
                    out[1] = 10;
                    out[2] = 11;
                }//"polpo con patate e fagioli";
                else if (contains_6){
                    if (!contains_8) out[0] = 8;//"pollo";
                    else if(contains_2 && contains_5) out[0] = 8;//"pollo";
                    else {
                        out[0] = 6;
                        out[1] = 10;  
                    }//"fagioli e lonza";
                }
                else {
                  out[0] = 6;
                  out[1] = 10;  
                }//"fagioli e lonza";
                break;
                }
            
            default:{
                out[0] = 0;//"unknown";
                break;
                } 
            }
            
            
            
            //////////////////////
            // std::cout << "\ncose cercate:" << std::endl;
            // for (int i : search) {
            //    std::cout << i << " ";
            // }
            // std::cout << "\ncose trovate:" << std::endl;
            // for (int i : found.first) {
            //    std::cout << i << " ";
            // }
            // std::cout << std::endl;
            // for (float i : found.second) {
            //    std::cout << i << " ";
            // }
            // std::cout << std::endl;
            
            plate plate_r;
            std::vector<food> Food_vect;
            
            for (int i = 0; i < 3; i++) {
                if(out[i]!=-1){
                    food food_t;
                    food_t.ID = out[i];
                    Food_vect.push_back(food_t);
                    }
            }
            plate_r.foods = Food_vect;
            //std::cout << std::endl;

            return plate_r;
}

bool is_this_salad(const cv::Mat& img){
    std::string salad_folder="../../../Utilities/Salad_folder";
    std::vector<cv::Mat> data = load_images_from_folder_l(salad_folder);
    bool salad = false;
    cv::Scalar c = get_center_color(img);
    if(c[2]>150 && c[1]>50 && c[0]<50 || c[2]>110 && c[1]>60 && c[0]<50){ //orange/red or lot of green
        std::pair<int, float> is_in = search_for_salad_bread(img,data,10);
        if(is_in.first!=0) salad = true;
    }
    return salad;
}
bool is_this_bread(const cv::Mat& img){
    std::string bread_folder = "../../../Utilities/Bread_folder";
    std::vector<cv::Mat> data = load_images_from_folder_l(bread_folder);
    bool bread = false;
    std::pair<int, float> is_in = search_for_salad_bread(img,data,7);
    if(is_in.first!=0) bread = true;
    
    return bread;
}

//this function is used to ckeck where is the first plate in the leftover try
int process_leftover(const std::vector<cv::Mat>& images,const plate& plate_l) {

    for(auto f : plate_l.foods){
        if(f.ID==1){//pasta pesto
            if(images.size()==1){
                //std::cout<<"solo un piatto "<<std::endl;
                std::pair<std::vector<int>, std::vector<float>> found = search_for(images.at(0), {0}, 0,7);
                bool contains_0 = std::find(found.first.begin(), found.first.end(), 0) != found.first.end();
                if(contains_0) return 0; // first plate index 0
                else return -1; // no first plate
            }
            int i=0;
            int id_pasta=0;
            int max_g=0;
            for(auto img : images){
                cv::Scalar c = get_center_color(img);
                ////////
                // std::cout<<"colors: "<<c[0]<<" "<<c[1]<<" "<<c[2]<<" "<<std::endl;
                // cv::imshow("IMG", img);
                // cv::waitKey(0);
                if(c[1]>max_g && c[0]<100){
                    max_g=c[1];
                    id_pasta=i;
                }
                i++;
            }
            return id_pasta;
        }else if(f.ID==2 || f.ID==3 || f.ID==4){ //red pasta
            if(images.size()==1){
                std::pair<std::vector<int>, std::vector<float>> found = search_for(images.at(0), {0}, 0,7);
                bool contains_0 = std::find(found.first.begin(), found.first.end(), 0) != found.first.end();
                if(contains_0) return 0; // first plate index 0
                else return -1; // no first plate
            }
            int i=0;
            int id_pasta=0;
            int max_r=0;
            int min_b=999;
            for(auto img : images){
                cv::Scalar c = get_center_color(img);
                /////////
                // std::cout<<"colors: "<<c[0]<<" "<<c[1]<<" "<<c[2]<<" "<<std::endl;
                // cv::imshow("IMG", img);
                // cv::waitKey(0);
                if(c[2]>max_r && c[0]<155 && f.ID==4 || c[2]>max_r && c[0]<65){
                    max_r=c[2];
                    id_pasta=i;
                    min_b=c[0];
                }
                i++;
            }
            return id_pasta;
        }else if(f.ID==5){//rice case
            if(images.size()==1){
                std::pair<std::vector<int>, std::vector<float>> found = search_for(images.at(0), {4}, 0,4);
                bool contains_4 = std::find(found.first.begin(), found.first.end(), 4) != found.first.end();
                if(contains_4) return 0; // first plate index 0
                else return -1; // no first plate
            }
            std::vector<bool> rice_here;
            int i=0;
            for(auto img : images){
                ////////
                // cv::imshow("IMG", img);
                // cv::waitKey(0);
                std::pair<std::vector<int>, std::vector<float>> found = search_for(img, {4}, 0,4);
                bool contains_4 = std::find(found.first.begin(), found.first.end(), 4) != found.first.end();
                if(contains_4){
                    rice_here.push_back(i);
                }
            }
            if(rice_here.size()==1) return rice_here.at(0);
            else{
                int i=0;
                int id_riso=0;
                int min=999;
                std::vector<bool> rice_here;
                for(auto img : images){
                    cv::Scalar c = get_center_color(img);
                    ///////////
                    // std::cout<<"colors: "<<c[0]<<" "<<c[1]<<" "<<c[2]<<" "<<std::endl;
                    // cv::imshow("IMG", img);
                    // cv::waitKey(0);
                    int sum = c[0]+c[1]+c[2];
                    if(sum<min && c[2]>99){
                    min=sum;
                    id_riso=i;
                }
                    i++;
                }
                return id_riso;
            }
        
        }
    }
            

}