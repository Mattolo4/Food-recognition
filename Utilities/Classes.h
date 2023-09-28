#ifndef CLASSES
#define CLASSES


#include <opencv2/imgproc.hpp>
#include <map>



class food{
    public:
        cv::Mat foodMask;
        int ID;
        std::map <int, std::vector<int>> bbox;
        food(cv::Mat FoodMask, int id, std::map <int, std::vector<int>> BBox){
            foodMask = FoodMask;
            ID = id;
            bbox = BBox;
        }
        food(int id){
            ID = id;
        }
        food(){
        }

};

class plate{
    public:
        std::vector<food> foods;
        cv::Mat plateMask;
        cv::Mat plateImage;
        bool isEmpty;
        plate(std::vector<food> Foods, cv::Mat PlateMask, cv::Mat PlateImage, bool IsEmpty){
            foods = Foods;
            plateMask = PlateMask;
            plateImage = PlateImage;
            isEmpty = IsEmpty;
        }
        plate(){
        }
};

class tray{
    public:
        plate firstCourse;
        plate mainCourse;
        food bread;
        food salad;
        cv::Mat trayImage;
        bool hasBread;
        bool hasSalad;
        tray(plate FirstCourse,plate MainCourse,food Bread,food Salad,cv::Mat TrayImage){
            firstCourse = FirstCourse;
            mainCourse = MainCourse;
            salad = Salad;
            bread = Bread;
            trayImage = TrayImage;
        }
        tray(cv::Mat TrayImage){
            trayImage = TrayImage;
        }
        tray(){
        }
};



#endif 