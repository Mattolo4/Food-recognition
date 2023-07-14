#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <stdio.h>
#include <map>
#include <vector>

using namespace cv;
using namespace std;

void printMap(std::map<int, std::vector<int>> map);
Mat plotBB(Mat img, std::map<int, std::vector<int>> map, Scalar col);
Mat plotAreaBB(Mat img, std::map<int, std::vector<int>> map, Scalar col);
vector<int> getArea(std::map<int, std::vector<int>> map);
std::map<int, std::vector<int>> ComputeIntersection(std::map<int, std::vector<int>> map1,
                                                    std::map<int, std::vector<int>> map2);
vector<float> getIoU(std::map<int, std::vector<int>> gtbb, std::map<int, std::vector<int>> pred);



int main(){
    string base = "../../../assets/tray1/";

    Mat imgBase = imread(base + "food_image.jpg");
    try{
        if (imgBase.empty()){
            throw invalid_argument("The img is empty!");
        }
    }
    catch (invalid_argument &e){
        cout << "Exception: " << e.what() << endl;
        return 1;
    }

    // ***  IoU  ***// 
    // Ground-Truth Bounding box
    map <int, vector<int>> gtbb;
    gtbb.insert({ 6, {370, 436, 313, 331}});
    gtbb.insert({ 1, {737, 145, 384, 400}});
    gtbb.insert({10, {259, 532, 347, 357}});
    gtbb.insert({13, {235,  79, 243, 178}});

    // Prediction bounding box to compare
    map<int, vector<int>> pred;
    pred.insert({ 6, {368, 436, 312, 330}});
    pred.insert({ 1, {787, 195, 384, 400}});
    pred.insert({10, {289, 502, 347, 357}});
    pred.insert({13, {295, 230, 250, 101}});

    
    // creating visual computation example
    Mat img = plotBB(imgBase, gtbb, Scalar(0, 255, 0));
    img = plotBB(img, pred, Scalar(240, 0, 0));

    map<int, vector<int>> inters = ComputeIntersection(gtbb, pred);
    img = plotBB(img, inters, Scalar(0, 127, 255));
    img = plotAreaBB(img, inters, Scalar(0, 0, 255));

    // Getting the IoU
    vector<float> iou = getIoU(gtbb, pred);
    for(int i=0; i<iou.size(); i++){
        cout << i << " : " << iou[i] << endl;
    }
    imshow("Computed img", img);
    waitKey(0);

    return 0;
}

// Compute the IoU for 2 given map: gtbb, pred (ground trouth map, prediction map)
vector<float> getIoU(std::map<int, std::vector<int>> gtbb, std::map<int, std::vector<int>> pred){

    try{
        if(gtbb.size() != pred.size()){
            throw invalid_argument("The maps' size is different!");
        }
    }catch(invalid_argument &e){
        cout << "Exception: " << e.what() << endl;        
    }

    std::map<int, std::vector<int>> intersections = ComputeIntersection(gtbb, pred);

    vector<int> areaGT     = getArea(gtbb);
    vector<int> areaPred   = getArea(pred);
    vector<int> areainters = getArea(intersections);
    vector<float> iou = {};

    vector<float> areaUnion  = {};
    for(int i=0; i<areaGT.size(); i++){
        areaUnion.push_back(areaGT[i] + areaPred[i] - areainters[i]);
        iou.push_back(areainters[i] / areaUnion[i]);
    }
    return iou;
}

// Returns the intersection map for given 2 maps 
std::map<int, std::vector<int>> ComputeIntersection(std::map<int, std::vector<int>> map1,
                                                    std::map<int, std::vector<int>> map2){

    // vectors used to compute the intersection map
    vector<Point> up_leftGT    = {};
    vector<Point> up_leftPred  = {};
    vector<int> widthPred      = {};
    vector<int> widthGT        = {};
    vector<int> heightGT       = {};
    vector<int> heightPred     = {};

    for (const auto &item : map1){
        Point p(item.second[0], item.second[1]);
        up_leftGT.push_back(p);
        widthGT.push_back(item.second[2]);
        heightGT.push_back(item.second[3]);
    }

    for (const auto &item : map2){
        Point p(item.second[0], item.second[1]);
        up_leftPred.push_back(p);
        widthPred.push_back(item.second[2]);
        heightPred.push_back(item.second[3]);
    }

    // Coordinates of the area of intersection
    map<int, vector<int>> intersection;
    int i=0;
    for (const auto& item : map1){
        vector<int> coords = {};
        Point p1(max(up_leftGT[i].x, up_leftPred[i].x),
                 max(up_leftGT[i].y, up_leftPred[i].y));
        Point p2(min(up_leftGT[i].x + widthGT[i], up_leftPred[i].x + widthPred[i]),
                 min(up_leftGT[i].y + heightGT[i], up_leftPred[i].y + heightPred[i]));

        // Intersection height and width
        int iWidth  = max(p2.x - p1.x + 1, 0);
        int iHeight = max(p2.y - p1.y + 1, 0);

        // Building the intersection map
        coords.push_back(p1.x);
        coords.push_back(p1.y);
        coords.push_back(iWidth);
        coords.push_back(iHeight);
        intersection.insert({item.first, coords});

        i += 1;
    }
    return intersection;
}



// compute the areas of each rect defined in the map
vector<int> getArea(std::map<int, std::vector<int>> map){
    vector<int> areas = {};
    for (const auto &item : map){
        int area = item.second[2] * item.second[3];
        areas.push_back(area);
    }
    return areas;
}


// Plots the area with in the bb provided
Mat plotAreaBB(Mat img, std::map<int, std::vector<int>> map, Scalar col){
    Mat temp;
    copyTo(img, temp, Mat());
    for (const auto &item : map){
        Point p1(item.second[0],  item.second[1]);
        Point p2(item.second[0] + item.second[2],
                 item.second[1] + item.second[3]);

        rectangle(temp, p1, p2, col, -1, LINE_8);
    }
    addWeighted(img, 0.7, temp, 0.3, 0, img);
    return img;
}


// Plot the bounding boxes having in input the img and its associated bb
// Returns the img with bb plotted
Mat plotBB(Mat img, std::map<int, std::vector<int>> map, Scalar col){
    for(const auto& item : map){
        Point p1(item.second[0],  item.second[1]);
        Point p2(item.second[0] + item.second[2],
                 item.second[1] + item.second[3]);

        rectangle(img, p1, p2, col, 2, LINE_8);
    }
    return img;
}

// prints the bounding box map (same format as the one in the dataset)
void printMap(std::map<int, std::vector<int>> map){
    for (const auto& item : map){
        cout << item.first << ":  ";
        
        for(const auto& value : item.second){
            cout << value << " ";
        }
        cout << endl;
    }
    return;
}
// void IoU()