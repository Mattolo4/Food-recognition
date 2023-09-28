#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <stdio.h>
#include <map>
#include <vector>
#include <string>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <sstream>

using namespace cv;
using namespace std;

typedef Vec<uchar, 3> Vec3b;

float IOU_TRESHOLD = 0.5;

// Returns the intersection map for given 2 maps 
std::map<int, std::vector<int>> ComputeIntersection(std::map<int, std::vector<int>> map1,
                                                    std::map<int, std::vector<int>> map2){

    // vectors used to compute the intersection map
    vector<Point> up_leftGT      = {};
    vector<Point> up_leftPred    = {};
    vector<int>   widthPred      = {};
    vector<int>   widthGT        = {};
    vector<int>   heightGT       = {};
    vector<int>   heightPred     = {};

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
        Point p2(min(up_leftGT[i].x +  widthGT[i], up_leftPred[i].x +  widthPred[i]),
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


// compute the area of each rect defined in the map
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

// prints the confidence value (iou) map
void printIoU(vector<pair<int, float>> vec){
    cout << "ID:  Iou" << endl;
    for (const auto& item : vec){
        cout << item.first << ":  " << item.second << endl;
    }
    cout << endl;
    return;
}

// prints the matches (TP, FP, NP) vector
void printMatches(vector<pair<int, string>> vec){
    cout << "ID:  Match" << endl;
    for (const auto& item : vec){
        cout << item.first << ":  " << item.second << endl;
    }
    cout << endl;
    return;
}

// prints the cumulative sum of TP ordered based on the IoU value
void printCumTP(vector<pair<int, int>> vec){
    cout << "ID:  Cumulative TP" << endl;
    for(const auto& item: vec){
        cout << item.first << ":  " << item.second << endl;
    }
    cout << endl;
    return;
}

// prints the cumulative sum of TP ordered based on the IoU value
void printCumFP(vector<pair<int, int>> vec){
    cout << "ID:  Cumulative FP" << endl;
    for(const auto& item: vec){
        cout << item.first << ":  " << item.second << endl;
    }
    cout << endl;
    return;
}

// prints the precision value 
void printPrecision(vector<pair<int, float>> vec){
    cout << "ID:  Precision" << endl;
    for (const auto& item : vec){
        cout << item.first << ":  " << item.second << endl;
    }
    cout << endl;
    return;
}

// prints the recall value 
void printRecall(vector<pair<int, float>> vec){
    cout << "ID:  Recall" << endl;
    for (const auto& item : vec){
        cout << item.first << ":  " << item.second << endl;
    }
    cout << endl;
    return;
}

// prints the cumulative sum of TP ordered based on the IoU value
void printSegments(vector<pair<int, int>> vec){
    cout << "Px  - Pixel count" << endl;
    for(const auto& item: vec){
        cout << item.first << " - " << item.second << endl;
    }
    cout << endl;
    return;
}


// Compute the IoU for 2 given map: gtbb, pred (ground trouth map, prediction map)
vector<pair<int, float>> getIoU(Mat img, std::map<int, std::vector<int>> gtbb, std::map<int, std::vector<int>> pred){

    try{
        if(gtbb.size() != pred.size()){
            throw invalid_argument("The maps' size is different!");
        }
    }catch(invalid_argument &e){
        cout << "Exception: " << e.what() << endl;        
    }

    // Computing IoU
    std::map<int, std::vector<int>> intersections = ComputeIntersection(gtbb, pred);
    
    vector<int>   areaGT     = getArea(gtbb);
    vector<int>   areaPred   = getArea(pred);
    vector<int>   areainters = getArea(intersections);
    vector<float> areaUnion  = {};
    vector<float> iou        = {};

    for(int i=0; i<areaGT.size(); i++){
        areaUnion.push_back(areaGT[i] + areaPred[i] - areainters[i]);
        iou.push_back(areainters[i] / areaUnion[i]);
    }

    // plotting the intersections area
    img = plotBB(img, intersections, Scalar(0, 127, 255));
    img = plotAreaBB(img, intersections, Scalar(0, 0, 255));


    // Inserting Id and its relative iou in a map
    std::map<int, float> pred_iou = {};

    //putting the text in the bottom left corner describing the IoU for each intersection
    int i=0;
    for(const auto& item : intersections){

        int fontface = cv::FONT_HERSHEY_PLAIN;
        double scale = 0.95;
        int thickness = 1;
        int baseline = 0;
        string info = "ID: " + to_string(item.first) + " - IoU: " + to_string(iou[i]);
        Point origin(item.second[0], item.second[1] + item.second[3] + 10);  // bottom left coords

        // get the size of the text to make the background dynamic
        Size text = getTextSize(info, fontface, scale, thickness, &baseline);

        rectangle(img, origin + Point(0, baseline), origin + Point(text.width, -text.height), Scalar(0, 0, 230), FILLED);
        putText(img, info, origin, fontface, scale, Scalar(0, 0, 0), thickness, LINE_AA);

        // Inserting Id and its relative iou in a map
        pred_iou.insert({item.first, iou[i]});
        i += 1;
    }

    // Declare vector of pairs
    vector<pair<int, float>> pairVec;

    for(const auto& item: pred_iou){
        pairVec.push_back(item);
    }

    // Sort in ascending order the key value pairs (id - iou). 
    // For this purpose we use a vector of pair instead of a map
    sort(pairVec.begin(), pairVec.end(), [](const auto& a, const auto& b){
        return a.second > b.second;
    });
    return pairVec;
}




// get the matches (TP; FP ; TN; FN) depending on the IoU value associated to each ID
vector<pair<int, string>> getMatches(vector<pair<int, float>> vec){
    
    vector<pair<int, string>> match_vect = {};
    for(const auto& item: vec){
        string match = "";
        if(item.second == 0.0){
            match = "FN";
        }else if (item.second <= IOU_TRESHOLD) {
            match = "FP";
        }else{
            match = "TP";
        }
        match_vect.push_back(make_pair(item.first, match));
    }
    return match_vect;
}

// Compute the Cumulative TP having in Input the ordered vector,
// returns a vector of pairs (ID, CumTP)
vector<pair<int, int>> getCumTP(vector<pair<int, string>> vec){

    vector<pair<int, int>> pred_cumTP = {};
    int cumTP = 0;
    for(const auto& item: vec){
        if(item.second == "TP"){
            cumTP += 1;
        }
        pred_cumTP.push_back(make_pair(item.first, cumTP));
    }
    return pred_cumTP;
} 

// Compute the Cumulative FP having in Input the ordered vector,
// returns a vector of pairs (ID, CumFP)
vector<pair<int, int>> getCumFP(vector<pair<int, string>> vec){

    vector<pair<int, int>> pred_cumFP = {};
    int cumFP = 0;
    for(const auto& item: vec){
        if(item.second == "FP"){
            cumFP += 1;
        }
        pred_cumFP.push_back(make_pair(item.first, cumFP));
    }
    return pred_cumFP;
} 


// Compute the Precision as P  =  TP/(TP + FP)
vector<pair<int, float>> getPrecision(vector<pair<int, int>> cumTP, vector<pair<int, int>> cumFP){

    vector<pair<int, float>> precision = {};
    vector<int> cumTP_vec;

    for(const auto& item: cumTP){
        cumTP_vec.push_back(item.second);
    }

    int i=0;
    for(const auto& item: cumFP){
        float prec = cumTP_vec[i] / float(cumTP_vec[i] + item.second);
        precision.push_back(make_pair(item.first, prec));
        i+=1;
    }
    return precision;
}


// get the Recall according to R = TP / Total Ground Truths
vector<pair<int, float>> getRecall(vector<pair<int, int>> cumTP, float total){

    vector<pair<int, float>> recall = {};
    for(const auto& item: cumTP){
        float rec = item.second / total;
        recall.push_back(make_pair(item.first, rec));        
    }
    return recall;
}


// Allows to click on a specific pixel to obtain some info
void onMouse(int event, int x, int y, int f, void *img){

    //when the left mouse button is released
    if(event == EVENT_LBUTTONUP){
        Mat input = *(Mat *) img;
        
        //x, y, are inverted
        cout << "Pixel: x= " << x << "; y= " << y << endl;
        cout << "B: " << (int) input.at<uchar>(y, x) << endl;

        return;
    }
}

map<int, vector<int>> getBBFromFile(string fileName, bool debug){
    
    map<int, vector<int>> dataMap;
    // Open the input file
    ifstream inputFile(fileName);

    if (!inputFile){
        cerr << "Error opening input file." << endl;
    }

    string line;
    while (getline(inputFile, line)) {
        int id;
        vector<int> values;

        // Extract ID from the line
        size_t idPosition = line.find("ID: ");
        if (idPosition == string::npos) {
            cerr << "Error: ID not found in the line." << endl;
            continue;
        }
        id = stoi(line.substr(idPosition + 4));

        // Extract values from the line
        size_t valuesPosition = line.find("; [");
        if (valuesPosition == string::npos) {
            cerr << "Error: Values not found in the line." << endl;
            continue;
        }
        string valuesStr = line.substr(valuesPosition + 3);
        valuesStr.pop_back(); // Remove the closing bracket ']'

        // Parse the values into the vector
        istringstream iss(valuesStr);
        int value;
        while (iss >> value) {
            values.push_back(value);
            // Read and ignore the comma and space after each value
            iss.ignore();
        }

        // Insert into the map
        dataMap[id] = values;
    }

    // Close the file
    inputFile.close();

    if(debug){
        // Printing the dataMap for verification
        for (const auto& entry : dataMap) {
            cout << entry.first << ": [";
            for (size_t i = 0; i < entry.second.size(); ++i) {
                cout << entry.second[i];
                if (i != entry.second.size() - 1)
                    cout << " ";
            }
            cout << "]" <<endl;
        }
    }
}


// ###### IOU #######

// Detect the value of the pixel in the predicted mask as to obtain the ID of the food
int getId(Mat mask){
    try{
        if(mask.channels() != 1){
            throw invalid_argument("The image is not a GrayScale!");
        }
        if(mask.empty()){
            throw invalid_argument("The image is empty!");
        }
    }catch(invalid_argument &e){
        cout << "Exception: " << e.what() << endl;
    }

    int id = -1;
    for(int i=0; i<mask.rows; i++){
        for(int j=0; j<mask.cols; j++){
            int color = mask.at<uchar>(i, j);
            if(color != 0){
                id = color;
                break;
            }
        }
    }
    return id;
}


// Compute a nice visualization of GT mask, predicted mask, intersection and IoU
Mat getDetail(Mat maskGT, Mat maskPred, int id, float iou){

    cvtColor(maskGT, maskGT, COLOR_GRAY2BGR);
    cvtColor(maskPred, maskPred, COLOR_GRAY2BGR);
    Mat detail;
    copyTo(maskGT, detail, Mat());

    Vec3b segmentID(id, id, id);

    Point origin;   // For the rectangle origin
    for(int i=0; i<detail.rows; i++){
        for(int j=0; j<detail.cols; j++){

            Vec3b color = detail.at<Vec3b>(i, j);

            // pixel light blue if it s intersection
            if(color == segmentID && maskPred.at<Vec3b>(i, j) == segmentID){ 
                detail.at<Vec3b>(i, j) = Vec3b(200, 200, 0);
                origin.x = j; origin.y = i;
            }else{
                // pixel green if not intersection and ground trouth segment
                if(color == segmentID){
                    detail.at<Vec3b>(i, j) = Vec3b(0, 200, 0);
                    origin.x = j; origin.y = i;
                }
                // pixel blue if not intersection and prediction mask
                if(maskPred.at<Vec3b>(i, j) == segmentID){
                    detail.at<Vec3b>(i, j) = Vec3b(200, 0, 0);
                }
            }
        }
    }
    int fontface = cv::FONT_HERSHEY_PLAIN;
    double scale = 0.95;
    int thickness = 1;
    int baseline = 0;
    string info = "IoU: " + to_string(iou);

    // get the size of the text to make the background dynamic
    Size text = getTextSize(info, fontface, scale, thickness, &baseline);

    // center box
    origin.x -= text.width / 2;
    origin.y += 20;

    rectangle(detail, origin + Point(0, baseline), origin + Point(text.width, -text.height), Scalar(230, 230, 230), FILLED);
    putText(detail, info, origin, fontface, scale, Scalar(0, 0, 0), thickness, LINE_AA);
    return detail;
}

// Compute the number of pixels that overlaps between 2 given masks based on the id value
// returns an int
int getIntersectionCount(Mat maskGT, Mat mask, int id){

    try{
        if (maskGT.empty() || mask.empty()){
            throw invalid_argument("One of the imgs is empty!");
        }else if(maskGT.size() != mask.size()){
            throw invalid_argument("The size of the masks is different");
        }else if(maskGT.channels() != 1 || mask.channels() != 1){
            throw invalid_argument("The image is not a GrayScale!");
        }
    }
    catch (invalid_argument &e){
        cout << "Exception: " << e.what() << endl;
        return 1;
    }
    int count = 0;

    for(int i=0; i<maskGT.rows; i++){
        for(int j=0; j<maskGT.cols; j++){

            int colGT   = maskGT.at<uchar>(i, j);
            int colMask = mask.at<uchar>(i, j);

            if(colGT == id && colMask == id){   
                count += 1;
            }
        }
    }
    return count;
}


// Compute the IoU for segmentation
// Takes in input the 2 vectors that containt details about the segmented parts respectevely in the
// Ground Trouth mask and the predicted mask; the debig flas is added if more info are rquired
// Compute the iou for 2 given masks
float getIoU_fromMasks(Mat gt, Mat mask, bool debug){
    
    cvtColor(gt, gt, COLOR_BGR2GRAY);

    try{
        if (gt.empty() || mask.empty()){
            throw invalid_argument("One of the imgs is empty!");
        }else if(gt.size() != mask.size()){
            throw invalid_argument("The size of the masks is different");
        }else if(gt.channels() != 1 || mask.channels() != 1){
            throw invalid_argument("The image is not a GrayScale!");
        }
    }catch (invalid_argument &e){
        cout << "Exception: " << e.what() << endl;
        return 1;
    }


    int id = getId(mask);

    int intersection = getIntersectionCount(gt, mask, id);

    try{
        if(intersection == 0){
            throw invalid_argument("The predicted ID ("+to_string(id)+") is not correct according the Ground Trouth");
        }
    }catch(invalid_argument &e){
        cout << "Exceptiom: " << e.what() << endl;
    }

    float unionMaps = 0.;

    for(int i=0; i<gt.rows; i++){
        for(int j=0; j<gt.cols; j++){

            if(gt.at<uchar>(i, j) == id){
                unionMaps += 1;
            }
            if(mask.at<uchar>(i, j) == id){
                unionMaps += 1;
            }
        }
    }
    unionMaps -= intersection;
    float iou = intersection / unionMaps;

    // Debug
    if(debug && intersection != 0){
        cout << "ID matched segment: " << id << endl;
        cout << "Intersection count: " << intersection << endl;
        cout << "IoU: " << iou << endl << endl;

        Mat detail = getDetail(gt, mask, id, iou);

        imshow("Details img", detail);
        waitKey(0);
    }
    return iou;
}

// Compute the ratio as R = pixelAfter / pixelBefore 
// Input 2 masks: Before mask, After mask
// Output: R
float getRatio(Mat maskBefore, Mat maskAfter, bool debug){

    cvtColor(maskBefore, maskBefore, COLOR_BGR2GRAY);

    int id = getId(maskAfter);  //since it contains only 2 values: 1 for bg, the other for food ID
    
    float pixelBefore = 0;
    int pixelAfter = 0;

    for(int i=0; i<maskBefore.rows; i++){
        for(int j=0; j<maskBefore.cols; j++){
            
            if(maskBefore.at<uchar>(i, j) == id){
                pixelBefore += 1;
            }
            if(maskAfter.at<uchar>(i, j) == id){
                pixelAfter += 1;
            }
        }
    }

    float ratio = pixelAfter / pixelBefore;

    if(debug){
        cout << "id 'After' img: " << id << endl;

        namedWindow("Before", 0);
        namedWindow("After",  0);
        
        resizeWindow("Before", Size(maskBefore.cols / 2, maskBefore.rows / 2));
        resizeWindow("After",  Size(maskBefore.cols / 2, maskBefore.rows / 2));

        imshow("Before", maskBefore);
        imshow("After", maskAfter);
        waitKey(0);
    }

    return ratio;
}


// get the Mean IOU having a vector of predicted masks
// computing the iou for each food item and its mean
float getMiou(Mat gt, vector<Mat> predictedMasks, bool debug){

    vector<float> iou_forEachItem = {};
    for (const auto &mask : predictedMasks){

        if(!mask.empty()){
            float iou = getIoU_fromMasks(gt, mask, debug);
            iou_forEachItem.push_back(iou);
        }
    }
    float sum = 0;
    for(int i=0; i<iou_forEachItem.size(); i++){
        sum += iou_forEachItem[i];
    }
    float mIoU = sum / iou_forEachItem.size();

    if(debug){
        cout << "Ratio: " << mIoU << endl;
    }
    return mIoU;
}


// get the mean ratio for the whole tray
float getMeanRatio(Mat gt, vector<Mat> predictedMasks, bool debug){

    vector<float> ratio_forEachItem = {};
    float size = 0;
    float sum = 0;

    for(const auto& mask: predictedMasks){
        if(!mask.empty()){
            float ratio = getRatio(gt, mask, debug);
            ratio_forEachItem.push_back(ratio);
            size += 1;
            sum += ratio;
            cout << ratio << " " << endl;
        }
    }
    float R = sum / size;
    if(debug){
        cout << "Mean ratio for the tray: " << R << endl;
    }
    return R;
}
