#ifndef mAP
#define mAP

using namespace cv;
using namespace std;


void onMouse(int event, int x, int y, int f, void *img);
void printSegments(vector<pair<int, int>> vec);
void printMap(std::map<int, std::vector<int>> map);
void printIoU(vector<pair<int, float>> vec);
void printMatches(vector<pair<int, string>> vec);
void printCumTP(vector<pair<int, int>> vec);
void printCumFP(vector<pair<int, int>> vec);
void printPrecision(vector<pair<int, float>> vec);
void printRecall(vector<pair<int, float>> vec);


// mAP
map<int, std::vector<int>> getBBFromFile(string file, bool debug);
Mat plotBB(Mat img, std::map<int, std::vector<int>> map, Scalar col);
Mat plotAreaBB(Mat img, std::map<int, std::vector<int>> map, Scalar col);M
vector<int> getArea(std::map<int, std::vector<int>> map);
std::map<int, std::vector<int>> ComputeIntersection(std::map<int, std::vector<int>> map1,
                                                    std::map<int, std::vector<int>> map2);
vector<pair<int, float>> getIoU(Mat img, std::map<int, std::vector<int>> gtbb, std::map<int, std::vector<int>> pred);
vector<pair<int, string>> getMatches(vector<pair<int, float>> vec);
vector<pair<int, int>> getCumTP(vector<pair<int, string>> vec);
vector<pair<int, int>> getCumFP(vector<pair<int, string>> vec);
vector<pair<int, float>> getPrecision(vector<pair<int, int>> cumTP, vector<pair<int, int>> cumFP);
vector<pair<int, float>> getRecall(vector<pair<int, int>> cumTP, float total);

// IoU
int getId(Mat mask);
int getIntersectionCount(Mat maskGT, Mat mask, int id);
float getIoU_fromMasks(Mat gt, Mat mask, bool debug);
Mat getDetail(Mat maskGT, Mat maskPred, int id, float iou);

// Ratio
float getRatio(Mat maskBefore, Mat maskAfter, bool debug);
float getMiou(Mat gt, vector<Mat> predictedMasks, bool debug);
float getMeanRatio(Mat gt, vector<Mat> predictedMasks, bool debug);



#endif // mAP