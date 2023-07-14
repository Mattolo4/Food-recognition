#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <stdio.h>

using namespace cv;
using namespace std;

int main(){
    string base = "../../assets/tray1/";

    string path = base + "food_image.jpg";
    Mat img1 = imread(path);
    try
    {
        if (img1.empty())
        {
            throw invalid_argument("The img is empty!");
        }
    }
    catch (invalid_argument &e)
    {
        cout << "Exception: " << e.what() << endl;
        return 1;
    }

    cout << "Showing img" << endl;
    imshow("Image food", img1);
    waitKey(0);

    return 0;
}