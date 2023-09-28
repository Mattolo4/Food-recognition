#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>

// Global variables for trackbar values
int lowHue = 0, highHue = 179;
int lowSaturation = 0, highSaturation = 255;
int lowValue = 0, highValue = 255;
cv::Mat hueChannel, satChannel,hueMask,satMask,valMask,food,mask;
cv::Mat image, out;

cv::Mat equalize(cv::Mat img){
    cv::Mat channels[3];
    cv::split(img, channels);

    for (int i = 0; i < 3; ++i) {
        cv::equalizeHist(channels[i], channels[i]);
    }
    cv::Mat out;
    cv::merge(channels, 3, out);
    return out;
}

cv::Mat tresh(int lHue, int hHue, int lSat, int hSat,int lVal, int hVal){
    cv::Mat imgHSV,out;
    cv::cvtColor(image, imgHSV, cv::COLOR_BGR2HSV);
    std::vector<cv::Mat> ChannelsHSV;
    cv::split(imgHSV, ChannelsHSV);
    cv::Mat hueChannel = ChannelsHSV[0];
    cv::Mat satChannel = ChannelsHSV[1];
    cv::Mat valChannel = ChannelsHSV[2];
    cv::Mat mask, hsmask;
    inRange(hueChannel, lHue, hHue, hueMask);
    inRange(satChannel, lSat, hSat, satMask);
    inRange(valChannel, lVal, hVal, valMask);
    bitwise_and(hueMask,satMask,hsmask);
    bitwise_and(valMask,hsmask,mask);
    image.copyTo(out, mask);
    return out;
}

// Callback function for trackbar changes
void onTrackbar(int, void*)
{
    
    cv::Mat out = tresh(lowHue,highHue,lowSaturation,highSaturation,lowValue,highValue);
    cv::imshow("Trackbars", out);
    //Extract the Hue, Saturation, Value channels
}

int main()
{
    /* cv::String path = "../../assets/tray2/";
    cv::Mat img = cv::imread(path + "food_image.jpg");
    cv::Mat lft1 = cv::imread(path + "leftover1.jpg");
    cv::Mat lft2 = cv::imread(path + "leftover2.jpg");
    cv::Mat lft3 = cv::imread(path + "leftover3.jpg");
 */
    //image = cv::imread("tatos_1.png");
    
    cv::Mat img = cv::imread("equalizedBeans.png");
    bilateralFilter(img, image, 15, 1000*0.1, 800*0.01);
    //GaussianBlur(image,image, cv::Size(9, 9),3,3);

    //image = img;
    
    //Extract the Hue, Saturation, Value channels

    // Create a black image for trackbars display
    cv::Mat trackbarImage = cv::Mat::zeros(1, 400, CV_8UC3);

    // Create a window for trackbars
    cv::namedWindow("Trackbars");

    // Create trackbars for Hue channel
    cv::createTrackbar("Low Hue", "Trackbars", &lowHue, 180, onTrackbar);
    cv::createTrackbar("High Hue", "Trackbars", &highHue, 180, onTrackbar);

    // Create trackbars for Saturation channel
    cv::createTrackbar("Low Sat", "Trackbars", &lowSaturation, 255, onTrackbar);
    cv::createTrackbar("High Sat", "Trackbars", &highSaturation, 255, onTrackbar);

    // Create trackbars for Value channel
    cv::createTrackbar("Low Value", "Trackbars", &lowValue, 255, onTrackbar);
    cv::createTrackbar("High Value", "Trackbars", &highValue, 255, onTrackbar);
     

    onTrackbar(0, 0);

    // Wait for key press to exit
    cv::waitKey(0);

    return 0;
}
