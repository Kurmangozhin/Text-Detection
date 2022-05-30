#include <iostream>
#include <queue>
#include <iterator>
#include <sstream>
#include <fstream>
#include <iomanip>
#include <chrono>
#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/dnn/all_layers.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <vector>
#include <math.h>





using namespace cv;
using namespace std;



struct configs
{
    string model_path = "docx.onnx";
    double scale = 1/255.;
    int w = 512;
    int h = 512;
    
    };



void show_image(string path) {
    Mat image = imread(path);
    imshow("Image", image);
    waitKey(0);
    }






int main(int argc, char * argv[])

{


    configs c;
      
    auto net = cv::dnn::readNet(c.model_path);
    cv::Mat frame, blob, mask;


    frame = imread(argv[1]);

    int h = frame.size[0];
    int w = frame.size[1];


    cv::dnn::blobFromImage(frame, blob, c.scale, cv::Size(c.w, c.h), cv::Scalar(), true, false, CV_32F);
    net.setInput(blob);
    Mat outputs = net.forward();

    cv::Mat output(outputs.size[2], outputs.size[3], CV_32F, outputs.ptr<float>());  

    cv::resize(output, mask, cv::Size(w, h));

    vector<Vec4i> hierarchy;
    vector<vector<Point> > contours;
    vector<Rect>boundingRect(contours.size());

    mask.convertTo(mask, CV_8U);


    findContours(mask, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);


    for (int i = 0; i < contours.size(); i++) {

           cv::drawContours(frame, contours, i, cv::Scalar(255, 0, 0), 2);



    }



    cvtColor(frame, frame, cv::COLOR_BGR2RGB);

    imwrite(argv[2], frame);

    cout << "write out file : " << argv[2] << endl;

    imshow("image", frame);
    waitKey(0);

    return 0;
}

