#include <opencv2/core/core.hpp>                      
#include <opencv2/highgui/highgui.hpp>                    
#include "opencv2/objdetect/objdetect.hpp" 
#include "opencv2/imgproc/imgproc.hpp" 
#include <string> 
#include <iostream> 

using namespace cv;
using namespace std;

string face_cascade_name = "haarcascade_frontalface_alt.xml";
CascadeClassifier face_cascade;
string window_name = "Hello Face !";


void detectFace(Mat frame);

int main(int argc, char** argv)
{
    Mat frame;
    VideoCapture cap;
    int deviceID = 0;
    int apiID = cv::CAP_ANY;
    cap.open(deviceID, apiID);
    if (!cap.isOpened()) {
        cerr << "ERROR! Unable to open camera\n";
        return -1;
    }
    cap.read(frame);

    for (;;)
    {

        cap.read(frame);

        if (frame.empty()) {
            cerr << "ERROR! blank frame grabbed\n";
            break;
        }

        imshow("Live", frame);
        if (waitKey(5) >= 0)
            break;
    }

    namedWindow(window_name, 1);
    detectFace(frame);
    waitKey(0);
    return 0;
}

void detectFace(Mat img)
{
    vector<Rect> faces;
    Mat img_gray;

    cvtColor(img, img_gray, 6);
    face_cascade.detectMultiScale(img_gray, faces, 1.1, 3, 0 | 1, Size(50, 50));
    for (unsigned i = 0; i < faces.size(); i++)
    {
        Rect rect_face(faces[i]);

        rectangle(img, rect_face, Scalar(120, 5, 86), 2, 2, 0);
    }
    imshow("Live", img);
}
