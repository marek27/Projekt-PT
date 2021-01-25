#include "opencv2/objdetect.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/videoio.hpp"
#include <iostream>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/core.hpp>
#include "face.hpp"
#include "opencv2/objdetect/objdetect.hpp" 
#include "face/facerec.hpp"
#include "opencv2/core/eigen.hpp"
#include<direct.h>


using namespace cv::face;

using namespace std;
using namespace cv;
void detectAndDisplay(Mat frame);
void addFace();
static void imgRead(vector<Mat>& images, vector<int>& labels);
void eigenFaceTrainer();
void  FaceRecognition();

CascadeClassifier face_cascade;
CascadeClassifier eyes_cascade;
string filename;
int filenumber = 0;
string name;
int main(int argc, const char** argv)
{
    CommandLineParser parser(argc, argv,
        "{help h||}"
        "{face_cascade|data/haarcascades/haarcascade_frontalface_alt.xml|Path to face cascade.}"
        "{eyes_cascade|data/haarcascades/haarcascade_eye_tree_eyeglasses.xml|Path to eyes cascade.}"
        "{camera|0|Camera device number.}");
    parser.about("\nThis program demonstrates using the cv::CascadeClassifier class to detect objects (Face + eyes) in a video stream.\n"
        "You can use Haar or LBP features.\n\n");
    parser.printMessage();
    String face_cascade_name = samples::findFile(parser.get<String>("face_cascade"));

    //-- 1. Load the cascades
    if (!face_cascade.load(face_cascade_name))
    {
        cout << "Error loading face cascade\n";
        return -1;
    };
    \
        int camera_device = parser.get<int>("camera");
    VideoCapture capture;
    //-- 2. Read the video stream
    capture.open(camera_device);
    if (!capture.isOpened())
    {
        cout << "Error opening video capture\n";
        return -1;
    }
    Mat frame;
    while (capture.read(frame))
    {
        if (frame.empty())
        {
            cout << "No captured frame -- Break!\n";
            break;
        }
        //-- 3. Apply the classifier to the frame
        //detectAndDisplay(frame);
        //addFace();
        eigenFaceTrainer();
        FaceRecognition();
        if (waitKey(10) == 27)
        {
            break; // escape
        }
    }
    return 0;
}
void detectAndDisplay(Mat frame)
{
    Mat crop;
    Mat res;
    string text;
    stringstream sstm;
    Mat frame_gray;
    cvtColor(frame, frame_gray, COLOR_BGR2GRAY);
    equalizeHist(frame_gray, frame_gray);
    //-- Detect faces
    std::vector<Rect> faces;
    face_cascade.detectMultiScale(frame_gray, faces);

    Rect roi_b;
    Rect roi_c;

    size_t ic = 0;
    int ac = 0;

    size_t ib = 0;
    int ab = 0;

    for (size_t i = 0; i < faces.size(); i++)
    {
        roi_c.x = faces[ic].x;
        roi_c.y = faces[ic].y;
        roi_c.width = (faces[ic].width);
        roi_c.height = (faces[ic].height);

        ac = roi_c.width * roi_c.height;

        roi_b.x = faces[ib].x;
        roi_b.y = faces[ib].y;
        roi_b.width = (faces[ib].width);
        roi_b.height = (faces[ib].height);


        crop = frame(roi_b);
        resize(crop, res, Size(128, 128), 0, 0, INTER_LINEAR);
        stringstream ssfn;
        filename = "C:\\Dev\\Faces\\";
        ssfn << filename.c_str() << name << filenumber << ".jpg";
        filename = ssfn.str();
        imwrite(filename, res);
        filenumber++;
        Point center(faces[i].x + faces[i].width / 2, faces[i].y + faces[i].height / 2);

        rectangle(frame, faces[i], Scalar(190, 0, 2), 2, 2, 0);
        Mat faceROI = frame_gray(faces[i]);


    }
    //-- Show what you got
    imshow("Capture - Face detection", frame);
}

void addFace()
{
    cout << "\nEnter Your Name:  ";
    cin >> name;

    VideoCapture capture(0);

    if (!capture.isOpened())
        return;

    if (!face_cascade.load("C:\\Dev\\vcpkg\\opencv\\build\\install\\etc\\haarcascades\\haarcascade_frontalface_alt.xml"))
    {
        cout << "error" << endl;
        return;
    };

    Mat frame;
    cout << "\nCapturing your face 10 times, Press 'C' 10 times keeping your face front of the camera";
    char key;
    int i = 0;

    for (;;)
    {
        capture >> frame;

        imshow("Camera", frame);
        detectAndDisplay(frame);
        i++;
        if (i == 10)
        {
            cout << "Face Added";
            break;
        }
        //break;
        int c = waitKey(10);

        if (27 == char(c))
        {
            break;
        }
        imshow("Output Capture", frame);
    }

    return;
}