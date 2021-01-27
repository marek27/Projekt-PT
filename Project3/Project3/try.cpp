#include "opencv2/objdetect.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/videoio.hpp"
#include <iostream>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/core.hpp>
#include "opencv2/face.hpp"
#include "opencv2\highgui\highgui.hpp"
#include "opencv2\objdetect\objdetect.hpp"
#include "opencv2\opencv.hpp"
#include<direct.h>
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/face/facerec.hpp"
#include <string> 

using namespace cv::face;

using namespace std;
using namespace cv;
void detectAndDisplay(Mat frame);
void addFace();
static void imgRead(vector<Mat>& images, vector<int>& labels);
void eigenFaceTrainer();
void FaceRecognition();

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

    if (!face_cascade.load(face_cascade_name))
    {
        cout << "Error loading face cascade\n";
        return -1;
    };

    int camera_device = parser.get<int>("camera");

    VideoCapture capture;
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

        int choice;
        cout << "1. Rozpoznaj twarz\n";
        cout << "2. dodaj twarz\n";
        cout << "Wybór: ";
        cin >> choice;
        switch (choice)
        {
        case 1:
            detectAndDisplay(frame);
            FaceRecognition();
            break;
        case 2:
            addFace();
            eigenFaceTrainer();
            break;
        default:
            return 0;
        }

        if (waitKey(10) == 27)
        {
            break;
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
    cout << "\nPodaj swoje imie:  ";
    cin >> name;

    VideoCapture capture(0);

    if (!capture.isOpened())
        return;

    if (!face_cascade.load("C:\\tools\\build\\install\\etc\\haarcascades\\haarcascade_frontalface_alt.xml"))
    {
        cout << "error" << endl;
        return;
    };

    Mat frame;
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
            cout << "Dodano twarz";
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
static void imgRead(vector<Mat>& images, vector<int>& labels) {
    vector<cv::String> fn;
    filename = "C:\\Dev\\Faces\\";
    glob(filename, fn, false);

    size_t count = fn.size();

    for (size_t i = 0; i < count; i++)
    {
        string itsname = "";
        char sep = '\\';
        size_t j = fn[i].rfind(sep, fn[i].length());
        if (j != string::npos)
        {
            itsname = (fn[i].substr(j + 1, fn[i].length() - j - 6));
        }
        images.push_back(imread(fn[i], 0));
        labels.push_back(atoi(itsname.c_str()));
    }
}

void eigenFaceTrainer() {
    vector<Mat> images;
    vector<int> labels;
    imgRead(images, labels);

    cout << "Trening rozpoczety" << endl;
    Ptr<EigenFaceRecognizer> model = EigenFaceRecognizer::create();


    model->train(images, labels);
    model->save("C:\\Dev\\Faces\\eigenface.yml");

    cout << "Trening skonczony" << endl;
    waitKey(10000);
}

void  FaceRecognition() {

    cout << "rozpoznawanie rozpoczete" << endl;
    cv::Ptr<cv::face::FaceRecognizer> model = cv::face::FisherFaceRecognizer::create();
    model->read("C:\\Dev\\Faces\\eigenface.yml");

    Mat testSample = imread("C:\\Dev\\Faces\\0.jpg", 0);

    int img_width = testSample.cols;
    int img_height = testSample.rows;

    string window = "Capture - face detection";

    if (!face_cascade.load("C:\\tools\\build\\install\\etc\\haarcascades\\haarcascade_frontalface_alt.xml")) {
        cout << " Error loading file" << endl;
        return;
    }

    VideoCapture cap(0);


    if (!cap.isOpened())
    {
        cout << "exit" << endl;
        return;
    }
    namedWindow(window, 1);
    long count = 0;
    string Pname = "";

    while (true)
    {
        vector<Rect> faces;
        Mat frame;
        Mat graySacleFrame;
        Mat original;


        cap >> frame;
        cap.read(frame);
        count = count + 1;

        if (!frame.empty()) {
            original = frame.clone();

            cvtColor(original, graySacleFrame, COLOR_BGR2GRAY);
            equalizeHist(graySacleFrame, graySacleFrame);

            face_cascade.detectMultiScale(graySacleFrame, faces, 1.1, 3, 0, cv::Size(90, 90));

            std::string frameset = std::to_string(count);
            std::string faceset = std::to_string(faces.size());

            int width = 0, height = 0;

            cv::Rect roi;

            for (int i = 0; i < faces.size(); i++)
            {
                Rect face_i = faces[i];

                Mat face = graySacleFrame(face_i);

                Mat face_resized;
                cv::resize(face, face_resized, Size(img_width, img_height), 1.0, 1.0, INTER_CUBIC);

                int label = -1; double confidence = 0;
                model->predict(face_resized, label, confidence);

                cout << " confidence " << confidence << " Label: " << label << endl;

                Pname = to_string(label);

                rectangle(original, face_i, CV_RGB(190, 0, 2), 1);
                string text = Pname;

                int pos_x = std::max(face_i.tl().x - 10, 0);
                int pos_y = std::max(face_i.tl().y - 10, 0);

                putText(original, text, Point(pos_x, pos_y), FONT_HERSHEY_COMPLEX_SMALL, 1.0, CV_RGB(190, 0, 2), 1.0);


            }


            putText(original, "Ramka: " + frameset, Point(30, 60), FONT_HERSHEY_COMPLEX_SMALL, 1.0, CV_RGB(0, 255, 0), 1.0);
            putText(original, "Ilosc rozpoznanych osob " + to_string(faces.size()), Point(30, 90), FONT_HERSHEY_COMPLEX_SMALL, 1.0, CV_RGB(0, 255, 0), 1.0);

            cv::imshow(window, original);

        }
        if (waitKey(30) >= 0) break;
    }
}

