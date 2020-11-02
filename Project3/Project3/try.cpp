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
string window_name = "Face";
const string img_name = "twarze.jpeg";

void detectFace(Mat img);

int main()
{
    Mat img;                                          
    img = imread(img_name);                           
    if (!img.data)                                   
    {
        cout << "Nie znaleziono pliku " << img_name << ".";
        return -1;
    }
    if (!face_cascade.load(face_cascade_name))        
    {
        cout << "Nie znaleziono pliku " << face_cascade_name << ".";
        return -2;
    }
    namedWindow(window_name);   
    detectFace(img);
    waitKey(0);                                           
    return 0;
}

void detectFace(Mat img)
{
    vector<Rect> faces;                            
    Mat img_gray;                            

    cvtColor(img, img_gray, 1);               

    face_cascade.detectMultiScale(img_gray, faces, 1.1, 3, 0, Size(50, 50));
    for (unsigned i = 0; i < faces.size(); i++)
    {
        Rect rect_face(faces[i]);  
        rectangle(img, rect_face, Scalar(120, 5, 86), 2, 2, 0);
    }
    imshow(window_name, img);                       
}