//
//  facedetect.cpp
//  pro3
//
//  Created by Victor Young on 12/30/15.
//  Copyright (c) 2015 Victor Young. All rights reserved.
//



#include <cctype>
#include <iostream>
#include <fstream>
#include <iterator>
#include <stdio.h>
#include <stdlib.h>
#include "mean.h"

using namespace std;
using namespace cv;

Point2f offset_pct(0.3,0.3);
Point dest_sz(200,200);
int startnum = 100;
int endnum = 1500;
eigenFace ef;

bool errorflag;

string prefix = "/Volumes/mac/saved2/BioID-FaceDatabase-V1.2/";
string prefix2 = "/Volumes/mac/saved2/myface/";
string prefix3 = "/Volumes/mac/saved2/mymodel/";


double Distance(Point p1,Point p2){
    double dx = p2.x - p1.x;
    double dy = p2.y - p1.y;
    return sqrt(dx*dx+dy*dy);
}

Mat ScaleRotateTranslate(Mat image, double angle, Point center){

    int nx = center.x;
    int ny = center.y;
    double cosine = cos(angle);
    double sine = sin(angle);
    double a = cosine;
    double b = sine;
    double c = nx-nx*a-ny*b;
    double d = -sine;
    double e = cosine;
    double f = ny-nx*d-ny*e;
    Mat image2 = image.clone();
    Mat warp_mat= Mat::zeros(2,3,CV_32FC1);
    warp_mat.at<float>(0,0) = a;
    warp_mat.at<float>(0,1) = b;
    warp_mat.at<float>(0,2) = c;
    warp_mat.at<float>(1,0) = d;
    warp_mat.at<float>(1,1) = e;
    warp_mat.at<float>(1,2) = f;
    
    Mat dst = Mat::zeros(image2.rows, image2.cols, image2.type());
    
    warpAffine(image2, dst, warp_mat, dst.size());
    
    
    for (int i=0; i<warp_mat.size[0]; i++) {
        for (int j=0; j<warp_mat.size[1]; j++) {
            cout << warp_mat.at<float>(i,j) << " ";
        }
        cout << endl;
    }
    return dst;
}

Mat CropFace(Mat image, Point eye_left , Point eye_right, Point2f offset_pct, Point dest_sz){
    errorflag = false;
    int offset_h = offset_pct.x*dest_sz.x;
    int offset_v = offset_pct.y*dest_sz.y;
    double rotation = atan((eye_right.y - eye_left.y)*1.0/(eye_right.x - eye_left.x));
    double dist = Distance(eye_left, eye_right);
    double reference = dest_sz.x - 2.0*offset_h;
    double scale = dist/reference;
    
    Mat image2 = ScaleRotateTranslate(image,rotation,eye_right);
//    imshow("test2", image);
//    waitKey(0);
    // crop the rotated image
    int crop_x = eye_right.x - scale*offset_h;
    int crop_y = eye_right.y - scale*offset_v;
    int size_x = dest_sz.x*scale;
    int size_y = dest_sz.y*scale;
//    cout << "image2.size[0]:" << image2.rows << " image.size[1]:"<< image2.cols <<endl;
    if (crop_x<0||crop_y<0||crop_x+size_x>image2.cols||crop_y+size_y>image2.rows) {
        cout << "image overflow" << endl;
        errorflag = true;
        return image2;
    }
//    imshow("test2", image2);
//    waitKey(0);
    
    Mat newimage(image2,cv::Rect(crop_x, crop_y, size_x, size_y));
//    Mat newimage(image2,cv::Rect(474, 1300, 600, 600));

//    imshow("test3", newimage);
//    waitKey(0);
    
    // resize it
    
    Mat newimage2 = Mat::zeros(dest_sz.x,dest_sz.y, CV_32FC1);
    resize(newimage,newimage2,Size(dest_sz.x,dest_sz.y));

    return newimage2;

}

Mat getface(int i, int flag =0){
    
//    Mat face,Point eye_left,Point eye_right,Point2f offset_pct,Point dest_sz
   
    cout << "image" << i << endl;
    
    
    stringstream ss;
    ss << i;
    string facefile,eyef ;
    if (i<20) {
        facefile = prefix2 + ss.str()+ ".jpg";
        eyef = prefix2 + ss.str() + ".eye";
    }
    else if (i<1000) {
        facefile = prefix + "BioID_0" + ss.str()+ ".pgm";
        eyef = prefix + "BioID_0" + ss.str()+ ".eye";
    }
    else{
        facefile = prefix + "BioID_" + ss.str()+ ".pgm";
        eyef = prefix + "BioID_" + ss.str()+ ".eye";
    }
    
    Mat face = imread(facefile, IMREAD_GRAYSCALE);
    if (!face.data) {
        cout << "Open file failure" << endl;
        return face;
    }
    ifstream eyefile(eyef.c_str());
    string temp;
    getline(eyefile,temp);
    eyefile>>temp;
    cout << temp << " ";
    int lx = stoi(temp);
    eyefile>>temp;
    cout << temp << " ";
    int ly = stoi(temp);
    eyefile>>temp;
    cout << temp << " ";
    int rx = stoi(temp);
    eyefile>>temp;
    cout << temp << " ";
    int ry = stoi(temp);
//    if (i<20) {
//        int temp = lx;
//        lx =ly;
//        ly =temp;
//        temp = rx;
//        rx = ry;
//        ry = temp;
//    }
//    imshow("test2", face);
//    waitKey(0);
    Mat newface =  face.clone();
    if (flag == 0) {
        newface = CropFace(face, Point(lx,ly),Point(rx,ry), offset_pct, dest_sz);
        equalizeHist(newface,newface);
    }


    return newface;

    
}

int query(Mat target,eigenFace ef2){
    
    Mat targetcopy = target.clone();
    //    Ma t weights = Mat::zeros(1, 10, CV_32FC1);
    Mat tranform(ef2.geteigenvectors().t());
    tranform.convertTo(tranform,CV_32FC1);
    targetcopy = targetcopy.reshape(1,1);
    
    Mat weight = targetcopy*tranform;
    
    double distance = 0;
    double min = 10000000000;
    int line = 0;
    Mat weights = ef2.getweights();
    for (int i=0; i<ef2.getdbsize(); i++) {
        distance = norm(weights.row(i),weight);
        if (distance < min) {
            min = distance;
            line = i;
        }
    }
    
    //Mat::mul 和 multiply都是对应元素相乘
    cout << "OK" << endl;
    return line;
}

void test1(){
    
    eigenFace ef2;
    ef2.loadFile(prefix3+"model.xml");
    
    int visit[2000];
    memset(visit, 0, 2000*sizeof(int));
//    for (int i=8; i<=12; i++) {
//        Mat target = getface(i);
//        target  = target.reshape(1, 1);
//        target.convertTo(target, CV_32FC1);
//        int index = query(target,ef2);
//        
//        string num = "";
//        if (visit[index]) {
//            char buffer[2];
//            sprintf(buffer,"%d",visit[index]);
//            num += string(buffer);
//        }
//        else{
//            num += "0";
//        }
//        visit[index] += 1;
////        cout << "num " << num <<endl;
//
//        index+= startnum;
//        char temp[5];
//        sprintf(temp, "%d", index);
//        string s(temp);
//        sprintf(temp, "%d", i-8);
//        string t(temp);
////        cout << "num " << num <<endl;
////        Mat target2 = ef.norm_0_255(target).reshape(1, dest_sz.x);
//    
//        Mat targetOriginal = getface(i,1);
//        resize(targetOriginal, targetOriginal, Size(targetOriginal.cols/5,targetOriginal.rows/5));
//        Text(targetOriginal, "Index: "+s+" OK", 10, 50);
//        imshow("target"+t, targetOriginal);
//        imshow("result"+s+"_"+num,  ef.getdb()[index-startnum]);
//        waitKey(0);
//    }

    for (int i=1500; i<=1520; i++) {
        Mat target = getface(i);
        target  = target.reshape(1, 1);
        target.convertTo(target, CV_32FC1);
        int index = query(target,ef2);
        
        string num;
        
        if (visit[index]) {
            char buffer[2];
            sprintf(buffer,"%d",visit[index]);
            num += string(buffer);
        }
        else{
            num += "0";
        }
        visit[index] += 1;

        
        index+= startnum;
        char temp[3];
        sprintf(temp, "%d", index);
        string s(temp);
        sprintf(temp, "%d", i-1500);
        string t(temp);
        Mat target2 = ef.norm_0_255(target).reshape(1, dest_sz.x);
        Text(target2, "Index: "+s+" OK", 5, 15);
        imshow("target"+t,target2);
        imshow("result"+s+"_"+num,  ef.getdb()[index-startnum]);
        waitKey(0);
    }
    
}

void Text(Mat img, string text, int x, int y)
{
    double fontScale = 0.5;
    int thickness = 2;
    int fontFace = FONT_HERSHEY_SIMPLEX;
//    InitFont(&font,CV_FONT_HERSHEY_SIMPLEX | CV_FONT_ITALIC,hscale,vscale,0,linewidth);
    Scalar textColor =Scalar(255,255,255);
    Point textPos =Point(x, y);
    putText(img, text, textPos, fontFace,fontScale,textColor,thickness);
}

int main(){
    Mat face;
    for (int i = startnum; i<endnum; i++) {
        face = getface(i);
        if (!errorflag) {
            ef.addData(face);
        }
    }
    for (int i=1; i<=8; i++) {
        face = getface(i);
        if (!errorflag) {
            ef.addData(face);
        }
    }
//    ef.train();
//    ef.saveFile(prefix3+"model.xml");
    test1();
    
//    test2();
    return 0;

}