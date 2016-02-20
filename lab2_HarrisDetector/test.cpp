//
//  test.cpp
//  opencvCFSS
//
//  Created by Victor Young on 12/13/15.
//  Copyright (c) 2015 Victor Young. All rights reserved.
//

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <stdio.h>
#include <stdlib.h>

using namespace cv;
using namespace std;

char source_window[] = "Source image";
char corners_window[] = "R Value";
//char corners_window[] = "Min lamda";
Mat src,x2y2, xy,mtrace,src_gray, x_derivative, y_derivative,x2_derivative, y2_derivative,
xy_derivative,x2g_derivative, y2g_derivative,xyg_derivative,dst,dst_norm, dst_norm_scaled,lamdamin,lamdamax,delta,normmax,normmin;
int thresh = 128;
int GaussianSize=7;

void onTrackbar( int, void* );

int main( int argc, char** argv )
{
    src = imread( "/Users/Victor/Desktop/lena.jpg");
    namedWindow( source_window, CV_WINDOW_AUTOSIZE );
    createTrackbar( "Threshold", source_window, &thresh, 255, onTrackbar);
    imshow( source_window, src );
    onTrackbar(thresh,0);
    
    waitKey(0);
    return(0);
}

void onTrackbar( int, void* ){
    cvtColor( src, src_gray, CV_BGR2GRAY );
    Sobel(src_gray, x_derivative, CV_32FC1 , 1, 0, 3, BORDER_DEFAULT);
    //namedWindow( corners_window, CV_WINDOW_AUTOSIZE );
//    imshow(corners_window, x_derivative);
//    waitKey(0);

    Sobel(src_gray, y_derivative, CV_32FC1 , 0, 1, 3, BORDER_DEFAULT);
    //imshow(corners_window, y_derivative);
    pow(x_derivative,2.0,x2_derivative);
    pow(y_derivative,2.0,y2_derivative);
    multiply(x_derivative,y_derivative,xy_derivative);
//    x2g_derivative = x2_derivative;
//    y2g_derivative = y2_derivative;
//    xyg_derivative = xy_derivative;
    GaussianBlur(x2_derivative,x2g_derivative,Size(GaussianSize,GaussianSize),2.0,0.0,BORDER_DEFAULT);
    GaussianBlur(y2_derivative,y2g_derivative,Size(GaussianSize,GaussianSize),0.0,2.0,BORDER_DEFAULT);
    GaussianBlur(xy_derivative,xyg_derivative,Size(GaussianSize,GaussianSize),2.0,2.0,BORDER_DEFAULT);
    //forth step calculating R with k=0.04
    multiply(x2g_derivative,y2g_derivative,x2y2);
    multiply(xyg_derivative,xyg_derivative,xy);
    pow((x2g_derivative + y2g_derivative),2.0,mtrace);
    dst = (x2y2 - xy) - 0.04 * mtrace;
    //normalizing result from 0 to 255
    normalize( dst, dst_norm, 0, 255, NORM_MINMAX, CV_32FC1, Mat() );
    convertScaleAbs( dst_norm, dst_norm_scaled );
    // Drawing a circle around corners
    for( int j = 0; j < src_gray.rows ; j++ )
    {
        for( int i = 0; i < src_gray.cols; i++ )
        {
            if( (int) dst_norm.at<float>(j,i) > thresh )
            {
                circle( src_gray, Point( i, j ), 8,  Scalar(255), 1, 8, 0 );
            }
        }
    }
    pow(mtrace-(x2y2-xy)*4,0.5,delta);
    lamdamin = ((x2g_derivative + y2g_derivative)-delta)*0.5;
    lamdamax = ((x2g_derivative + y2g_derivative)+delta)*0.5;
    normalize(lamdamax,normmax,0,1,NORM_MINMAX,CV_32FC1, Mat());
    normalize(lamdamin,normmin,0,1,NORM_MINMAX,CV_32FC1, Mat());
//    for( int j = 0; j < src_gray.rows ; j++ )
//    {
//        for( int i = 0; i < src_gray.cols; i++ )
//        {
////            cout << (int)normmax.at<unsigned char>(j, i) << " ";
////            cout << normmin.at<float>(j, i) << " ";
//            cout << (int)dst_norm_scaled.at<uchar>(j, i) << " ";
//
//        }
//    }
    // Showing the result
    namedWindow( corners_window, CV_WINDOW_AUTOSIZE );
    //imshow( corners_window, src_gray );
    //imshow( corners_window, normmax);
    imshow( corners_window, normmax);
//    normmax.convertTo(normmax, CV_8UC3, 255.0);
//    imwrite("/Users/Victor/Desktop/Max2.jpg",normmax);

}