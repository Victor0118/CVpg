//
//  eigenface.h
//  pro3
//
//  Created by Victor Young on 12/30/15.
//  Copyright (c) 2015 Victor Young. All rights reserved.
//

#ifndef __pro3__eigenface__
#define __pro3__eigenface__


#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <fstream>
#include <sstream>
#include <iostream>
using namespace cv;

#define ENERGYPERCENTAGE 0.96
class eigenFace{
    
private:
    Mat eigenvalues;
    Mat eigenvectors;
    vector<Mat> db;
    Mat weights;
    
public:
    Mat norm_0_255(const Mat& src);
    Mat asRowMatrix(const vector<Mat>& src, int rtype, double alpha, double beta);
    int train();
    void addData(Mat image);
    Mat geteigenvectors();
    int getdbsize();
    Mat getweights();
    Vector<Mat> getdb();
    void saveFile(string file);
    void loadFile(string file);
};

#endif /* defined(__pro3__eigenface__) */
