//
//  mean.h
//  pro3
//
//  Created by Victor Young on 12/31/15.
//  Copyright (c) 2015 Victor Young. All rights reserved.
//

#ifndef pro3_mean_h
#define pro3_mean_h

#include "eigenface.h"
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
using namespace cv;


void test2();
void Text(Mat img, string text, int x, int y);
int query(Mat target,eigenFace ef);
Mat CropFace(Mat image, Point eye_left , Point eye_right, Point2f offset_pct, Point dest_sz);

#endif
