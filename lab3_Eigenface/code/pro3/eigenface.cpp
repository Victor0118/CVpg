//
//  eigenface.cpp
//  pro3
//
//  Created by Victor Young on 12/30/15.
//  Copyright (c) 2015 Victor Young. All rights reserved.
//


#include "eigenface.h"

using namespace cv;
using namespace std;

void eigenFace::saveFile(string file){
    FileStorage fs;
    fs.open(file, FileStorage::WRITE);
    fs << "eigenvectors" << eigenvectors;
    fs << "weights" << weights;
    fs.release();                                       // explicit close
    cout << "Write Done." << endl;
}

void eigenFace::loadFile(string file){
    FileStorage fs;
    fs.open(file, FileStorage::READ);
    fs["eigenvectors"] >> eigenvectors ;
    fs["weights"] >> weights;
    fs.release();                                       // explicit close
    cout << "Read Done." << endl;
}

Vector<Mat> eigenFace::getdb(){
    return db;
}
Mat eigenFace::getweights(){
    return weights;
}
int eigenFace::getdbsize(){
    return (int)weights.rows;
}
Mat eigenFace::geteigenvectors(){
    return eigenvectors;
}

void eigenFace::addData(Mat image){
    
    db.push_back(image);

}
// Normalizes a given image into a value range between 0 and 255.
Mat eigenFace::norm_0_255(const Mat& src) {
    // Create and return normalized image:
    Mat dst;
    switch(src.channels()) {
        case 1:
            cv::normalize(src, dst, 0, 255, NORM_MINMAX, CV_8UC1);
            break;
        case 3:
            cv::normalize(src, dst, 0, 255, NORM_MINMAX, CV_8UC3);
            break;
        default:
            src.copyTo(dst);
            break;
    }
    return dst;
}

// Converts the images given in src into a row matrix.
Mat eigenFace::asRowMatrix(const vector<Mat>& src, int rtype, double alpha = 1, double beta = 0) {
    // Number of samples:
    size_t n = src.size();
    // Return empty matrix if no matrices given:
    if(n == 0)
        return Mat();
    // dimensionality of (reshaped) samples
    size_t d = src[0].total();
    // Create resulting data matrix:
    Mat data(n, d, rtype);
    // Now copy data:
    for(int i = 0; i < n; i++) {
        //
        if(src[i].empty()) {
            string error_message = format("Image number %d was empty, please check your input data.", i);
            CV_Error(CV_StsBadArg, error_message);
        }
        // Make sure data can be reshaped, throw a meaningful exception if not!
        if(src[i].total() != d) {
            string error_message = format("Wrong number of elements in matrix #%d! Expected %d was %d.", i, d, src[i].total());
            CV_Error(CV_StsBadArg, error_message);
        }
        // Get a hold of the current row:
        Mat xi = data.row(i);
        // Make reshape happy by cloning for non-continuous matrices:
        if(src[i].isContinuous()) {
            src[i].reshape(1, 1).convertTo(xi, rtype, alpha, beta);
        } else {
            src[i].clone().reshape(1, 1).convertTo(xi, rtype, alpha, beta);
        }
    }
    return data;
}

int eigenFace::train() {
    // Holds some images:
//    vector<Mat> db;
    
    // Load the greyscale images. The images in the example are
    // taken from the AT&T Facedatabase, which is publicly available
    // at:
    //
    //      http://www.cl.cam.ac.uk/research/dtg/attarchive/facedatabase.html
    //
    // This is the path to where I stored the images, yours is different!
    //
//    string prefix = "/Volumes/mac/Mac Download/att_faces/";
//    ;
//    Mat face = imread(prefix + "s1/1.pgm", IMREAD_GRAYSCALE);
//    equalizeHist(face,face);
//    
//    db.push_back(imread(prefix + "s1/1.pgm", IMREAD_GRAYSCALE));
//    db.push_back(imread(prefix + "s1/2.pgm", IMREAD_GRAYSCALE));
//    db.push_back(imread(prefix + "s1/3.pgm", IMREAD_GRAYSCALE));
//    db.push_back(imread(prefix + "s1/4.pgm", IMREAD_GRAYSCALE));
//    db.push_back(imread(prefix + "s1/5.pgm", IMREAD_GRAYSCALE));
//    db.push_back(imread(prefix + "s1/6.pgm", IMREAD_GRAYSCALE));
//    db.push_back(imread(prefix + "s1/7.pgm", IMREAD_GRAYSCALE));
//    db.push_back(imread(prefix + "s1/8.pgm", IMREAD_GRAYSCALE));
//    //    db.push_back(imread(prefix + "s1/9.pgm", IMREAD_GRAYSCALE));
//    //    db.push_back(imread(prefix + "s1/10.pgm", IMREAD_GRAYSCALE));
//    
//    db.push_back(imread(prefix + "s2/1.pgm", IMREAD_GRAYSCALE));
//    db.push_back(imread(prefix + "s2/2.pgm", IMREAD_GRAYSCALE));
//    db.push_back(imread(prefix + "s2/3.pgm", IMREAD_GRAYSCALE));
//    db.push_back(imread(prefix + "s2/4.pgm", IMREAD_GRAYSCALE));
//    db.push_back(imread(prefix + "s2/5.pgm", IMREAD_GRAYSCALE));
//    db.push_back(imread(prefix + "s2/6.pgm", IMREAD_GRAYSCALE));
//    db.push_back(imread(prefix + "s2/7.pgm", IMREAD_GRAYSCALE));
//    db.push_back(imread(prefix + "s2/8.pgm", IMREAD_GRAYSCALE));
//    //    db.push_back(imread(prefix + "s2/9.pgm", IMREAD_GRAYSCALE));
//    //    db.push_back(imread(prefix + "s2/10.pgm", IMREAD_GRAYSCALE));
//    //
//    db.push_back(imread(prefix + "s3/1.pgm", IMREAD_GRAYSCALE));
//    db.push_back(imread(prefix + "s3/2.pgm", IMREAD_GRAYSCALE));
//    db.push_back(imread(prefix + "s3/3.pgm", IMREAD_GRAYSCALE));
//    db.push_back(imread(prefix + "s3/4.pgm", IMREAD_GRAYSCALE));
//    db.push_back(imread(prefix + "s3/5.pgm", IMREAD_GRAYSCALE));
//    db.push_back(imread(prefix + "s3/6.pgm", IMREAD_GRAYSCALE));
//    db.push_back(imread(prefix + "s3/7.pgm", IMREAD_GRAYSCALE));
//    db.push_back(imread(prefix + "s3/8.pgm", IMREAD_GRAYSCALE));
//    //    db.push_back(imread(prefix + "s2/9.pgm", IMREAD_GRAYSCALE));
    //    db.push_back(imread(prefix + "s2/10.pgm", IMREAD_GRAYSCALE));
    //
    //    db.push_back(imread(prefix + "s4/1.pgm", IMREAD_GRAYSCALE));
    //    db.push_back(imread(prefix + "s4/2.pgm", IMREAD_GRAYSCALE));
    //    db.push_back(imread(prefix + "s4/3.pgm", IMREAD_GRAYSCALE));
    
    
    // Build a matrix with the observations in row:
    Mat data = asRowMatrix(db, CV_32FC1);
    
    Mat meanface = Mat::zeros(1, data.size[1], CV_32FC1);
    for (int i = 0; i<data.size[0];i++ ) {
        meanface += data.row(i);
    }
    meanface = meanface * 1.0/data.size[0];
//    imshow("testt", norm_0_255(meanface.reshape(1, db[0].rows)));
//    waitKey();
    Mat A = data - repeat(meanface, data.size[0],1);
    
    
    SVD svd(A*A.t(),SVD::FULL_UV);
    eigenvectors = svd.vt;
    eigenvectors = (A.t()*eigenvectors).t();
    double currentEnergy = 0;
    for (int i=0; i<eigenvectors.rows; i++) {
        currentEnergy += eigenvectors.at<float>(i,i);
    }
    double sumenergy = currentEnergy;
    currentEnergy = 0;
    int i;
    for (i=0; i<eigenvectors.rows; i++) {
        currentEnergy += eigenvectors.at<float>(i,i);
        if (currentEnergy/sumenergy > ENERGYPERCENTAGE) {
            cout << "now:" << i <<" total:" <<eigenvectors.rows<<endl;
            break;
        }
    }
    
    eigenvectors = eigenvectors.rowRange(0, i);
    //    eigen(A*A.t(), eigenvalues, eigenvectors);
    //    eigenvectors = (A.t()*eigenvectors).t();
    
    
    //    Mat mean = mean(data);
    
    // Number of components to keep for the PCA:
//    int num_components = 10;
    
    // Perform a PCA:
    //    PCA pca(data, Mat(), CV_PCA_DATA_AS_ROW, num_components);
    
    // And copy the PCA results:
    //    Mat mean = pca.mean.clone();
    //    Mat eigenvalues2 = pca.eigenvalues.clone();
    //    Mat eigenvectors2 = pca.eigenvectors.clone();
    
    
    imshow("avg", norm_0_255(meanface.reshape(1, db[0].rows)));
    // The mean face:
    //    imshow("avg", norm_0_255(meanface.reshape(1, db[0].rows)));
    
    // The first three eigenfaces:
    imshow("pc1", norm_0_255(eigenvectors.row(0)).reshape(1, db[0].rows));
    imshow("pc2", norm_0_255(eigenvectors.row(1)).reshape(1, db[0].rows));
    imshow("pc3", norm_0_255(eigenvectors.row(2)).reshape(1, db[0].rows));
    
    // Show the images:
    waitKey(0);
    
    weights = data*eigenvectors.t();
//    
    // Success!
    return 0;
}