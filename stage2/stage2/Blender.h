//
//  File.h
//  stage2
//
//  Created by Nelson Rigby on 21/10/13.
//  Copyright (c) 2013 Nelson Rigby. All rights reserved.
//

#ifndef __stage2__Blender__
#define __stage2__Blender__

#include "opencv2/opencv.hpp"
#include "opencv2/nonfree/features2d.hpp"
#include <iostream>
#include <sstream>
#include <sys/stat.h>
#include <unistd.h>
#include <dirent.h>
#include <algorithm>
#include <math.h>
#include <vector>

using namespace cv;
using namespace std;

class Blender {
public:
    Blender();
    void getBlended(const Mat& overlay, const Mat& target, Mat& blendedImage);
private:
    void maskBackground(Mat& overlay_padded, Mat& mask);
    void getLaplacianPyramid(const Mat_<Vec3f>& image,
                             vector<Mat_<Vec3f> >& laplacianPyramid,
                             int numLevels);
    void getGaussianPyramid(const Mat_<float>& mask,
                            vector<Mat_<Vec3f> >& gaussianPyramidMask,
                            int numLevels);
    void getBlended(const vector<Mat_<Vec3f> >& targetLapPyr,
                    const vector<Mat_<Vec3f> >& overlayLapPyr,
                    const vector<Mat_<Vec3f> >& mask, Mat_<Vec3f>& blendedImage,
                    int numLevels);
};

#endif /* defined(__stage2__Blender__) */
