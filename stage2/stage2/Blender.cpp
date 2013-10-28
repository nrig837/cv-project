//
//  main.cpp
//  stage2
//
//  Created by Nelson Rigby on 20/10/13.
//  Copyright (c) 2013 Nelson Rigby. All rights reserved.
//

#include "opencv2/opencv.hpp"
#include "opencv2/nonfree/features2d.hpp"
#include "Blender.h"
#include <iostream>
#include <sstream>
#include <sys/stat.h>
//#include <unistd.h>
//#include <dirent.h>
#include <algorithm>
#include <math.h>
#include <vector>


using namespace cv;
using namespace std;

Blender::Blender() {
    // A better design would initialise stuff here
    // rather than just using Blender::getBlended()
}

void Blender::getBlended(const Mat& orig_overlay, const Mat& target, Mat& blendedImage) {
	// TEMP: resize overlay to 3/4 size of target so it will always fit
	Mat overlay;
	resize(orig_overlay, overlay, Size(3*target.cols/4, 3*target.rows/4));

    // TEMP: initial value range
    // Programmatically specify mask image
    Mat_<float> mask(target.rows, target.cols, 1.0);
    // TODO: assert overlay is small enough to center in target
    // Center overlay in image
    // First, expand overlay image to same size as target
    int top = (target.rows - overlay.rows) / 2;
    int bottom = top;
    int left = (target.cols - overlay.cols) / 2;
    int right = left;
    Mat overlay_padded;
    copyMakeBorder(overlay, overlay_padded, top, bottom, left, right, BORDER_REPLICATE);
    // Explicitly resize because of integer truncation in border calculations
    resize(overlay_padded, overlay_padded, target.size());
    // Set mask around centered overlay
    mask(Range(top, top + overlay.rows), Range(left, left + overlay.cols)) = 0.0;
    // Mask background of overlay
    maskBackground(overlay_padded, mask);
    
    // TEMP: test generating laplacian pyramid of target image, see what it looks like
    // Get target laplacian pyramid
    vector<Mat_<Vec3f> > targetLapPyr;
    int numLevels = 5;
    getLaplacianPyramid(target, targetLapPyr, numLevels);
    // Get overlay laplacian pyramid
    vector<Mat_<Vec3f> > overlayLapPyr;
    getLaplacianPyramid(overlay_padded, overlayLapPyr, numLevels);
    
    // TEMP: test generating gaussian pyramid of mask, see what it looks like
    vector<Mat_<Vec3f> > gaussianPyramidMask;
    getGaussianPyramid(mask, gaussianPyramidMask, numLevels);
    
    // Blend target and overlay images
    Mat_<Vec3f> blended;
    //blendedImage.convertTo(blended, blended.type());
    getBlended(targetLapPyr, overlayLapPyr, gaussianPyramidMask,
               blended, numLevels);
    blendedImage = blended;
    // Normalize for display
    blendedImage.convertTo(blendedImage, CV_8UC3);
    //normalize(blendedImage, blendedImage, 0, 1, NORM_MINMAX, CV_32FC1);
}

void Blender::maskBackground(Mat& overlay_padded, Mat& mask) {
    // For now, just mask all pixels that are white in overlay
    // i.e. set pixels that are white in overlay to white in mask
    // Note: we may want to look at a segmentation approach to split
    // into background / foreground
    // Check dimensions of matrices match
    overlay_padded.convertTo(overlay_padded, CV_32FC3);
    assert(overlay_padded.size() == mask.size());
    assert(overlay_padded.channels() == mask.channels() * 3);
    float *overlay_ptr;
    float *mask_ptr;
    int rows = overlay_padded.rows;
    int cols = overlay_padded.cols; //* overlay_padded.channels();
    const float WHITE = 255;
    for (int i = 0; i < rows; i++) {
        overlay_ptr = overlay_padded.ptr<float>(i);
        mask_ptr = mask.ptr<float>(i);
        for (int j = 0; j < cols; j++) {
            if (overlay_ptr[j*3] == WHITE &&
                overlay_ptr[j*3 + 1] == WHITE &&
                overlay_ptr[j*3 + 2] == WHITE) {
                cout << "Masking row, col from overlay: " << i << ", " << j << endl;
                mask_ptr[j] = 1.0;
            }
        }
    }
    
}

void Blender::getLaplacianPyramid(const Mat_<Vec3f>& image, vector<Mat_<Vec3f> >& laplacianPyramid,
                         int numLevels) {
    //laplacianPyramid.push_back(image);
    // Construct each level of the pyramid
    Mat_<Vec3f> curImage;
    //image.convertTo(curImage, CV_32FC3);
    curImage = image;
    for (int level = 0; level < numLevels; level++) {
        if (level < numLevels - 1) {
            Mat_<Vec3f> newLevel;
            Mat_<Vec3f> down_tmp, up_tmp;
            pyrDown(curImage, down_tmp);
            pyrUp(down_tmp, up_tmp, curImage.size());
            newLevel = curImage - up_tmp;
            laplacianPyramid.push_back(newLevel);
            curImage = down_tmp;
        } else {
            // Top level of laplacian pyramid is just a Gaussian-blurred
            // image (not difference of gaussian/laplacian)
            laplacianPyramid.push_back(curImage);
        }
    }
}


void Blender::getGaussianPyramid(const Mat_<float>& mask, vector<Mat_<Vec3f> >& gaussianPyramidMask,
                        int numLevels) {
    Mat curMask;
    cvtColor(mask, curMask, CV_GRAY2BGR);
    gaussianPyramidMask.push_back(curMask);
    // Construct each level of the pyramid
    curMask = mask;
    for (int level = 0; level < numLevels; level++) {
        Mat newLevel_gray;
        Mat_<Vec3f> newLevel_BGR;
        pyrDown(curMask, newLevel_gray);
        cvtColor(newLevel_gray, newLevel_BGR, CV_GRAY2BGR);
        gaussianPyramidMask.push_back(newLevel_BGR);
        curMask = newLevel_gray;
    }
}

// Blend each level of laplacian pyramids in accordance with Gaussian mask
void Blender::getBlended(const vector<Mat_<Vec3f> >& targetLapPyr, const vector<Mat_<Vec3f> >& overlayLapPyr,
                const vector<Mat_<Vec3f> >& mask, Mat_<Vec3f>& blendedImage,
                int numLevels) {
    // Blend each level of the laplacian pyramids
    vector<Mat_<Vec3f> > blendedLapPyr;
    for (int level = 0; level < numLevels; level++) {
        // Weight target pixels by mask values
        Mat_<Vec3f> targetMasked = targetLapPyr[level].mul(mask[level]);
        // Calculate inverse mask to use with overlay level
        Mat_<Vec3f> inverseMask = Scalar(1.0, 1.0, 1.0) - mask[level];
        // Weight overlay pixels by inverse mask values
        Mat_<Vec3f> overlayMasked = overlayLapPyr[level].mul(inverseMask);
        // Blend two images: blended = alpha(target_pixel_i) + (1-alpha)(overlay_pixel_i), for all i
        // alpha is weight given by mask pixels
        Mat_<Vec3f> blended = targetMasked + overlayMasked;
        blendedLapPyr.push_back(blended);
    }
    
    // Reconstruct blended image from blended pyramid
    Mat_<Vec3f> currentImage = blendedLapPyr[numLevels-1];
    for (int level = numLevels - 2; level >= 0; level--) {
        // Make sizes of currentImage and the level below the same
        // (by upsampling currentImage)
        Mat_<Vec3f> upsampled;
        pyrUp(currentImage, upsampled, blendedLapPyr[level].size());
        // Image reconstruction simply involves summing different levels
        currentImage = upsampled + blendedLapPyr[level];
    }
    blendedImage = currentImage;
}
