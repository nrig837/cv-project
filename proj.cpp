#include "opencv2/opencv.hpp"
#include "opencv2/nonfree/features2d.hpp" 
#include <iostream>
#include <vector>

using namespace cv;
using namespace std;

// Get system time in ms
double getTime();

int main(int argc, char **argv) {
   if (argc != 3) {
      cout << "Incorrect number of arguments." << endl;
      cout << "usage: ./" << argv[0] << " video.type target_object.type" << endl;
      return -1;
   } 

   // Read in target object and video file
   Mat target_object = imread(argv[2], CV_LOAD_IMAGE_COLOR);

   const string video_file = argv[1];
   VideoCapture capture(video_file);
   if (!capture.isOpened()) {
      cout << "Unable to open video file." << endl;
      return -1;
   }

   double start;

   // Setup SURF detector and precompute ketpoints for object
    int minHessian = 1000; // Larger = faster, worse matching. Smaller = slower, better matching
    SurfFeatureDetector detector(minHessian);
    vector<KeyPoint> object_keypoints, frame_keypoints;
    detector.detect(target_object, object_keypoints);

   // Setup decriptors and precompute for object
    SurfDescriptorExtractor extractor;
    Mat object_descriptors, frame_descriptors;
    extractor.compute(target_object, object_keypoints, object_descriptors);

    // Setup FLANN Matcher
    FlannBasedMatcher matcher;
    vector< vector<DMatch> > matches;

   // Hunt for object in each video frame
   namedWindow("Object Matching", 1);
   Mat frame;
   while (capture.read(frame)) {
      start = getTime();
      //cvtColor(frame, frame, CV_BGR2GRAY);

      // Detect keypoints using SURF descriptor
      detector.detect(frame, frame_keypoints);
      
      // Calculate descriptors (based off keypoints)
      extractor.compute(frame, frame_keypoints, frame_descriptors);
      
      // Match descriptors (using FLANN matcher)
      matches.clear(); //Delete previous matches
      matcher.knnMatch(object_descriptors, frame_descriptors, matches, 2);

      // Draw only the "good" matches
      vector<DMatch> good_matches;
      for(int k = 0; k < matches.size(); k++) {
         // matches[k].size() == 2 is rather strict. 0.7 is a good compromise between bad matches and no matches
          if(matches[k].size() == 2 && matches[k][0].distance < 0.7*(matches[k][1].distance))
              good_matches.push_back(matches[k][0]);
      }
      
      Mat img_matches;

      drawMatches(target_object, object_keypoints, frame, frame_keypoints,
                  good_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
                  vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

      if (good_matches.size() < 4) {
         cout << good_matches.size() << " good matches, skipping frame" << endl;
      } else {
         cout << good_matches.size() << " good matches" << endl;

         // Actually find object in frame (work out pose + location)
         vector<Point2f> object_points, frame_points;
         // Only take keypoints from the "good" matches
         for (int i = 0; i < good_matches.size(); i++) {
            object_points.push_back(object_keypoints[good_matches[i].queryIdx].pt);
            frame_points.push_back(frame_keypoints[good_matches[i].trainIdx].pt);
         }
         // Compute homography matrix, which gives the mapping of the query
         // (planar) object to the frame (i.e. coordinate system mapping)
         Mat H = findHomography(object_points, frame_points, CV_RANSAC);
         // Get corners of the query object (object to be detected)
         vector<Point2f> object_corners(4);
         object_corners[0] = cvPoint(0, 0); 
         object_corners[1] = cvPoint(target_object.cols, 0);
         object_corners[2] = cvPoint(target_object.cols, target_object.rows); 
         object_corners[3] = cvPoint(0, target_object.rows);
         vector<Point2f> frame_corners(4);
         perspectiveTransform(object_corners, frame_corners, H);

         // Draw the actual lines around the detected object
         line(img_matches, frame_corners[0] + Point2f(target_object.cols, 0), 
               frame_corners[1] + Point2f(target_object.cols, 0), Scalar(0, 255, 0), 4);
         line(img_matches, frame_corners[1] + Point2f(target_object.cols, 0), 
               frame_corners[2] + Point2f(target_object.cols, 0), Scalar( 0, 255, 0), 4);
         line(img_matches, frame_corners[2] + Point2f(target_object.cols, 0), 
               frame_corners[3] + Point2f(target_object.cols, 0), Scalar( 0, 255, 0), 4);
         line(img_matches, frame_corners[3] + Point2f(target_object.cols, 0), 
               frame_corners[0] + Point2f(target_object.cols, 0), Scalar( 0, 255, 0), 4);
      }
      // Add frame timer to image
      putText(img_matches, format("%0.0fms",getTime()-start),Point(frame.cols-10,20), FONT_HERSHEY_PLAIN, 1.0, CV_RGB(0,255,0), 1);

      imshow("Object Matching", img_matches);
      if(waitKey(30) >= 0) break;
   }
   return 0;
}

// MAC/LINUX
#if defined TARGET_OS_MAC || defined __linux__
//#include <sys/time.h>
double getTime() {
/* //UNTESTED
    timeval t;
    gettimeofday(&t, NULL);
    return t.tv_usec / 1000.0;*/
    return 0.0;
}

// WINDOWS
#elif defined _WIN32 || defined _WIN64
#include <windows.h>
double getTime() {
    return timeGetTime();
}

// UNKNOWN OS
#else
double getTime() {
    return 0.0;
}
#endif
