#include "opencv2/opencv.hpp"
#include "opencv2/nonfree/features2d.hpp" 
#include <iostream>
#include <vector>

using namespace cv;
using namespace std;

typedef unsigned int uint;

// Get system time in ms
double getTime();

int main(int argc, char **argv) {
   if (argc < 3) {
      cout << "Incorrect number of arguments." << endl;
      cout << "usage: ./" << argv[0] << " video.type target_object1.type [target_object2.type ...]" << endl;
      return -1;
   } 

   // Read in target object and video file
   vector<Mat> target_list;
   for(int i = 2; i < argc; ++i)
      target_list.push_back(imread(argv[i], CV_LOAD_IMAGE_COLOR));
   const uint NUM_TARGETS = target_list.size();
   cout << "Read " << NUM_TARGETS << " images" << endl;

   const string video_file = argv[1];
   VideoCapture capture(video_file);
   if (!capture.isOpened()) {
      cout << "Unable to open video file." << endl;
      return -1;
   }

   // Setup SURF detector and precompute ketpoints for objects
   int minHessian = 1000; // Larger = faster, worse matching. Smaller = slower, better matching
   SurfFeatureDetector detector(minHessian);
   vector<KeyPoint> frame_keypoints;
   vector< vector<KeyPoint> > object_keypoints_list;
   for(uint i = 0; i < NUM_TARGETS; ++i) {
      vector<KeyPoint> kp;
      detector.detect(target_list[i], kp);
      object_keypoints_list.push_back(kp);
   }

   // Setup decriptors and precompute for objects
   SurfDescriptorExtractor extractor;
   Mat frame_descriptors;
   vector<Mat> object_descriptors_list(NUM_TARGETS);
   for(uint i = 0; i < NUM_TARGETS; ++i)
      extractor.compute(target_list[i], object_keypoints_list[i], object_descriptors_list[i]);

   // Setup FLANN Matcher
   FlannBasedMatcher matcher;
   vector< vector< vector<DMatch> > > matches;

   // Misc setup
   double start;
   namedWindow("Object Matching", 1);
   Mat frame;

   // Hunt for object in each video frame
   while (capture.read(frame)) {
      start = getTime();
      //cvtColor(frame, frame, CV_BGR2GRAY);

      // Detect keypoints using SURF descriptor
      frame_keypoints.clear();
      detector.detect(frame, frame_keypoints);
      
      // Calculate descriptors (based off keypoints)
      extractor.compute(frame, frame_keypoints, frame_descriptors);
      
      // Match descriptors (using FLANN matcher)
      matches.clear(); //Delete previous matches
      for(uint i = 0; i < NUM_TARGETS; ++i) {
         vector< vector<DMatch> > m;
         matcher.knnMatch(object_descriptors_list[i], frame_descriptors, m, 2);
         matches.push_back(m);
      }

      // Draw only the "good" matches
      vector< vector<DMatch> > good_match_list;
      for(uint i = 0; i < NUM_TARGETS; ++i) {
         vector< vector<DMatch> >& match = matches[i];
         vector<DMatch> gm_tmp;
         for(uint k = 0; k < match.size(); k++) {
            // matches[k].size() == 2 is rather strict. 0.7 is a good compromise between bad matches and no matches
            if(match[k].size() == 2 && match[k][0].distance < 0.7*(match[k][1].distance))
               gm_tmp.push_back(match[k][0]);
         }
         good_match_list.push_back(gm_tmp);
      }

      Mat img_matches;
      
      // Only draws matches for first image, the rest just get lines
      drawMatches(target_list[0], object_keypoints_list[0], frame, frame_keypoints,
                  good_match_list[0], img_matches, Scalar::all(-1), Scalar::all(-1),
                  vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

      for(uint obj = 0; obj < NUM_TARGETS; ++obj) {
         vector<DMatch>& good_matches = good_match_list[obj];
         vector<KeyPoint>& object_keypoints = object_keypoints_list[obj];
         Mat& target_object = target_list[obj];

         if (good_matches.size() < 4) {
            cout << obj << ": " << good_matches.size() << " good matches, skipping image" << endl;
         } else {
            cout << obj << ": " << good_matches.size() << " good matches" << endl;

            // Actually find object in frame (work out pose + location)
            vector<Point2f> object_points, frame_points;
            // Only take keypoints from the "good" matches
            for (uint i = 0; i < good_matches.size(); i++) {
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
                  frame_corners[1] + Point2f(target_object.cols, 0), Scalar(0, 255, 0), 3);
            line(img_matches, frame_corners[1] + Point2f(target_object.cols, 0), 
                  frame_corners[2] + Point2f(target_object.cols, 0), Scalar(0, 255, 0), 3);
            line(img_matches, frame_corners[2] + Point2f(target_object.cols, 0), 
                  frame_corners[3] + Point2f(target_object.cols, 0), Scalar(0, 255, 0), 3);
            line(img_matches, frame_corners[3] + Point2f(target_object.cols, 0), 
                  frame_corners[0] + Point2f(target_object.cols, 0), Scalar(0, 255, 0), 3);
         }
      }
      // Add frame timer to image
      putText(img_matches, format("%0.0fms",getTime()-start),Point(frame.cols-60,20), FONT_HERSHEY_PLAIN, 1.0, CV_RGB(0,255,0), 1);

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
