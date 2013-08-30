#include "opencv2/opencv.hpp"
#include "opencv2/nonfree/features2d.hpp" 
#include <iostream>
#include <vector>

using namespace cv;
using namespace std;

int main(int argc, char **argv) {
   if (argc != 3) {
      cout << "Incorrect number of arguments." << endl;
      cout << "usage: ./" << argv[0] << " target_object.type video.type" << endl;
      return -1;
   } 

   // Read in target object and video file
   Mat target_object = imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE);

   const string video_file = argv[2];
   VideoCapture capture(video_file);
   if (!capture.isOpened()) {
      cout << "Unable to open video file." << endl;
      return -1;
   }

   // Hunt for object in each video frame
   namedWindow("Object Matching", 1);
   Mat frame;
   while (capture.read(frame)) {
      cvtColor(frame, frame, CV_BGR2GRAY);

      // Detect keypoints using SURF descriptor
      int minHessian = 400;
      SurfFeatureDetector detector(minHessian);
      vector<KeyPoint> object_keypoints, frame_keypoints;
      detector.detect(target_object, object_keypoints);
      detector.detect(frame, frame_keypoints);
      
      // Calculate descriptors (based off keypoints)
      SurfDescriptorExtractor extractor;
      Mat object_descriptors, frame_descriptors;
      extractor.compute(target_object, object_keypoints, object_descriptors);
      extractor.compute(frame, frame_keypoints, frame_descriptors);
      
      // Match descriptors (using FLANN matcher)
      FlannBasedMatcher matcher;
      vector<DMatch> matches;
      matcher.match(object_descriptors, frame_descriptors, matches);

      // Draw only the "good" matches (distance less than 3 * min_dist)
      // TODO: consider improvements (this is bit suspect/rough)
      double max_dist = 0;
      double min_dist = 100;
      for (int i = 0; i < object_descriptors.rows; i++) { 
         double dist = matches[i].distance;
	    if (dist < min_dist) min_dist = dist;
	    if (dist > max_dist) max_dist = dist;
      }
      vector<DMatch> good_matches;
      for (int i = 0; i < object_descriptors.rows; i++)
         if (matches[i].distance < 3 * min_dist)
	    good_matches.push_back(matches[i]);
      Mat img_matches;
      drawMatches(target_object, object_keypoints, frame, frame_keypoints,
                  good_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
                  vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

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

      imshow("Object Matching", img_matches);
      if(waitKey(30) >= 0) break;
   }
   return 0;
}

