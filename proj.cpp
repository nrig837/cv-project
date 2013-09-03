#include "opencv2/opencv.hpp"
#include "opencv2/nonfree/features2d.hpp" 
#include <iostream>
#include <algorithm>
#include <vector>

using namespace cv;
using namespace std;

typedef unsigned int uint;

const Scalar colour[] = {Scalar(0, 255, 0), Scalar(255, 0, 0), Scalar(0, 0, 255)};

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
   int hessianThresh = 500;//200;//1000; // Larger = faster, worse matching. Smaller = slower, better matching
   // Setup some playback parameters (related to detector parameters)
   double average_fps = 0.0;
   double DESIRED_FPS = 15.0;
   int tooSlow = -1;
   SurfFeatureDetector detector(hessianThresh);
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
   namedWindow("Object Matching", CV_WINDOW_AUTOSIZE);
   Mat frame;
   double temp_start = getTime();
   int num_frames = 0;
   Mat img_out, img_matches;
   int x = 0, y = 0;
   // Hunt for object in each video frame
   while (capture.read(frame)) {
      num_frames++;
      start = getTime();
      //cvtColor(frame, frame, CV_BGR2GRAY);
      // Set size of output matrix if needed
      if (img_out.empty() && x == 0) {
         // Attach other target objects to the left side
         int widthPad = 0;
         int maxWidth = 0;
	 int maxHeight = 0;
         for (int i = 0; i < NUM_TARGETS; i++) {
	    maxWidth = max(maxWidth, target_list[i].cols);
	    maxHeight = max(maxHeight, target_list[i].rows);
	    cout << "maxWidth: " << maxWidth << endl;
         } 
	 // Add a little extra for safety
	 widthPad += maxWidth * NUM_TARGETS;
         // Create expanded matrix to store objects on the side
         img_out.create(max(frame.rows, maxHeight), frame.cols + widthPad, 
			frame.type());
         for (int i = 1; i < NUM_TARGETS; i++) {
	    if (y + target_list[i].rows > img_out.rows) {
               x += maxWidth;
	       y = 0;
	    }
	    cout << "Placing " << i << "th image at x: " << x << endl;
   	    Mat displayRegion(img_out, Rect(x, y, target_list[i].cols, 
   				            target_list[i].rows));
	    y += target_list[i].rows;
            target_list[i].copyTo(displayRegion);
         }
	 // Messy
	 x += target_list[NUM_TARGETS-1].cols; //widthPad - target_list[0].cols;
	 cout << "rest will start from x: " << x << endl;
      }
      // Detect keypoints using SURF descriptor
      // (adapt hessian threshold based on running average of frame rate)
      double cur_time = getTime() - temp_start;
      average_fps = (double) num_frames / (cur_time / 1000.0);
      bool stable = (num_frames > 15);
      if (stable && tooSlow == -1) {
         tooSlow = (average_fps < DESIRED_FPS);
	 cout << "tooSlow set to: " << tooSlow << endl;
      }

      bool recompute = false;
      int minHessian = 50, maxHessian = 1000;
      cout << "tooSlow: " << tooSlow << " average_fps: " << average_fps << endl;
      if (stable && tooSlow == 1 && !(num_frames % 10) && 
          average_fps < DESIRED_FPS && 
	  minHessian < hessianThresh &&
	  maxHessian > hessianThresh) {
	 cout << "Increasing threshold (speeding up)..." << endl;
         hessianThresh += 50;
	 tooSlow = 1; 
	 recompute = true;
      } else if (stable && !tooSlow && !(num_frames % 10) && 
		 average_fps > DESIRED_FPS &&
		 minHessian < hessianThresh &&
		 maxHessian > hessianThresh) {
	 cout << "Decreasing threshold (slowing down)..." << endl;
	 hessianThresh -= 50;
	 tooSlow = 0; 
	 recompute = true;
      } else {
	 recompute = false;
      }
      if (recompute) {
	 cout << "Recomputing: " << hessianThresh << endl;
         SurfFeatureDetector detector_new(hessianThresh);
	 detector = detector_new;
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
      }
      frame_keypoints.clear();
      detector.detect(frame, frame_keypoints);
      
      // Calculate descriptors (based off keypoints)
      extractor.compute(frame, frame_keypoints, frame_descriptors);
      
      // Match descriptors (using FLANN matcher)
      // Ensure enough descriptors for matching 
      // (rough, could be refactored)
      if (frame_descriptors.rows <= 1) {
	 cout << "frame descriptor list too small nigga!!!" << endl;
         continue;     
      }
      bool enough_descriptors = true;
      for (uint i = 0; i < NUM_TARGETS; i++)
         if (object_descriptors_list[i].rows <= 1)
            enough_descriptors = false;
      if (!enough_descriptors)
         continue;
      
      // TEMP: output for debugging only
      cout << object_descriptors_list[0].size() << endl;
      cout << frame_descriptors.size() << endl;
      matches.clear(); //Delete previous matches
      for (uint i = 0; i < NUM_TARGETS; ++i) {
         vector< vector<DMatch> > m;
	 //vector<DMatch> match_vec;
         matcher.knnMatch(object_descriptors_list[i], frame_descriptors, m, 2);
         matches.push_back(m);
      }
      cout << "Made it through matching nigga" << endl;

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

      
      // Only draws matches for first image, the rest just get lines
      drawMatches(target_list[0], object_keypoints_list[0], frame, frame_keypoints,
                  good_match_list[0], img_matches, Scalar::all(-1), Scalar::all(-1),
                  vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

      for(uint obj = 0; obj < NUM_TARGETS; ++obj) {
         vector<DMatch>& good_matches = good_match_list[obj];
         vector<KeyPoint>& object_keypoints = object_keypoints_list[obj];
         Mat& target_object = target_list[obj];

         if (good_matches.size() < 8) {
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
            float xOffset = static_cast<float>(target_list[0].cols);
            line(img_matches, frame_corners[0] + Point2f(xOffset, 0), 
                  frame_corners[1] + Point2f(xOffset, 0), colour[obj % 3], 3);
            line(img_matches, frame_corners[1] + Point2f(xOffset, 0), 
                  frame_corners[2] + Point2f(xOffset, 0), colour[obj % 3], 3);
            line(img_matches, frame_corners[2] + Point2f(xOffset, 0), 
                  frame_corners[3] + Point2f(xOffset, 0), colour[obj % 3], 3);
            line(img_matches, frame_corners[3] + Point2f(xOffset, 0), 
                  frame_corners[0] + Point2f(xOffset, 0), colour[obj % 3], 3);
         }
      }
      // Add frame timer to image
      putText(img_matches, format("%0.0fms",getTime()-start),Point(frame.cols-60,20), FONT_HERSHEY_PLAIN, 1.0, CV_RGB(0,255,0), 1);
      cout << "x: " << x << " img_matches.cols: " << img_matches.cols << endl;
      cout << "img_out cols: " << img_out.cols << endl;
      Mat displayRest(img_out, Rect(x, 0, img_matches.cols, 
			            img_matches.rows));
      img_matches.copyTo(displayRest);

      imshow("Object Matching", img_out);
      if(waitKey(1) >= 0) break;
   }

   // TEMP: for testing, remove later
   double temp_end = getTime() - temp_start;
   cout << "Entire video took: " << temp_end << " ms " << endl;
   cout << "There were: " << num_frames << " frames" << endl;
   cout << "Average fps: " << (double) num_frames / (temp_end / 1000.0) << endl;
   
   return 0;
}

// MAC/LINUX
//#if defined TARGET_OS_MAC || defined __linux__
//#include <sys/time.h>
double getTime() {
    double t = (double) getTickCount();
    return (1000.0 * t) / getTickFrequency(); 
}

/*// WINDOWS
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
#endif*/
