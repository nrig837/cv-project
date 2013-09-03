#include "opencv2/opencv.hpp"
#include "opencv2/nonfree/features2d.hpp" 
#include <iostream>
#include <sys/stat.h>
#include <unistd.h>
#include <dirent.h>
#include <algorithm>
#include <math.h>
#include <vector>

using namespace cv;
using namespace std;

typedef unsigned int uint;

const Scalar colour[] = {Scalar(0, 255, 0), Scalar(255, 0, 0), Scalar(0, 0, 255)};

// Read frames from video into matrix vector (for unified interface
// if we want to be able to slurp up an image sequence as well as a video file)
void readFrames(VideoCapture& vc, vector<Mat>& frames_list);

// Get system time in ms
double getTime();
// Check whether string is all numeric 
bool isNumeric(char* str);
// Return euclidean distance between two points
int dist(Point2f p1, Point2f p2);

int main(int argc, char **argv) {
   if (argc < 3) {
      cout << "Incorrect number of arguments." << endl;
      cout << "usage: ./" << argv[0] << 
	      " video.type target_object1.type [target_object2.type ...] [noise] [hessianThreshold] [fps]" << endl;
      return -1;
   }

   // Read in parameters for SURF and playback (if specified)
   // Note: this is totally dodgy but its like 2.00 am 
   int hessianThresh = 500;//200;//1000; // Larger = faster, worse matching. Smaller = slower, better matching
   // Setup some playback parameters (related to detector parameters)
   double average_fps = 0.0;
   double DESIRED_FPS = 15.0;
   bool noiseSuppress = false;
   if (isNumeric(argv[argc-1]) && isNumeric(argv[argc-2])) {
      // Assume if last two arguments end in digits, they are not image files
      hessianThresh = (double) atoi(argv[argc-2]);
      DESIRED_FPS = (double) atoi(argv[argc-1]);
      // Also assume that if these are specified, noise suppression option is too
      noiseSuppress = (argv[argc-3][0] == 'y') ? true : false;
      cout << "hessianThresh: " << hessianThresh << endl;
      cout << "DESIRED_FPS: " << DESIRED_FPS << endl;
      argc -= 3;
   } 

   // Read in target object and video file
   vector<Mat> target_list;
   for(int i = 2; i < argc; ++i)
      target_list.push_back(imread(argv[i], CV_LOAD_IMAGE_COLOR));
   const uint NUM_TARGETS = target_list.size();
   cout << "Read " << NUM_TARGETS << " images" << endl;
   // If directory, assume it contains sequence of image frames
   struct stat buf;
   stat(argv[1], &buf);
   vector<Mat> frames_list;
   VideoCapture capture;
   if (S_ISDIR(buf.st_mode)) {
      cout << "FOUND A DIRECTORY" << endl;
      DIR* dir;
      struct dirent* ent;
      if ((dir = opendir(argv[1])) != NULL) {
         // Read all files as video frames
	 while ((ent = readdir(dir)) != NULL) {
	    Mat frame;
	    if (ent->d_name[0] == '.')
	       continue;
	    string img_file = string(argv[1])+"/"+string(ent->d_name);
	    cout << "img_file: " << img_file << endl;
            frame = imread(img_file, CV_LOAD_IMAGE_COLOR);
	    if (!frame.data) {
               cout << "Problem reading image frame." << endl;
	       return -1;
	    }
	    frames_list.push_back(frame);
	 }
      } else {
	 cout << "Unable to open directory" << endl;
	 return -1;
      }	 
   } else {
      // If video file, then read that
      const string video_file = argv[1];
      capture.open(video_file);
      if (!capture.isOpened()) {
         cout << "Unable to open video file." << endl;
         return -1;
      }
      cout << "TRYING TO READ FRAMES.." << endl;
      readFrames(capture, frames_list);
      cout << "READ FRAMES" << endl;
   }
   vector<Point2f> dummy;
   vector< vector<Point2f> > prev_frame_corners(NUM_TARGETS, dummy);
   vector<int> stable_area;
   for (int i = 0; i < NUM_TARGETS; i++)
      stable_area.push_back(target_list[i].rows * target_list[i].cols);
   // Setup SURF detector and precompute ketpoints for objects
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
   for (int f = 0; f < frames_list.size(); f++) {//while(capture.read(frame)) {
      frame = frames_list[f];
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
	 if (NUM_TARGETS > 1)
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

	    // Counter-act frame noise by considering past frame(s) 
	    // Ignore sudden lurchy mis-detection due to noise
	    if (prev_frame_corners[obj].empty()) {
	       for (int i = 0; i < frame_corners.size(); i++)
                  prev_frame_corners[obj].push_back(frame_corners[i]);
	    }
	    const int NOISE_THRESH = 50;
	    bool ignore_frame = false;
	    for (int c = 0; c < 4; c++) {
               if (abs(prev_frame_corners[obj][c].x - frame_corners[c].x) > NOISE_THRESH ||
	           abs(prev_frame_corners[obj][c].y - frame_corners[c].y) > NOISE_THRESH) {
		  if (num_frames % 10)
		     ignore_frame = true;
	       }
	    }
	    // Consider area of match relative to most recent, stable area of object
	    // Assume trapezoidal for now (should use general form for area of a quadrilateral)
	    float a = (float) dist(frame_corners[0], frame_corners[1]);
	    float b = (float) dist(frame_corners[1], frame_corners[2]);
	    float c = (float) dist(frame_corners[2], frame_corners[3]);
	    float d = (float) dist(frame_corners[3], frame_corners[0]);
	    //int A = asin(
            //double sinA = sin(A);
	    int h = abs(frame_corners[0].y - frame_corners[3].y);
	    int match_area = (a+b)/2 * h;
	    // Below code unsafe division
            //float a_b = (float) a / (float) b;
	    //float b_c = (float) b / (float) c;
            //float prev_a = (float) dist(prev_frame_corners[obj][0], prev_frame_corners[obj][1]);
	    //float prev_b = (float) dist(prev_frame_corners[obj][1], prev_frame_corners[obj][2]);
            //float prev_c = (float) dist(prev_frame_corners[obj][2], prev_frame_corners[obj][3]);
	    //float prev_d = (float) dist(prev_frame_corners[obj][3], prev_frame_corners[obj][0]);
	    //h = abs(prev_frame_corners[obj][0].y - frame_corners[3].y);
	    //int prev_match_area = (prev_a + prev_b)/2 * h;
	    // TEMP: try ratio of side lengths 
	   // float prev_a_b = (float) prev_a / (float) prev_b;
	    //float prev_b_c = (float) prev_b / (float) prev_c;
	    //float prev_c_d = (float) prev_c / (float) prev_d; 
            // If new match actually looks "better", update
	    // (we don't want to get stuck comparing against a dodgy match)
	    /*if (prev_match_area >= 5 &&
	        //((float) match_area / (float) stable_area[obj] < 0.5 || 
		// (float) match_area / (float) stable_area[obj] > 1.5)) {
		//ignore_frame = true; 
	        abs(stable_area[obj] - match_area) < 
	        abs(stable_area[obj] - prev_match_area)) {
                //ignore_frame = false;
	    
	       const float SHAPE_THRESH = 0.5;
	       if ((prev_a != 0 && prev_c != 0 && prev_d != 0) &&
	           (a / prev_a < 1.0 - SHAPE_THRESH || a / prev_a > 1.0 + SHAPE_THRESH ||
	   	   c / prev_c < 1.0 - SHAPE_THRESH || c / prev_c > 1.0 + SHAPE_THRESH || 	   
	           d / prev_d < 1.0 - SHAPE_THRESH || d / prev_d > 1.0 + SHAPE_THRESH)) {
                  //ignore_frame = true;
	       }
	    }*/
            //const float AREA_THRESH = 0.2;
	    //if (stable_area[obj] != 0 && 
	    //    ((float) match_area / (float) stable_area[obj] > 1.0 - AREA_THRESH || 
		//(float) match_area / (float) stable_area[obj] < 1.0 + AREA_THRESH)) {
               if (noiseSuppress) {
   	          if (!ignore_frame) {
                     prev_frame_corners[obj] = frame_corners;
   	             //stable_area[obj] = match_area;
   	          } else {
   	             frame_corners = prev_frame_corners[obj];
   	          }
	       }
               
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
	    //}
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

// Read video frames in
void readFrames(VideoCapture& vc, vector<Mat>& frames_list) {
   Mat frame;
   while(vc.read(frame)) {
      frames_list.push_back(frame.clone());
   }   
}

// MAC/LINUX
//#if defined TARGET_OS_MAC || defined __linux__
//#include <sys/time.h>
double getTime() {
    double t = (double) getTickCount();
    return (1000.0 * t) / getTickFrequency(); 
}

bool isNumeric(char* str) {
   for (int i = 0; str[i] != '\0'; i++)
      if (!isdigit(str[i]))
         return false;
   return true;
}

int dist(Point2f p1, Point2f p2) {
   int x_dist = abs(p1.x - p2.x);
   int y_dist = abs(p2.y - p2.y);
   return (int) sqrt(std::pow((double) x_dist, 2) + std::pow((double) y_dist, 2));
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
