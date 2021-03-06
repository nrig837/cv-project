#include "opencv2/opencv.hpp"
#include "opencv2/nonfree/features2d.hpp"
#include "Blender.h"
#include <iostream>
#include <sys/stat.h>
//#include <unistd.h>
//#include <dirent.h>
#include <algorithm>
#include <math.h>
#include <vector>

using namespace cv;
using namespace std;

const bool SHOW_KALMAN_GRAPH = true;

typedef unsigned int uint;

const Scalar colour[] = {Scalar(0, 255, 0), Scalar(255, 0, 0), Scalar(0, 0, 255)};

// Read frames from video into matrix vector (for unified interface
// if we want to be able to slurp up an image sequence as well as a video file)
void readFrames(VideoCapture& vc, vector<Mat>& frames_list);
void writeFrames(vector<Mat>& frames_list, double fps);

// Make a bounding Rect from an irregular set of 4 points
Rect makeBoundingBox(vector<Point2f>&);

// Make a vector of corners
vector<Point2f> makeCornerVec(Mat& src);

// returns the region defined by corners, warped to be rectangular, transform returns the transform used in the warp
Mat extractAndWarp(Mat& image, vector<Point2f>& corners);
void unWarp(Mat& image, vector<Point2f>& corners, Mat& target);

// Get system time in ms
double getTime();
// Check whether string is all numeric
bool isNumeric(char* str);
// Return euclidean distance between two points
int dist(Point2f p1, Point2f p2);

int main(int argc, char **argv) {
    if (argc < 4) {
        cout << "Incorrect number of arguments." << endl;
        cout << "usage: ./" << argv[0] <<
        " video.type overlay_image target_object1.type [target_object2.type ...] [noise] [hessianThreshold] [fps]" << endl;
        return -1;
    }
    
    // Read in parameters for SURF and playback (if specified)
    // Note: this is totally dodgy but its like 2.00 am
    int hessianThresh = 100;//200;//1000; // Larger = faster, worse matching. Smaller = slower, better matching
    
    // Read in overlay image
    Mat overlay = imread(argv[2], CV_LOAD_IMAGE_COLOR);
    
    // Read in target object and video file
    vector<Mat> target_list;
    for(int i = 3; i < argc; ++i)
        target_list.push_back(imread(argv[i], CV_LOAD_IMAGE_COLOR));
    const uint NUM_TARGETS = target_list.size();
    cout << "Read " << NUM_TARGETS << " images" << endl;
    
    // TEMP: blend first target and overlay
    Mat blendedOverlay;
    Blender blender;
    blender.getBlended(overlay, target_list[0], blendedOverlay);
    
    // TEMP: show pre-blended image
    namedWindow("Pre-blended");
    imshow("Pre-blended", blendedOverlay);
    waitKey(100);
    
    // Read in video file
    vector<Mat> frames_list;
    VideoCapture capture;
    const string video_file = argv[1];
    capture.open(video_file);
    if (!capture.isOpened()) {
        cout << "Unable to open video file." << endl;
        return -1;
    }
	double fps = capture.get(CV_CAP_PROP_FPS);
    cout << "Trying to read video frames..." << endl;
    readFrames(capture, frames_list);
    cout << "Read frames." << endl;
	capture.release();
    
    // Setup SURF detector and precompute keypoints for objects
    vector<Point2f> dummy;
    vector< vector<Point2f> > prev_frame_corners(NUM_TARGETS, dummy);
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
    
    // Setup Brute Force Matcher
    BFMatcher matcher;
    vector< vector< vector<DMatch> > > matches;

	// Setup Kalman
	// 4 pts * 2 axis(x,y) = 8 dimentionality for measurement
	// 8 dims * 2 time steps = 16 total dimentionality for state
	KalmanFilter KF(16, 8, 0);

	// Transition Marix
	Mat_<float> transMatrix(16, 16);
	transMatrix.setTo(Scalar(0));
	for (int i = 0; i < 16; ++i)
		transMatrix(i, i) = 1;
	for (int i = 0; i < 8; ++i)
		transMatrix(i, i+8) = 1;
	KF.transitionMatrix = transMatrix;
	/* Resulting maxtrix looks like
	1 0 0 0  0 0 0 0  1 0 0 0  0 0 0 0
	0 1 0 0  0 0 0 0  0 1 0 0  0 0 0 0
	0 0 1 0  0 0 0 0  0 0 1 0  0 0 0 0
	...
	0 0 0 0  0 0 0 1  0 0 0 0  0 0 0 1
	0 0 0 0  0 0 0 0  1 0 0 0  0 0 0 0
	...
	0 0 0 0  0 0 0 0  0 0 0 0  0 1 0 0
	0 0 0 0  0 0 0 0  0 0 0 0  0 0 1 0
	0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 1
	meaning old co-ords are copied 1:1 (top half of matrix), new are added also with weight 1 (btm half matrix)
	*/

	// Pre-allocate measurement array, gets recycled each frame
	Mat_<float> measurement(8,1);
	measurement.setTo(Scalar(0));

	// Initial state is 0
	KF.statePre.setTo(Scalar(0));

	setIdentity(KF.measurementMatrix);
	setIdentity(KF.processNoiseCov, Scalar::all(1e-5));
	setIdentity(KF.measurementNoiseCov, Scalar::all(1e-1));
	setIdentity(KF.errorCovPost, Scalar::all(1e-1));

	bool firstMeasure = true;
	// End Kalman Setup
    
    // Misc setup
	if (SHOW_KALMAN_GRAPH)
	    namedWindow("Kalman Graph", CV_WINDOW_AUTOSIZE);

    namedWindow("Object Matching", CV_WINDOW_AUTOSIZE);
    namedWindow("Overlay", CV_WINDOW_AUTOSIZE);

	Mat frame;
    Mat img_out, img_matches;
	Mat unwarpMask = Mat::zeros(frames_list[0].size(), CV_8U);
	Mat kalman_graph = Mat::zeros(frames_list[0].rows, frames_list.size(), CV_8UC3);

	double start;
    double temp_start = getTime();
    int num_frames = 0;
    int x = 0, y = 0;
    
	// Hunt for object in each video frame
    for (int f = 0; f < frames_list.size(); f++) {//while(capture.read(frame)) {
        frame = frames_list[f];
        num_frames++;
        start = getTime();
        // Set size of output matrix if needed
        if (img_out.empty() && x == 0) {
            // Attach other target objects to the left side
            int widthPad = 0;
            int maxWidth = 0;
            int maxHeight = 0;
            for (int i = 0; i < NUM_TARGETS; i++) {
                maxWidth = max(maxWidth, target_list[i].cols);
                maxHeight = max(maxHeight, target_list[i].rows);
                //cout << "maxWidth: " << maxWidth << endl;
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
                //cout << "Placing " << i << "th image at x: " << x << endl;
                Mat displayRegion(img_out, Rect(x, y, target_list[i].cols,
                                                target_list[i].rows));
                y += target_list[i].rows;
                target_list[i].copyTo(displayRegion);
            }
            // Messy
            if (NUM_TARGETS > 1)
                x += target_list[NUM_TARGETS-1].cols; //widthPad - target_list[0].cols;
            //cout << "rest will start from x: " << x << endl;
        }
        
        // Detect keypoints using SURF descriptor
        //cout << "Recomputing: " << hessianThresh << endl;
        SurfFeatureDetector detector_new(hessianThresh);
        detector = detector_new;
        vector<KeyPoint> frame_keypoints;
        vector< vector<KeyPoint> > object_keypoints_list;
        for(uint i = 0; i < NUM_TARGETS; ++i) {
            vector<KeyPoint> kp;
            detector.detect(target_list[i], kp);
            object_keypoints_list.push_back(kp);
        }
        
        // Setup descriptors and compute for objects
        SurfDescriptorExtractor extractor;
        Mat frame_descriptors;
        vector<Mat> object_descriptors_list(NUM_TARGETS);
        for(uint i = 0; i < NUM_TARGETS; ++i)
            extractor.compute(target_list[i], object_keypoints_list[i], object_descriptors_list[i]);
        
        frame_keypoints.clear();
        detector.detect(frame, frame_keypoints);
        
        // Calculate descriptors (based off keypoints)
        extractor.compute(frame, frame_keypoints, frame_descriptors);
        
        // Match descriptors (using FLANN matcher)
        // Ensure enough descriptors for matching
        // (rough, could be refactored)
        if (frame_descriptors.rows <= 1) {
            //cout << "frame descriptor list too small nigga!!!" << endl;
            continue;
        }
        bool enough_descriptors = true;
        for (uint i = 0; i < NUM_TARGETS; i++)
            if (object_descriptors_list[i].rows <= 1)
                enough_descriptors = false;
        if (!enough_descriptors)
            continue;
        
        // TEMP: output for debugging only
        //cout << object_descriptors_list[0].size() << endl;
        //cout << frame_descriptors.size() << endl;
        matches.clear(); //Delete previous matches
        for (uint i = 0; i < NUM_TARGETS; ++i) {
            vector< vector<DMatch> > m;
            //vector<DMatch> match_vec;
            matcher.knnMatch(object_descriptors_list[i], frame_descriptors, m, 2);
            matches.push_back(m);
        }
        //cout << "Made it through matching nigga" << endl;
        
        // Draw only the "good" matches
        vector< vector<DMatch> > good_match_list;
        for(uint i = 0; i < NUM_TARGETS; ++i) {
            vector< vector<DMatch> >& match = matches[i];
            vector<DMatch> gm_tmp;
            for(uint k = 0; k < match.size(); k++) {
                // matches[k].size() == 2 is rather strict. 0.7 is a good compromise between bad matches and no matches
                if(match[k].size() == 2 && match[k][0].distance < 0.6*(match[k][1].distance))
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
            
            if (good_matches.size() < 10) {
				// Update kalman filter with no new measurement

				KF.measurementMatrix.setTo(0);
				KF.predict();
				Mat estimated = KF.correct(measurement);
				// Graph esiamted point
				circle(kalman_graph, Point2f(f, estimated.at<float>(2*2 +1)), 1, Scalar(0, 255, 0), -1);

            } else {
                //cout << obj << ": " << good_matches.size() << " good matches" << endl;
				
				//Align drawings with the frame
                float xOffset = static_cast<float>(target_list[0].cols);

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
                vector<Point2f> object_corners = makeCornerVec(target_object);
                vector<Point2f> frame_corners(4);
                perspectiveTransform(object_corners, frame_corners, H);

				//// Kalman Smoothing Step ////
				// Need a flat array
				for (int i = 0; i < 4; ++i) {
					measurement(2*i)    = frame_corners[i].x;
					measurement(2*i +1) = frame_corners[i].y;
					// Draw un-corrected point (in pink, yay pink!)
					circle(img_matches, frame_corners[i] + Point2f(xOffset, 0), 5, Scalar(255, 0, 255), -1);
				}
				
				// Need to init the position if this is the first time the object is found
				if (firstMeasure) {
					for (int i = 0; i < 8; ++i)
						KF.statePost.at<float>(i) = measurement(i);
					firstMeasure = false;
				}
				setIdentity(KF.measurementMatrix);

				// First call predict, to update the internal statePre variable
				KF.predict();
				
				// Graph un-corrected point
				circle(kalman_graph, Point2f(f, frame_corners[2].y), 1, Scalar(255, 0, 255), -1);
				
				// Correct measurement using estimation
				Mat estimated = KF.correct(measurement);
				for (int i = 0; i < 4; ++i) {
					frame_corners[i].x = estimated.at<float>(2*i);
					frame_corners[i].y = estimated.at<float>(2*i +1);
				}

				// Graph corrected point
				circle(kalman_graph, Point2f(f, frame_corners[2].y), 1, Scalar(0, 255, 0), -1);

				// Extract the target area from the frame and warp to the be rectangular for blending
				Mat exract_transform;
				Mat cur_target = extractAndWarp(frame, frame_corners);
				// Do blend
				blender.getBlended(overlay, cur_target, blendedOverlay);
				imshow("Pre-blended", blendedOverlay);

				// Make mask of extracted area so it is properly replaced
				unwarpMask.setTo(0);
				Point f_pts[4];
				for (int i = 0; i < 4; ++i)
					f_pts[i] = frame_corners[i];
				fillConvexPoly(unwarpMask, f_pts, 4, 255, 8, 0);
				
				// Un-warp and replace
				Mat unwarpTarget = Mat::zeros(frame.size(), frame.type());
				unWarp(blendedOverlay, frame_corners, unwarpTarget);
				unwarpTarget.copyTo(frame, unwarpMask);

                // Overlay image (in separate frame display, outside of img_matches)
                //warpPerspective(blendedOverlay, frame, H, frame.size(), INTER_LINEAR, BORDER_TRANSPARENT);
                
                // Draw the actual lines around the detected object
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
        Mat displayRest(img_out, Rect(x, 0, img_matches.cols, 
                                      img_matches.rows));
        img_matches.copyTo(displayRest);
        
        imshow("Object Matching", img_out);
        // TEMP: show frame with overlay
        imshow("Overlay", frame);
		if (SHOW_KALMAN_GRAPH)
			imshow("Kalman Graph", kalman_graph);
        if(waitKey(1) >= 0) break;
    }

    // TEMP: for testing, remove later
    double temp_end = getTime() - temp_start;
    cout << "Entire video took: " << temp_end << " ms " << endl;
    cout << "There were: " << num_frames << " frames" << endl;
    cout << "Average fps: " << (double) num_frames / (temp_end / 1000.0) << endl;

	cout << "Writing processed video at " << fps << " fps" << endl;
	writeFrames(frames_list, fps);
	cout << "Done writing" << endl;
    
    return 0;
}

// Read video frames in
void readFrames(VideoCapture& vc, vector<Mat>& frames_list) {
    Mat frame;
    while(vc.read(frame)) {
        frames_list.push_back(frame.clone());
    }   
}

void writeFrames(vector<Mat>& frames_list, double fps) {
	VideoWriter vOut = VideoWriter("out.avi", CV_FOURCC('P','I','M','1'), fps, frames_list[0].size());
	for (unsigned f = 0; f < frames_list.size(); ++f) {
        vOut.write(frames_list[f]);
    }   
}

vector<Point2f> makeCornerVec(Mat& src) {
	vector<Point2f> vec(4);
	vec[0] = Point2f(0, 0);
	vec[1] = Point2f(src.cols, 0);
	vec[2] = Point2f(src.cols, src.rows);
	vec[3] = Point2f(0, src.rows);
	return vec;
}

vector<Point2f> rectToVec(Rect_<float>& r) {
	vector<Point2f> vec(4);
	vec[0] = Point2f(r.x, r.y);
	vec[1] = Point2f(r.x+r.width, r.y);
	vec[2] = Point2f(r.x+r.width, r.y+r.height);
	vec[3] = Point2f(0, r.y+r.height);
	return vec;
}

Mat extractAndWarp(Mat& image, vector<Point2f>& corners) {
	Rect_<float> bound = makeBoundingBox(corners);
	bound = bound - Point2f(bound.x, bound.y); // x = y = 0, we dont want any offset
	vector<Point2f> outSize = rectToVec(bound);
    Mat transform = getPerspectiveTransform(corners, outSize);
    Mat out = Mat::zeros(bound.height, bound.width, image.type());
    warpPerspective(image, out, transform, out.size());
    return out;
}

void unWarp(Mat& image, vector<Point2f>& corners, Mat& target) {
	vector<Point2f> imgPts = rectToVec(Rect_<float>(0, 0, image.cols, image.rows));	
    Mat transform = getPerspectiveTransform(imgPts, corners);
    warpPerspective(image, target, transform, target.size());
}

Rect makeBoundingBox(vector<Point2f>& pts) {
	Rect_<float> r;
	r.x = pts[0].x;
	r.y = pts[0].y;
	float maxX= pts[0].x;
	float maxY= pts[0].y;
	for (int i = 1; i < 4; ++i) {
		r.x = std::min(r.x, pts[i].x);
		r.y = std::min(r.y, pts[i].y);
		maxX= std::max(maxX, pts[i].x);
		maxY= std::max(maxY, pts[i].y);
	}
	r.width = maxX - r.x;
	float height = maxY - r.y;
	r.height = height;
	return r;
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
    float x_dist = abs(p1.x - p2.x);
    float y_dist = abs(p2.y - p2.y);
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
