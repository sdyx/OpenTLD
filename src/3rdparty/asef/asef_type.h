#ifndef ASEF_TYPE_H_
#define ASEF_TYPE_H_

//#include <cv.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/video/video.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/video/tracking.hpp>
#include "opencv2/core/core.hpp"

typedef struct
{

	IplImage * input_image;
	CvRect face_rect;	
	CvPoint left_eye, right_eye;
	
	// Internal face detection vairables
	//CvHaarClassifierCascade* face_detection_classifier;
	CvMemStorage* face_detection_buffer;

	// Internal ASEF variables
	CvMat *lfilter, *rfilter;
	int n_rows, n_cols;
	CvRect lrect, rrect;
	CvMat *lfilter_dft, *rfilter_dft;

	CvMat face_image;
	CvMat *scaled_face_image_32fc1, *scaled_face_image_8uc1;
	
	CvMat *lcorr, *rcorr;
	CvMat *lroi, *rroi;
	CvMat *lut;

} AsefEyeLocator;

#endif
