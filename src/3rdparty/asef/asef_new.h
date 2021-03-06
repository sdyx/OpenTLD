#ifndef ASEF_H_
#define ASEF_H_

//#include <cv.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/video/video.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/video/tracking.hpp>
#include "opencv2/core/core.hpp"

/*
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
*/
/*
int asef_initialze( AsefEyeLocator *asef,
		const char *asef_file_name );//,
//		const char *fd_file_name
//);
void asef_destroy( AsefEyeLocator *asef );
int asef_detect_face( AsefEyeLocator *asef,
		CvRect area
);
void asef_locate_eyes( AsefEyeLocator *asef, CvRect area );
//void asef_set_image( AsefEyeLocator *asef, IplImage *image );
*/

void ASEF_locate_eyes( AsefEyeLocator *asef, CvRect area );
int ASEF_detect_face( AsefEyeLocator *asef, CvRect area );
int ASEF_read_line( FILE* fp, char* buf, int size );
void ASEF_destroy( AsefEyeLocator *asef );
int ASEF_load_filters( const char* file_name,
		int *p_n_rows,
		int *p_n_cols, 
		CvRect *left_eye_region,
		CvRect *right_eye_region, 
		CvMat **p_left_filter,
		CvMat **p_right_filter
);
int ASEF_initialze( AsefEyeLocator *asef,
		const char *asef_file_name);

#endif
