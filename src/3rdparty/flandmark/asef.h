#ifndef ASEF_H_
#define ASEF_H_

#include <cv.h>

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

#endif
