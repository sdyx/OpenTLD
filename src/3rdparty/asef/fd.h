#ifndef _FD_H_
#define _FD_H_

#include <cv.h>

CvHaarClassifierCascade* fd_load_detector( const char* cascade_path );
int fd_detect_face(IplImage* image, CvHaarClassifierCascade* cascade, CvRect *rect, CvMemStorage* buffer);

#endif