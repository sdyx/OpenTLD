#include "fd.h"

CvHaarClassifierCascade* fd_load_detector( const char* cascade_path )
{
	return (CvHaarClassifierCascade*)cvLoad( cascade_path, NULL, NULL, NULL);
}

int fd_detect_face(IplImage* image, CvHaarClassifierCascade* cascade, CvRect *rect, CvMemStorage* buffer){
	
	CvSeq* faces;
	int rv = 0;

/* use the fastest variant */
	faces = cvHaarDetectObjects( image,
			cascade,
			buffer,
			1.2,
			3, 
			CV_HAAR_DO_CANNY_PRUNING | CV_HAAR_FIND_BIGGEST_OBJECT | CV_HAAR_DO_ROUGH_SEARCH, 
			cvSize( 0, 0 ),
			cvSize(0, 0)
	);


/* draw all the rectangles */
	if ( faces->total > 0)
	{
/* extract the rectanlges only */
		*rect = *(CvRect*) cvGetSeqElem( faces, 0 );
		rv = 1;
	} 
	else 
	{
		rv = 0;
	}
	return rv;
}
