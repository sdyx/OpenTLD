#include <stdio.h>
#include "asef.h"

#define LINE_BUF_SIZE 1024

// FROM https://github.com/laoyang/ASEF

static int load_asef_filters( const char* file_name,
		int *p_n_rows,
		int *p_n_cols, 
		CvRect *left_eye_region,
		CvRect *right_eye_region, 
		CvMat **p_left_filter,
		CvMat **p_right_filter
);

static int read_line( FILE *fp, char *buf, int size );


int asef_initialze( AsefEyeLocator *asef,
		const char *asef_file_name//,
//		const char *fd_file_name		was the haarcascade path
)
{

	if ( !asef ||
			!asef_file_name ||
//			!fd_file_name || 
			strlen( asef_file_name ) == 0// ||
//			strlen(fd_file_name)==0
	)
	{
		return -1;
	}

  // For face detection:
	/*
	asef->face_detection_buffer = cvCreateMemStorage( 0 );
	asef->face_detection_classifier = fd_load_detector( fd_file_name );

	if ( !asef->face_detection_classifier )
	{
		return -1;
	}
	*/ 

  // For asef eye locator:

	if ( load_asef_filters( asef_file_name,
			&asef->n_rows,
			&asef->n_cols,
			&asef->lrect,
			&asef->rrect,
			&asef->lfilter,
			&asef->rfilter
		)
	)
	{
		return -1;
	}

	asef->lfilter_dft = cvCreateMat(asef->n_rows, asef->n_cols, CV_32FC1);
	asef->rfilter_dft = cvCreateMat(asef->n_rows, asef->n_cols, CV_32FC1);

	asef->scaled_face_image_32fc1 = cvCreateMat(asef->n_rows, asef->n_cols, CV_32FC1);
	asef->scaled_face_image_8uc1 = cvCreateMat(asef->n_rows, asef->n_cols, CV_8UC1);

	asef->lcorr = cvCreateMat(asef->n_rows, asef->n_cols, CV_32FC1);
	asef->rcorr = cvCreateMat(asef->n_rows, asef->n_cols, CV_32FC1);

	asef->lroi = cvCreateMatHeader(asef->n_rows, asef->n_cols, CV_32FC1);
	asef->rroi = cvCreateMatHeader(asef->n_rows, asef->n_cols, CV_32FC1);

	asef->lut = cvCreateMat(256, 1, CV_32FC1);

	if ( !(asef->lfilter_dft &&
			asef->rfilter_dft &&
			asef->scaled_face_image_32fc1 &&
			asef->scaled_face_image_8uc1 &&
			asef->lcorr &&
			asef->rcorr &&
			asef->lroi &&
			asef->rroi &&
			asef->lut
	) )
	{
		return -1;
	}

	cvDFT( asef->lfilter, asef->lfilter_dft, CV_DXT_FORWARD, 0 );
	cvDFT( asef->rfilter, asef->rfilter_dft, CV_DXT_FORWARD, 0 );

	cvGetSubRect( asef->lcorr, asef->lroi, asef->lrect );
	cvGetSubRect( asef->rcorr, asef->rroi, asef->rrect );


	for ( int i = 0; i < 256; i++ )
	{
		cvmSet( asef->lut, i, 0, 1.0 + i );
	}
	cvLog( asef->lut, asef->lut );

	return 0;
}
/*
void asef_set_image( AsefEyeLocator *asef, IplImage *image )
{
	asef->input_image = image;
}
*/


void asef_destroy( AsefEyeLocator *asef )
{

	cvReleaseMemStorage( &asef->face_detection_buffer );
//	cvReleaseHaarClassifierCascade( &asef->face_detection_classifier );

	cvReleaseMat( &asef->lfilter );
	cvReleaseMat( &asef->rfilter );
	cvReleaseMat( &asef->lfilter_dft );
	cvReleaseMat( &asef->rfilter_dft );
	cvReleaseMat( &asef->scaled_face_image_32fc1 );
	cvReleaseMat( &asef->scaled_face_image_8uc1 );
	cvReleaseMat( &asef->lcorr );
	cvReleaseMat( &asef->rcorr );
	cvReleaseMat( &asef->lroi );
	cvReleaseMat( &asef->rroi );
	cvReleaseMat( &asef->lut );
}

int asef_detect_face( AsefEyeLocator *asef, CvRect area )
{
	//return fd_detect_face(asef->input_image, asef->face_detection_classifier, 
	//	&asef->face_rect, asef->face_detection_buffer);
	
	// FB
	//&asef->face_rect = cv::Rect( 1, 1, 100, 100 );
	asef->face_rect = area;
	return 1;
}


void asef_locate_eyes( AsefEyeLocator *asef, CvRect area )
{
	
	asef->face_image.cols = asef->face_rect.width;
	asef->face_image.rows = asef->face_rect.height;
	
	/*asef->face_image.cols = area.width;
	asef->face_image.rows = area.height;
	*/
	
	asef->face_image.type = CV_8UC1;
	asef->face_image.step = asef->face_rect.width;

	cvGetSubRect( asef->input_image,
			&asef->face_image,
			asef->face_rect
	);

	double xscale = ((double)asef->scaled_face_image_8uc1->cols)/((double)asef->face_image.cols);
	double yscale = ((double)asef->scaled_face_image_8uc1->rows)/((double)asef->face_image.rows);

	cvResize(&asef->face_image, asef->scaled_face_image_8uc1, CV_INTER_LINEAR);

	cvLUT(asef->scaled_face_image_8uc1, asef->scaled_face_image_32fc1, asef->lut);

	cvDFT(asef->scaled_face_image_32fc1, asef->scaled_face_image_32fc1, CV_DXT_FORWARD, 0);
	cvMulSpectrums(asef->scaled_face_image_32fc1, asef->lfilter_dft, asef->lcorr, CV_DXT_MUL_CONJ);
	cvMulSpectrums(asef->scaled_face_image_32fc1, asef->rfilter_dft, asef->rcorr, CV_DXT_MUL_CONJ);

	cvDFT(asef->lcorr, asef->lcorr, CV_DXT_INV_SCALE, 0);
	cvDFT(asef->rcorr, asef->rcorr, CV_DXT_INV_SCALE, 0);

	cvMinMaxLoc(asef->lroi, NULL, NULL, NULL, &asef->left_eye, NULL);
	cvMinMaxLoc(asef->rroi, NULL, NULL, NULL, &asef->right_eye, NULL);

	asef->left_eye.x = (asef->lrect.x + asef->left_eye.x)/xscale + asef->face_rect.x;
	asef->left_eye.y = (asef->lrect.y + asef->left_eye.y)/yscale + asef->face_rect.y;
	asef->right_eye.x = (asef->rrect.x + asef->right_eye.x)/xscale + asef->face_rect.x;
	asef->right_eye.y = (asef->rrect.y + asef->right_eye.y)/yscale + asef->face_rect.y;
}



int load_asef_filters(const char* file_name, int *p_n_rows, int *p_n_cols, 
	CvRect *left_eye_region, CvRect *right_eye_region, 
	CvMat **p_left_filter, CvMat **p_right_filter){

	int rv;

	char buf[LINE_BUF_SIZE];

	FILE *fp = fopen(file_name, "r");

	if (!fp){
		return -1;
	}

int n_rows, n_cols; // row and column size


read_line(fp, buf, LINE_BUF_SIZE);
printf("%s\n", buf);
if (strcmp(buf, "CFEL")){
	return -1;
}

// Print comments and copyright
for (int i = 0; i < 2; i++){
	if (read_line(fp, buf, LINE_BUF_SIZE) <= 0){
		return -1;
	}
	printf("%s\n", buf);
}

read_line(fp, buf, LINE_BUF_SIZE);
sscanf(buf, "%d %d", &n_rows, &n_cols);
*p_n_rows = n_rows;
*p_n_cols = n_cols;

int rect_x, rect_y, rect_width, rect_hight;
read_line(fp, buf, LINE_BUF_SIZE);
sscanf(buf, "%d %d %d %d", &rect_x, &rect_y, &rect_width, &rect_hight);

if (left_eye_region){
	*left_eye_region = cvRect(rect_x, rect_y, rect_width, rect_hight); 
}

read_line(fp, buf, LINE_BUF_SIZE);
sscanf(buf, "%d %d %d %d", &rect_x, &rect_y, &rect_width, &rect_hight);

if (right_eye_region){
	*right_eye_region = cvRect(rect_x, rect_y, rect_width, rect_hight); 
}

uint32_t endien_checker;
unsigned long endianness;
read_line(fp, buf, LINE_BUF_SIZE);  
endien_checker = *(uint32_t*)buf;


if ( !strcmp(buf, "ABCD") ){
// Big endian
	endianness = 0;
} else if ( !strcmp(buf, "DCBA")){
// Little endian
// Almost always this case on your x86 machine. 
// Not sure about ARM (Android/iOS), you can test it out :) 
	endianness = 1;
} else {
	endianness = -1;
}

// TODO: handle big endian with byte swap;

size_t filter_data_size = n_rows * n_cols * sizeof(float);
CvScalar mean, std_dev;


unsigned char* lfilter_data = (unsigned char*)malloc(filter_data_size);
rv = fread(lfilter_data, 1, filter_data_size, fp);
assert(rv == filter_data_size);

if (p_left_filter){ 

	CvMat *left_filter = cvCreateMatHeader(n_rows, n_cols, CV_32FC1);
	cvSetData(left_filter, lfilter_data, CV_AUTO_STEP);
	cvAvgSdv(left_filter, &mean, &std_dev, NULL);
	cvScale(left_filter, left_filter, 1.0/std_dev.val[0], -mean.val[0]*1.0/std_dev.val[0]);
	*p_left_filter = left_filter; 

} else{
	free(lfilter_data);
}

unsigned char* rfilter_data = (unsigned char*)malloc(filter_data_size);
rv = fread(rfilter_data, 1, filter_data_size, fp);
assert(rv == filter_data_size);

if (p_right_filter){
	CvMat *right_filter = cvCreateMatHeader(n_rows, n_cols, CV_32FC1);
	cvSetData(right_filter, rfilter_data, CV_AUTO_STEP);
	cvAvgSdv(right_filter, &mean, &std_dev, NULL);
	cvScale(right_filter, right_filter, 1.0/std_dev.val[0], -mean.val[0]*1.0/std_dev.val[0]);
	*p_right_filter = right_filter;
} else{
	free(rfilter_data);
}

fclose(fp);

return 0;

}


int read_line(FILE* fp, char* buf, int size){
	int c, i = 0;
	while (i < (size - 1) && (c = fgetc(fp)) != EOF){
		if ( c == '\n' ) {
			break;
		} 
		buf[i++] = c;
	}

	buf[i] = '\0';
	return i;
}
