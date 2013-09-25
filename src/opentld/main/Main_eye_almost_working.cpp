/*  Copyright 2011 AIT Austrian Institute of Technology
*
*   This file is part of OpenTLD.
*
*   OpenTLD is free software: you can redistribute it and/or modify
*   it under the terms of the GNU General Public License as published by
*    the Free Software Foundation, either version 3 of the License, or
*   (at your option) any later version.
*
*   OpenTLD is distributed in the hope that it will be useful,
*   but WITHOUT ANY WARRANTY; without even the implied warranty of
*   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
*   GNU General Public License for more details.
*
*   You should have received a copy of the GNU General Public License
*   along with OpenTLD.  If not, see <http://www.gnu.org/licenses/>.
*
*/
/*
 * MainX.cpp
 *
 *  Created on: Nov 17, 2011
 *      Author: Georg Nebehay
 *
 * Main_eye.cpp
 * 	Adapted by: Felix Baumann on Sept. 09, 2013
 */

#include "Main.h"

#include "Config.h"
#include "ImAcq.h"
#include "Gui.h"
#include "TLDUtil.h"
#include "Trajectory.h"
#include "flandmark_detector.h"

//#include "asef_new.h"
#include "asef_type.h"

#define LINE_BUF_SIZE 1024

using namespace tld;
using namespace cv;


double Main::calculateVariance( cv::Mat& image )
{
	// FB: code from
	// http://stackoverflow.com/questions/2608782/mean-and-variance-of-image-in-single-pass
	
	int pi,a,b;
	double var = 0.0;
	int h,w;
	h = image.rows;
	w = image.cols;
	
	for( int i = 1; i < h - 1; i++ )
	{
		for( int j = 1; j < w - 1; j++ )
		{   
			int sq = 0, sum = 0;
		    double mean = 0;
		    var = 0;
		    for( a = -1; a <= 1; a++ )
		    {
		        for( b =- 1; b <= 1; b++ )
		        {
		            //pi = data[ ( i + a ) * step + ( j + b ) ];
		            pi = image.at< uchar >( cv::Point( i + a , j + b ));
		            sq = pi * pi;
		            sum = sum + sq;
		            mean = mean + pi;
		        }
		    }
		    mean = mean / 9;
		    double soa = mean * mean;//square of average
		    double aos = sum / 9;//mean of squares
		    var = aos - soa;//variance
		}
	}
	return var;
}

int ASEF_detect_face( AsefEyeLocator *asef, CvRect area )
{
	//return fd_detect_face(asef->input_image, asef->face_detection_classifier, 
	//	&asef->face_rect, asef->face_detection_buffer);
	
	// FB
	//&asef->face_rect = cv::Rect( 1, 1, 100, 100 );
	asef->face_rect = area;
	return 1;
}

void ASEF_locate_eyes( AsefEyeLocator *asef, CvRect area )
{
	
	asef->face_image.cols = asef->face_rect.width;
	asef->face_image.rows = asef->face_rect.height;
	
	//asef->face_image.cols = area.width;
	//asef->face_image.rows = area.height;
	
	
	asef->face_image.type = CV_8UC1;
	asef->face_image.step = asef->face_rect.width;

	cvGetSubRect( asef->input_image,
			&asef->face_image,
			asef->face_rect
	);

	double xscale = ( ( double ) asef->scaled_face_image_8uc1->cols ) 
			/ ( ( double ) asef->face_image.cols );
	double yscale = ( ( double ) asef->scaled_face_image_8uc1->rows )
			/ ( ( double )asef->face_image.rows );
			
	// error here

	cvResize( &asef->face_image,
			asef->scaled_face_image_8uc1,
			CV_INTER_LINEAR
	);

	// src, dst, lut
	// asef->lut = cvCreateMat(256, 1, CV_32FC1);
	cvLUT( asef->scaled_face_image_8uc1,
			asef->scaled_face_image_32fc1,
			asef->lut
	);

	cvDFT( asef->scaled_face_image_32fc1,
			asef->scaled_face_image_32fc1,
			CV_DXT_FORWARD,
			0
	);
	cvMulSpectrums( asef->scaled_face_image_32fc1,
			asef->lfilter_dft,
			asef->lcorr,
			CV_DXT_MUL_CONJ
	);
	//cvMulSpectrums( asef->scaled_face_image_32fc1,
	//		asef->rfilter_dft,
	//		asef->rcorr,
	//		CV_DXT_MUL_CONJ
	//);

	cvDFT( asef->lcorr,
			asef->lcorr,
			CV_DXT_INV_SCALE,
			0
	);
	//cvDFT( asef->rcorr,
	//		asef->rcorr,
	//		CV_DXT_INV_SCALE,
	//		0
	//);

	cvMinMaxLoc( asef->lroi,
			NULL,
			NULL,
			NULL,
			&asef->left_eye,
			NULL
	);
	//cvMinMaxLoc( asef->rroi,
	//		NULL,
	//		NULL,
	//		NULL,
	//		&asef->right_eye,
	//		NULL
	//);

	asef->left_eye.x = ( asef->lrect.x + asef->left_eye.x)
			/ xscale + asef->face_rect.x;
	asef->left_eye.y = ( asef->lrect.y + asef->left_eye.y)
			/ yscale + asef->face_rect.y;
			
	//asef->right_eye.x = (asef->rrect.x + asef->right_eye.x)/xscale + asef->face_rect.x;
	//asef->right_eye.y = (asef->rrect.y + asef->right_eye.y)/yscale + asef->face_rect.y;
}

int ASEF_read_line(FILE* fp, char* buf, int size){
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


void ASEF_destroy( AsefEyeLocator *asef )
{

	cvReleaseMemStorage( &asef->face_detection_buffer );

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

int ASEF_load_filters( const char* file_name,
		int *p_n_rows,
		int *p_n_cols, 
		CvRect *left_eye_region,
		CvRect *right_eye_region, 
		CvMat **p_left_filter,
		CvMat **p_right_filter
)
{
	// Variable definitions
	CvScalar mean, std_dev;

	uint32_t endian_checker;
	unsigned long endianness;
				
	int rv;

	char buf[LINE_BUF_SIZE];
	int n_rows, n_cols; // row and column size
	int rect_x, rect_y, rect_width, rect_hight;

	FILE *fp = fopen( file_name, "r" );

	if ( !fp )
	{
		return -1;
	}



	ASEF_read_line( fp, buf, LINE_BUF_SIZE );
	printf( "%s\n", buf );
	if ( strcmp( buf, "CFEL" ) )
	{
		return -1;
	}

	// Print comments and copyright
	for ( int i = 0; i < 2; i++ )
	{
		if ( ASEF_read_line( fp, buf, LINE_BUF_SIZE ) <= 0 )
		{
			return -1;
		}
		printf( "%s\n", buf );
	}

	ASEF_read_line( fp, buf, LINE_BUF_SIZE );
	sscanf( buf, "%d %d", &n_rows, &n_cols );
	*p_n_rows = n_rows;
	*p_n_cols = n_cols;

	size_t filter_data_size = n_rows * n_cols * sizeof( float );
	unsigned char* lfilter_data = 
			( unsigned char* ) malloc( filter_data_size );
	unsigned char* rfilter_data = 
			( unsigned char* ) malloc( filter_data_size );

	
	ASEF_read_line( fp, buf, LINE_BUF_SIZE );
	sscanf( buf,
			"%d %d %d %d",
			&rect_x,
			&rect_y,
			&rect_width,
			&rect_hight
	);

	if ( left_eye_region )
	{
		*left_eye_region = cvRect( rect_x,
				rect_y,
				rect_width,
				rect_hight
		); 
	}

	ASEF_read_line( fp, buf, LINE_BUF_SIZE );
	sscanf( buf,
			"%d %d %d %d",
			&rect_x,
			&rect_y,
			&rect_width,
			&rect_hight
	);

	if ( right_eye_region )
	{
		*right_eye_region = cvRect( rect_x,
				rect_y,
				rect_width,
				rect_hight
		); 
	}
	ASEF_read_line( fp, buf, LINE_BUF_SIZE );
	endian_checker = *( uint32_t* ) buf;


	if ( !strcmp(buf, "ABCD" ) )
	{
		// Big endian
		endianness = 0;
	} 
	else if ( !strcmp(buf, "DCBA" ) )
	{
		// Little endian
		// Almost always this case on your x86 machine. 
		// Not sure about ARM (Android/iOS), you can test it out :) 
		endianness = 1;
	} else {
		endianness = -1;
	}

	// TODO: handle big endian with byte swap;

	rv = fread( lfilter_data, 1, filter_data_size, fp );
	assert( rv == filter_data_size );

	if ( p_left_filter )
	{
		CvMat *left_filter = cvCreateMatHeader( n_rows,
				n_cols,
				CV_32FC1
		);
		cvSetData( left_filter, lfilter_data, CV_AUTO_STEP );
		cvAvgSdv( left_filter, &mean, &std_dev, NULL );
		cvScale( left_filter,
				left_filter,
				1.0 / std_dev.val[ 0 ],
				-mean.val[ 0 ] * 1.0 / std_dev.val[ 0 ]
		);
		*p_left_filter = left_filter; 

	} 
	else
	{
		free( lfilter_data );
	}

	
	rv = fread( rfilter_data, 1, filter_data_size, fp );
	assert( rv == filter_data_size );

	if ( p_right_filter )
	{
		CvMat *right_filter = cvCreateMatHeader( n_rows,
				n_cols,
				CV_32FC1
		);
		cvSetData( right_filter, rfilter_data, CV_AUTO_STEP );
		cvAvgSdv( right_filter, &mean, &std_dev, NULL );
		cvScale( right_filter,
				right_filter,
				1.0 / std_dev.val[ 0 ],
				-mean.val[ 0 ] * 1.0 / std_dev.val[ 0 ]
		);
		*p_right_filter = right_filter;
	} 
	else
	{
		free( rfilter_data );
	}

	fclose( fp );

	return 0;

}

int ASEF_initialze( AsefEyeLocator *asef,
		const char *asef_file_name//,
//		const char *fd_file_name		was the haarcascade path
)
{

	if ( !asef ||
			!asef_file_name ||
			strlen( asef_file_name ) == 0
	)
	{
		return -1;
	}

	if ( ASEF_load_filters( asef_file_name,
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

	asef->lfilter_dft = cvCreateMat( asef->n_rows,
			asef->n_cols,
			CV_32FC1
	);
	asef->rfilter_dft = cvCreateMat( asef->n_rows,
			asef->n_cols,
			CV_32FC1
	);

	asef->scaled_face_image_32fc1 = cvCreateMat( asef->n_rows,
			asef->n_cols,
			CV_32FC1
	);
	asef->scaled_face_image_8uc1 = cvCreateMat( asef->n_rows,
			asef->n_cols,
			CV_8UC1
	);

	asef->lcorr = cvCreateMat( asef->n_rows,
			asef->n_cols,
			CV_32FC1
	);
	asef->rcorr = cvCreateMat( asef->n_rows, asef->n_cols, CV_32FC1 );

	asef->lroi = cvCreateMatHeader( asef->n_rows,
			asef->n_cols,
			CV_32FC1
	);
	asef->rroi = cvCreateMatHeader( asef->n_rows,
			asef->n_cols,
			CV_32FC1
	);

	asef->lut = cvCreateMat( 256, 1, CV_32FC1 );

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

cv::Point Main::findPupil( cv::Mat image, cv::Rect boundingBox )
{
	
	char *asef_locator_path = "EyeLocatorASEF128x128.fel";
	//*asef_locator_path = "EyeLocatorASEF128x128.fel";
	AsefEyeLocator asef;
	if ( ASEF_initialze( &asef,
			asef_locator_path
		)
	)
	{
		return cv::Point( 0, 0 );
	}
	IplImage iplimg = image;
	asef.input_image = &iplimg;
	
	//asef_set_image( &asef, &iplimg ); 

	if( ASEF_detect_face( &asef, boundingBox ) )
	{
		ASEF_locate_eyes( &asef, boundingBox );
		//draw_markers(color_img, asef.face_rect, asef.left_eye, asef.right_eye);
	}
	
	cv::Point retPoint = cv::Point( 0, 0 );	
	/*
	retPoint = findEyeCenter( image,
			boundingBox,
			"pupil-A"
	);
	*/
	ASEF_destroy( &asef );
	return retPoint;
	
}
void Main::doHaar( cv::Mat image )
{
	double lowerBound = 80.0;
	double upperBound = 120.0;
	
	cv::Mat grayImage = image.clone();
	
	cv::CascadeClassifier faceCascade;
	std::vector<cv::Rect> objectToDetect;
	std::vector<cv::Rect> listOfBB;
	cv::Rect lastBBElement;
	cv::Scalar imageMean;
	int loops = 10; // loop over 10 frames and extract
					// the bounding boxes for the faces.
					// Select the one that overlays the most.
	
	faceCascade.load( "haarcascade_frontalface_alt2.xml" );	
	
	if ( faceCascade.empty() )
	{
		std::cout << "ERR: Could not load a cascade. ABORT"
				<< std::endl;
		//break;
	}
	
				
	// FB: TODO
	// the idea is to get the mean/average of the face in order to
	// make an assumption (later on) wheter we are currently tracking the
	// face or not. (Facial mean has to be in a specific range). TBD
	
	for ( int counter = 0;
			( counter <= loops || listOfBB.size() == 0 );
			counter++ )
	{
		faceCascade.detectMultiScale( grayImage,
					objectToDetect,
					1.2,
					3,
					//0|CV_HAAR_SCALE_IMAGE,
					CV_HAAR_SCALE_IMAGE|CV_HAAR_DO_CANNY_PRUNING,
					cv::Size( 20, 20 )
		);
		
		imageMean = cv::mean( grayImage );
		
		// FB: TODO
		// 1. check if variance is roughly similar in all of the detected faces
		// 2. check if the variance is inbetween specific bounds
		// 3. check if the detected areas are located in the same region
		// 4. check if the dimensions of the areas are roughly the same
		// 
		// if the last one detected does not match the criteria keep 
		// searching for it.
		if ( objectToDetect.size() )
		{
        	printf( "\t[ %d, %d, %d, %d, %f ]\n",
        		objectToDetect[ 0 ].x,
        		objectToDetect[ 0 ].y,
        		objectToDetect[ 0 ].width,
        		objectToDetect[ 0 ].height,
        		imageMean[ 0 ]
        	);
			
		}
		if ( objectToDetect.size() &&
				imageMean[ 0 ] >= lowerBound &&
				imageMean[ 0 ] <= upperBound
		)
		{
			listOfBB.push_back( objectToDetect[ 0 ] );
		}
		
    	grayImage = imAcqGetImg( imAcq );
    	cvtColor( cv::Mat( grayImage )  , grayImage, CV_BGR2GRAY );
    	//cv::imshow( "searching ...", grayImage );
    	
                char fileName[ 256 ];
                sprintf( fileName, 
                		"%.5d.png",
						imAcq->currentFrame - 1
                );
				cv::rectangle( grayImage, objectToDetect[ 0 ], CV_RGB( 10, 20, 100 ) );
				cv::imwrite( fileName, grayImage );
//                cvSaveImage( fileName, grayImage );
	}
	
	lastBBElement = listOfBB.back();

	initialBB[ 0 ] = lastBBElement.x;
	initialBB[ 1 ] = lastBBElement.y;
	initialBB[ 2 ] = lastBBElement.width;
	initialBB[ 3 ] = lastBBElement.height;
	
	if ( initialBB[ 0 ] < 0 ||
			initialBB[ 1 ] < 0 ||
			initialBB[ 2 ] < 0 ||
			initialBB[ 3 ] < 0
	)
	{
		initialBB[ 0 ] = initialBB[ 1 ] = initialBB[ 2 ] = initialBB[ 3 ] = 1;
	}
	cv::rectangle( grayImage, lastBBElement, CV_RGB( 10, 20, 100 ) );
	cv::imshow( "final picture of detection stage", grayImage );
	
	//cvReleaseImage( grayImage );
}

cv::Rect Main::doDetectEye( cv::Mat image )
{
	cv::Mat grayImage;
	
	grayImage = image( cv::Rect( tld->currBB->x,
			tld->currBB->y,
			tld->currBB->width,
			tld->currBB->height
	) 	);
	
	cv::CascadeClassifier eyeCascade;
	std::vector<cv::Rect> objectToDetect;
	std::vector<cv::Rect> listOfBB;
	cv::Rect eyeRect = cv::Rect( 0, 0, 10, 10 );
	//cv::Scalar imageMean;
	
	eyeCascade.load( "shameem_eye.xml" );	
	
	if ( eyeCascade.empty() )
	{
		std::cout << "ERR: Could not load a cascade. ABORT"
				<< std::endl;
	}
	
	eyeCascade.detectMultiScale( grayImage,
				objectToDetect,
				1.2,
				3,
				//0|CV_HAAR_SCALE_IMAGE,
				CV_HAAR_SCALE_IMAGE|CV_HAAR_DO_CANNY_PRUNING,
				cv::Size( 20, 20 )
	);
	if ( objectToDetect.size() )
	{
    	printf( "\tEye [ %d, %d, %d, %d ]\n",
    		objectToDetect[ 0 ].x,
    		objectToDetect[ 0 ].y,
    		objectToDetect[ 0 ].width,
    		objectToDetect[ 0 ].height
    	);
		eyeRect = objectToDetect[ 0 ];	
	}	
	return eyeRect;
}

std::vector<cv::Point> Main::doFlandmark( IplImage image, cv::Rect& detectionArea )
{
	std::vector<cv::Point> retVector;
	FLANDMARK_Model * flandmarkModel = flandmark_init( "flandmark_model.dat" );
	int flandmarkBoundingBox[] = { 0, 0, 0, 0 };
	flandmarkBoundingBox[ 0 ] = detectionArea.x;
	flandmarkBoundingBox[ 1 ] = detectionArea.y;
	flandmarkBoundingBox[ 2 ] = detectionArea.width;
	flandmarkBoundingBox[ 3 ] = detectionArea.height;
	
	cv::Mat tmp = cv::Mat( &image );
	//IplImage imgGrayscale = tmp( detectionArea );
	IplImage imgGrayscale = image;
	
	double * flandmarkLandmarks;
	cv::Point leftEyeLeftCorner_Flandmark, 
			rightEyeRightCorner_Flandmark, 
			mouthRight_Flandmark,
			mouthLeft_Flandmark;
			
	flandmarkLandmarks = (double*) malloc( 
			2 * flandmarkModel->data.options.M * sizeof( double ) 
	);
	//imgGrayscale = image( &detectionArea );
	flandmark_detect( &imgGrayscale,
			flandmarkBoundingBox,
			flandmarkModel,
			flandmarkLandmarks
	);
	
	std::cout << "l " <<
			flandmarkLandmarks[ 4 ] << " y " <<
			flandmarkLandmarks[ 5 ] << std::endl;
			
	std::cout << "r " <<
			flandmarkLandmarks[ 2 ] << " y " <<
			flandmarkLandmarks[ 3 ] << std::endl;
	leftEyeLeftCorner_Flandmark = cv::Point( flandmarkLandmarks[ 4 ],
			flandmarkLandmarks[ 5 ] );
	rightEyeRightCorner_Flandmark = cv::Point( flandmarkLandmarks[ 2 ],
			flandmarkLandmarks[ 3 ] );
	/*
	mouthRight_Flandmark = cv::Point( flandmarkLandmarks[ 8 ],
			flandmarkLandmarks[ 9 ] );
	mouthLeft_Flandmark = cv::Point( flandmarkLandmarks[ 6 ],
			flandmarkLandmarks[ 7 ] );
	*/
	retVector.push_back( leftEyeLeftCorner_Flandmark );
	retVector.push_back( rightEyeRightCorner_Flandmark );
	
	cv::circle ( tmp,
			leftEyeLeftCorner_Flandmark,
			2,
			CV_RGB( 0, 0, 255 ),
			2
	);
	
	cv::imshow( "eye-det-area", tmp );
	/*
	cv::circle ( imgGrayscale ,
			rightEyeRightCorner_Flandmark,
			2,
			CV_RGB( 0, 255, 255 ),
			2
	);
	cv::imshow ( "Flandmark", imgGrayscale );
	*/
	
    delete flandmarkLandmarks;
    delete flandmarkModel;	
    return retVector;
}

void Main::doWork()
{
	Trajectory trajectory;
	//usleep( 1000 );
    IplImage *img = imAcqGetImg( imAcq );
    std::vector<cv::Point> flandmarkVector;
	cv::Point pupil;
	
    int loops = 50; // skip the first 50 frames of the video.
    for ( int counter = 0; counter <= loops; counter++ )
    {
    	img = imAcqGetImg( imAcq );
    }
    Mat grey( img->height, img->width, CV_8UC1 );
    cvtColor( cv::Mat( img ), grey, CV_BGR2GRAY );

    tld->detectorCascade->imgWidth = grey.cols;
    tld->detectorCascade->imgHeight = grey.rows;
    tld->detectorCascade->imgWidthStep = grey.step;
	
	if( showTrajectory )
	{
		trajectory.init( trajectoryLength );
	}

    if( selectManually )
    {

        CvRect box;

        if( getBBFromUser( img, box, gui ) == PROGRAM_EXIT )
        {
            return;
        }

        if( initialBB == NULL )
        {
            initialBB = new int[ 4 ];
        }

        initialBB[ 0 ] = box.x;
        initialBB[ 1 ] = box.y;
        initialBB[ 2 ] = box.width;
        initialBB[ 3 ] = box.height;
    }

	// initialBB is not being set when there is no parameter -b given
	doHaar( grey );
	
    
    
    FILE *resultsFile = NULL;

    if( printResults != NULL )
    {
        resultsFile = fopen( printResults, "w" );
    }

    bool reuseFrameOnce = false;
    bool skipProcessingOnce = false;


	
	
    if( loadModel && modelPath != NULL )
    {
        tld->readFromFile(modelPath);
        reuseFrameOnce = true;
    }
    else if( initialBB != NULL )
    {
        Rect bb = tldArrayToRect( initialBB );

        printf( "Starting at %d %d %d %d\n",
				bb.x,
				bb.y,
				bb.width,
				bb.height
		);

        tld->selectObject( grey, &bb );
        skipProcessingOnce = true;
        reuseFrameOnce = true;
    }
    
    // deleting initialBB to avoid memLeaks
    delete initialBB;
	cv::Rect faceBB;
	
	eyeTld->detectorCascade->imgWidth = tld->currBB->width;
    eyeTld->detectorCascade->imgHeight = tld->currBB->height;
    eyeTld->detectorCascade->imgWidthStep = grey.step;
    cv::Rect eye = doDetectEye( img );
    
	eye.x += tld->currBB->x;
	eye.y += tld->currBB->y;
	eyeTld->selectObject( grey, &eye );
	
	//cv::Point pupil = cv::Point( 0, 0 );
	
    while( imAcqHasMoreFrames( imAcq ) )
    {
        double tic = cvGetTickCount();

        if( !reuseFrameOnce )
        {
            img = imAcqGetImg( imAcq );

            if( img == NULL )
            {
                printf( "current image is NULL, assuming end of input.\n" );
                break;
            }

            cvtColor( cv::Mat( img ), grey, CV_BGR2GRAY );
        }

        if( !skipProcessingOnce )
        {
            tld->processImage( img );
            eyeTld->processImage( img );
        }
        else
        {
            skipProcessingOnce = false;
        }
        faceBB = cv::Rect( tld->currBB->x,
        		tld->currBB->y,
        		tld->currBB->width,
        		tld->currBB->height
        );
        std::cout << "bb [ " <<
	        	tld->currBB->x << ", " <<
	        	tld->currBB->y << ", " <<
	        	tld->currBB->width << ", " <<
	        	tld->currBB->height << "]" <<
	        	std::endl;
		pupil = findPupil( img, cv::Rect( eyeTld->currBB->x,
				eyeTld->currBB->y,
				eyeTld->currBB->width,
				eyeTld->currBB->height
		) );
		std::cout << "pupil [ " <<
				pupil.x <<
				", " <<
				pupil.y <<
				" ]" <<
				std::endl;
				
	        	
		/*
        flandmarkVector = doFlandmark( *img, faceBB );
        if ( flandmarkVector.size() == 2 )
        {
        		std::cout << " [ " << 
        				flandmarkVector[ 0 ].x << ", " << 
        				flandmarkVector[ 0 ].y << " ]";
        		std::cout << " -- [ " << 
        				flandmarkVector[ 1 ].x << ", " << 
        				flandmarkVector[ 1 ].y << " ]" <<
        				std::endl;

				if ( flandmarkVector[ 0 ].x > 0 && 
						flandmarkVector[ 0 ].y > 0 && 
						flandmarkVector[ 0 ].x < tld->detectorCascade->imgWidth &&
						flandmarkVector[ 0 ].y < tld->detectorCascade->imgHeight
				)
				{
					cvCircle ( img,
							flandmarkVector[ 0 ],
							4,
							CV_RGB( 0, 111, 255 ),
							4
					);
				}
				if ( flandmarkVector[ 1 ].x > 0 && 
						flandmarkVector[ 1 ].y > 0 &&
						flandmarkVector[ 1 ].x < tld->detectorCascade->imgWidth &&
						flandmarkVector[ 1 ].y < tld->detectorCascade->imgHeight
				)
				{
					cvCircle ( img,
							flandmarkVector[ 1 ],
							4,
							CV_RGB( 0, 111, 255 ),
							4
					);
				}
        }
        else
        {
        	std::cout << "Flandmark vector is of size " <<
        			flandmarkVector.size() << std::endl;
        }
		*/
		
		/*
		printf( "eye: %d %d %d %d\n",
				eye.x,
				eye.y,
				eye.width,
				eye.height
		);
		*/
		//eye.width += tld->currBB->width;
		//eye.height += tld->currBB->height;
		

        if( printResults != NULL )
        {
            if( tld->currBB != NULL )
            {
                fprintf( resultsFile,
                		"%d %.2d %.2d %.2d %.2d %f\n",
                		imAcq->currentFrame - 1,
                		tld->currBB->x,
                		tld->currBB->y,
                		tld->currBB->width,
                		tld->currBB->height,
                		tld->currConf
                );
            }
            else
            {
                fprintf( resultsFile,
                		"%d NaN NaN NaN NaN NaN\n",
                		imAcq->currentFrame - 1
                );
            }
        }

        double toc = ( cvGetTickCount() - tic ) / cvGetTickFrequency();

        toc = toc / 1000000;

        float fps = 1 / toc;

        int confident = (tld->currConf >= threshold) ? 1 : 0;

        if( showOutput || saveDir != NULL )
        {
            char string[128];
            char learningString[10] = "";

            if( tld->learning && !eyeTld->learning)
            {
                strcpy( learningString, "Learning+-" );
            }
            else if( tld->learning && eyeTld->learning )
            {
                strcpy( learningString, "Learning++" );
            }
            else if( !tld->learning && eyeTld->learning )
            {
                strcpy( learningString, "Learning-+" );
            }
            
            

            sprintf( string,
            		"#%d,Posterior %.2f; fps: %.2f, #numwindows:%d, %s",
            		imAcq->currentFrame - 1,
            		tld->currConf,
            		fps,
            		tld->detectorCascade->numWindows,
            		learningString
            );
            CvScalar yellow = CV_RGB( 255, 255, 0 );
            CvScalar blue = CV_RGB( 0, 0, 255 );
            CvScalar black = CV_RGB( 0, 0, 0 );
            CvScalar white = CV_RGB( 255, 255, 255 );

            if( tld->currBB != NULL && eyeTld->currBB != NULL )
            {
                CvScalar rectangleColor = ( confident ) ? blue : yellow;
                cvRectangle( img,
                		tld->currBB->tl(),
                		tld->currBB->br(),
                		rectangleColor,
                		8,
                		8,
                		0
                );
                
                CvScalar rectangleColor2 = ( confident ) ? blue : yellow;
                cvRectangle( img,
                		eyeTld->currBB->tl(),
                		eyeTld->currBB->br(),
                		rectangleColor2,
                		8,
                		8,
                		0
                );
                /*
				pupil = findPupil( img, cv::Rect( eyeTld->currBB->x, eyeTld->currBB->y, eyeTld->currBB->width, eyeTld->currBB->height ) );
				cvRectangle( img,
						cv::Point( pupil.x, pupil.y ),
						cv::Point( pupil.x + 2, pupil.y + 2 ),
						2,
						2,
						0
				);
				*/
						
                /*
                if ( eye.x > 0 && 
						eye.y > 0 &&
						eye.width > 0 &&
						eye.height > 0
				)
                {						
					cvRectangle( img,
							cv::Point( eye.x, eye.y ),
							cv::Point( eye.x + eye.width,
									eye.y + eye.height
							),
							rectangleColor,
							8,
							8,
							0
					);
				}*/

				if( showTrajectory )
				{
					CvPoint center = cvPoint( 
							tld->currBB->x + tld->currBB->width / 2,
							tld->currBB->y + tld->currBB->height / 2
					);
					cvLine( img,
							cvPoint( center.x - 2, center.y - 2 ),
							cvPoint( center.x + 2, center.y + 2 ),
							rectangleColor,
							2
					);
					cvLine( img,
							cvPoint( center.x - 2, center.y + 2),
							cvPoint( center.x + 2, center.y - 2),
							rectangleColor,
							2
					);
					trajectory.addPoint( center, rectangleColor );
				}
            }
			else if( showTrajectory )
			{
				trajectory.addPoint( cvPoint( -1, -1 ),
						cvScalar( -1, -1, -1 ) 
				);
			}

			if( showTrajectory )
			{
				trajectory.drawTrajectory( img );
			}

            CvFont font;
            cvInitFont( &font, CV_FONT_HERSHEY_SIMPLEX, .5, .5, 0, 1, 8 );
            // display a black bar at the top of the output window
            // for a better background to write on.
            cvRectangle( img,
            		cvPoint( 0, 0 ),
            		cvPoint( img->width, 50 ),
            		black,
            		CV_FILLED,
            		8,
            		0
            );
            cvPutText( img, string, cvPoint( 25, 25 ), &font, white );

            if( showForeground )
            {
                for( size_t i = 0;
                		i < tld->detectorCascade->detectionResult->fgList->size();
                		i++
                )
                {
                    Rect r = tld->detectorCascade->detectionResult->fgList->at( i );
                    cvRectangle( img, r.tl(), r.br(), white, 1 );
                }

            }


            if( showOutput )
            {
                gui->showImage( img );
                char key = gui->getKey();

                if( key == 'q' )
                {
                	break;
                }

                if( key == 'b' )
                {
                    ForegroundDetector *fg = 
                    		tld->detectorCascade->foregroundDetector;

                    if( fg->bgImg.empty() )
                    {
                        fg->bgImg = grey.clone();
                    }
                    else
                    {
                        fg->bgImg.release();
                    }
                }

                if( key == 'c' )
                {
                    //clear everything
                    tld->release();
                }
                
                if( key == 'l' )
                {
                    tld->learningEnabled = !tld->learningEnabled;
                    printf( "LearningEnabled: %d\n", tld->learningEnabled );
                }

                if( key == 'a' )
                {
                    tld->alternating = !tld->alternating;
                    printf( "alternating: %d\n", tld->alternating );
                }

                if( key == 'e' )
                {
                    tld->writeToFile( modelExportFile );
                }

                if( key == 'i' )
                {
                    tld->readFromFile( modelPath );
                }

                if( key == 'r' )
                {
                    CvRect box;

                    if( getBBFromUser( img, box, gui ) == PROGRAM_EXIT )
                    {
                        break;
                    }

                    Rect r = Rect( box );

                    tld->selectObject( grey, &r );
                }
            }

            if( saveDir != NULL )
            {
                char fileName[ 256 ];
                sprintf( fileName, 
                		"%s/%.5d.png",
                		saveDir,
                		imAcq->currentFrame - 1
                );

                cvSaveImage( fileName, img );
            }
        }

        if( !reuseFrameOnce )
        {
            cvReleaseImage( &img );
        }
        else
        {
            reuseFrameOnce = false;
        }
    }

    if( exportModelAfterRun )
    {
        tld->writeToFile( modelExportFile );
    }
    //FB: Avoiding leakage
	if( printResults != NULL )
    {
		fclose( resultsFile );
    }
}
