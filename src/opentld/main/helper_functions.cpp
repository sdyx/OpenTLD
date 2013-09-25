#include <assert.h>     /* assert */
#include <limits.h>
#include <float.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/video/video.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/video/tracking.hpp>
#include "opencv2/core/core.hpp"
#include <iostream>
#include "clean_functions_array.h"
#include <vector>
#include "flandmark_detector.h"

extern int GLOBAL_COUNT;
extern const int ARRAY_SIZE;
extern const double PI;

struct eye {
	//cv::Point pupilTrace [ ARRAY_SIZE ];
	//cv::Point cornerTrace [ ARRAY_SIZE ];
	std::vector< cv::Point > pupilTrace;
	std::vector< cv::Point > cornerTrace;
	cv::Point currentPupil;
	cv::Point currentCorner;
	double maxXDeviation,
			minXDeviation,
			maxYDeviation,
			minYDeviation;
	cv::Rect searchArea;
	cv::Point associatedCornerFlandmark;
	double mean;
};

struct face {
	cv::Rect boundingBox;
	cv::Point center;
	cv::Point centerPoint;
	cv::Point referencePoint;
	double mean;
//	std::vector< cv::Point > facialLandmarks;
	cv::Point flandmarkNose;
	cv::Point flandmarkMouthL;
	cv::Point flandmarkMouthR;
	cv::Point flandmarkLLEye;
	cv::Point flandmarkLREye;
	cv::Point flandmarkRLEye;
	cv::Point flandmarkRREye;
}

struct datapack {
	int counter;
	int frame;
	int startAfter;
	FLANDMARK_Model * fModel;
	double * landmarks;
}

datapack DATA;

eye LEFT_EYE, RIGHT_EYE;
face USER_FACE;

//cv::Point leftPupil[ ARRAY_SIZE ];
//cv::Point leftCorner[ ARRAY_SIZE ];

//cv::Point rightPupil[ ARRAY_SIZE ];
//cv::Point rightCorner[ ARRAY_SIZE ];

//cv::Point centerPoint[ ARRAY_SIZE ];
//cv::Point referencePoint[ ARRAY_SIZE ];

std::vector< cv::Point > leftPupil;
std::vector< cv::Point > leftCorner;

std::vector< cv::Point > rightPupil;
std::vector< cv::Point > rightCorner;

std::vector< cv::Point > centerPoint;
std::vector< cv::Point > referencePoint;


cv::Point currentLeftPupil, 
		currentRightPupil,
		currentLeftCorner,
		currentRightCorner;

double leftMinXVal, leftMaxXVal, leftMinYVal, leftMaxYVal;
double rightMinXVal, rightMaxXVal, rightMinYVal, rightMaxYVal;
double centerMinXVal, centerMaxXVal, centerMinYVal, centerMaxYVal;

double threshold = 0.4;



int verticalScreenResolution = 1080;
int horizontalScreenResolution = 1920;
int verticalCameraResolution = 480;
int horizontalCameraResolution = 640;

cv::Point screenCenter = cv::Point( verticalScreenResolution / 2,
		horizontalScreenResolution / 2 );
//cv::Point onScreenPointer[ ARRAY_SIZE ];
std::vector< cv::Point > onScreenPointer;
//cv::Point gazePointer[ ARRAY_SIZE ];
std::vector< cv::Point > gazePointer;
cv::Point currentOnScreenPointer;
cv::Point currentGazePointer;




kalman kFilterEyes;

double mean;
//double movingMean[ ARRAY_SIZE ];
std::vector< double> movingMean;
cv::RotatedRect pupilBox;


void increaseCounter()
{
	//if ( GLOBAL_COUNT == ARRAY_SIZE )
	if ( DATA.counter == ARRAY_SIZE )
	{
		//GLOBAL_COUNT == 0;
		DATA.counter = 0;
	}
	else
	{
		//GLOBAL_COUNT++;
		DATA.counter++;
	}
}
void initialize()
{
	leftMinXVal = leftMinYVal = rightMinXVal = rightMinYVal = DBL_MAX;
	leftMaxXVal = leftMaxYVal = rightMaxXVal = rightMaxYVal = DBL_MIN;
	centerMinXVal = centerMinYVal = DBL_MAX;
	centerMaxXVal = centerMaxYVal = DBL_MIN;
	
	DATA.startAfter = 50;
	DATA.frame = 0;
	DATA.counter = 0;
	DATA.fModel = flandmark_init( "flandmark_model.dat" );
	DATA.landmarks = (double*) malloc( 
				2 * DATA.fModel->data.options.M * sizeof( double )
	);
}

void workflow()
{
	initialize();
	aquireImage();
	preprocessImage( image );
	findFace( image, USER_FACE.boundingBox );

	//USER_FACE.boundingBox = boundingBoxFace;

	findEyes( image,
			USER_FACE.boundingBox,
			LEFT_EYE.searchArea, RIGHT_EYE.searchArea );

	findPupil( image, LEFT_EYE.searchArea, LEFT_EYE.currentPupil );
	findPupil( image, RIGHT_EYE.searchArea, RIGHT_EYE.currentPupil );
	findCorner( image, LEFT_EYE.searchArea, LEFT_EYE.currentCorner );
	findCorner( image, RIGHT_EYE.searchArea, RIGHT_EYE.currentCOrner );
	
	updatePoint( LEFT_EYE.currentPupil, LEFT_EYE.pupilTrace );
	updatePoint( RIGHT_EYE.currentPupil, RIGHT_EYE.pupilTrace );
	updatePoint( LEFT_EYE.currentCorner, LEFT_EYE.cornerTrace );
	updatePoint( RIGHT_EYE.currentCorner, RIGHT_EYE.cornerTrace );

	//TODO
	calculateCenterPoint( LEFT_EYE.currentPupil,
			LEFT_EYE.currentCorner,
			RIGHT_EYE.currentPupil,
			RIGHT_EYE.currentCorner,
			USER_FACE.centerPoint,
			USER_FACE.referencePoint
	);

	calculateMinMaxXY( LEFT_EYE.pupilTrace,
			LEFT_EYE.cornerTrace,
			LEFT_EYE.minXDeviation,
			LEFT_EYE.maxXDeviation,
			LEFT_EYE.minYDeviation,
			LEFT_EYE.maxYDeviation
	);

	calculateMinMaxXY( RIGHT_EYE.pupilTrace,
			RIGHT_EYE.cornerTrace,
			RIGHT_EYE.minXDeviation,
			RIGHT_EYE.maxXDeviation,
			RIGHT_EYE.minYDeviation,
			RIGHT_EYE.maxYDeviation
	);

	calculateMinMaxXY( USER_FACE.centerPoint,
			USER_FACE.referencePoint,
			centerMinXVal,
			centerMaxXVal,
			centerMinYVal,
			centerMaxYVal
	);
	mapGazeToScreen( gazePointer,
		onScreenPointer,
		centerMaxXVal,
		centerMinXVal,
		centerMaxYVal,
		centerMinYVal,
		horizontalScreenResolution,
		verticalScreenResolution
	);
	//TODO
	//printStatistics();
	//TODO
	//driveMouse();

	cv::destroyAllWindows();
}

//void updatePoint( const cv::Point &currentPoint,
//		std::vector< cv::Point > &previousPoints
//)
void updatePoint( const cv::Point &currentPoint,
		std::vector< cv::Point > previousPoints
)
{
	int weightFactor = 4;
	double threshold = 0.4;

	// if the left pupil is at a very different position than it was before, we
	// think this might be an error and do not take this value but weight it down.
	// Otherwise we will put the value into a rotating list.
	int select;

	if ( DATA.counter == 0 )
	{
		select = previousPoints.size() - 1;
	}
	else
	{
		select = DATA.counter - 1;
	}

	if ( ( previousPoints[ select ].x * ( 1 + threshold ) > currentPoint.x ) ||
			( previousPoints[ select ].x * ( 1 - threshold ) < currentPoint.x ) ||
			( previousPoints[ select ].y * ( 1 + threshold ) > currentPoint.y ) ||
			( previousPoints[ select ].y * ( 1 - threshold ) < currentPoint.y )
	)
	{
		previousPoints[ DATA.counter ].x = ( 
				previousPoints[ select ].x * weightFactor
				+ currentPoint.x ) / ( weightFactor + 1 );
				
		previousPoints[ DATA.counter ].y = ( 
				previousPoints[ select ].y * weightFactor
				+ currentPoint.y ) / ( weightFactor + 1 );
	}
	else
	{
		previousPoints[ DATA.counter ] = currentPoint;
	}
}



void calculateMinMaxXY( const std::vector< cv::Point > &pupilArray,
		const std::vector< cv::Point > &referenceArray,
		double &minXVal,
		double &maxXVal,
		double &minYVal,
		double &maxYVal
)
{
	double distanceX, distanceY;
	double avgDistanceX, avgDistanceY;

	assert( pupilArray.size() == referenceArray.size() );

	for ( int c = 0; c < pupilArray.size(); c++ )
	{
		distanceY = std::abs( pupilArray[ DATA.counter ].y
				- referenceArray[ DATA.counter ].y
		);
		distanceX = std::abs( distanceX = pupilArray[ DATA.counter ].x 
				- referenceArray[ DATA.counter ].x
		);
		avgDistanceX += distanceX;
		avgDistanceY += distanceY;
	}
	avgDistanceX /= pupilArray.size();
	avgDistanceY /= pupilArray.size();
	
	if ( avgDistanceX > maxXVal )
	{
		maxXVal = avgDistanceX;
	}

	if ( avgDistanceX < minXVal )
	{
		minXVal = avgDistanceX;
	}

	if ( avgDistanceY > maxYVal )
	{
		maxYVal = avgDistanceY;
	}

	if ( avgDistanceY < minYVal )
	{
		minYVal = avgDistanceY;
	}
} 


void mapGazeToScreen( const std::vector< cv::Point > &gaze,
		std::vector< cv::Point > &onScreen,
		const double maxX,
		const double minX,
		const double maxY,
		const double minY,
		const int screenX,
		const int screenY
)
{
	double xScale, yScale;
	xScale = maxX - minX;
	yScale = maxY - minY;

	onScreen[ DATA.counter ].x = ( gaze[ DATA.counter ].x / xScale )
			* screenX;
	onScreen[ DATA.counter ].y = ( gaze[ DATA.counter ].y / yScale )
			* screenY;
}


void findClosestPair( const std::vector< cv::Point2f > &arrayA,
		const std::vector< cv::Point2f > &arrayB,
		double &minDistance,
		int &arrayAIndex,
		int &arrayBIndex
)
{
	double minDist = DBL_MAX;
	int indexA = -1;
	int indexB = -1;
	int faDist, fbDist;
	double fcDist;

	for ( int i = 0; i <= arrayA.size(); i++ )
	{
		if (( arrayA[ i ].x == 0 ) || ( arrayA[ i ].y == 0 ))
		{
			continue;
		}
		for ( int a = 0; a <= arrayB.size(); a++)
		{
			// skip if the value (x or y) lies on the border.
			if (( arrayB[ a ].x == 0 ) || ( arrayB[ a ].y == 0 ))
			{
				continue;
			}
			faDist = std::abs( arrayA[ i ].x - arrayB[ a ].x );
			fbDist = std::abs( arrayA[ i ].y - arrayB[ a ].y );
			
			fcDist = sqrt( faDist * faDist + fbDist * fbDist );
			if ( fcDist < minDist )
			{
				indexA = i;
				indexB = a;
				minDist = fcDist;
			}
		}
	}
	minDistance = minDist;
	arrayAIndex = indexA;
	arrayBIndex = indexB;

	//std::cout << "findClosestPair:\n\tthe minimal distance was " 
	//		<< minDistance << " between indexA " << arrayAIndex 
	//		<< " and indexB " << arrayBIndex << " with point A "
	//		<< arrayA[ arrayAIndex ] << " and point B "
	//		<< arrayB[ arrayBIndex ] << std::endl;

	assert( indexA != -1 && indexB != -1 );
	//if (( indexA == -1 ) || ( indexB == -1 ))
	//{
	//	std::cout << "Could not find a suitable match (proximity)" << std::endl;
	//	//cv::rectangle( videoFrame, cv::Rect( fLeft.x, fLeft.y, 2, 2 ), cv::Scalar( 133, 0, 200) );
	//}
	//else
	//{
	//	//cv::rectangle( videoFrame, cv::Rect( ( corners[ minIndexF ].x + xOffset + fLeft.x + corners[ LRIndexA ].x + xOffset ) / 3, ( corners[ minIndexF ].y + yOffset + fLeft.y + corners[ LRIndexA ].y + yOffset) / 3, 1, 1 ), cv::Scalar( 233, 111, 20) );
	//	leftEstimate.x = ( leftEstimate.x + corners[ LRIndexA ].x  ) / 2;
	//	leftEstimate.y = ( leftEstimate.y + corners[ LRIndexA ].y  ) / 2;
	//}
}



float calculateDistance( const cv::Point &pointA,
		const cv::Point &pointB 
)
{
	float deltaX, deltaY, distance;
	deltaX = std::abs( pointA.x - pointB.x );
	deltaY = std::abs( pointA.y - pointB.y );

	distance = std::sqrt( std::pow( deltaX, 2 )
			+ std::pow( deltaY, 2 )
	);
	return distance;
}

float calculateAngle( const cv::Point &pointA,
		const cv::Point &pointB 
)
{
	float deltaX, deltaY, distance;
	float angleAlpha;

	deltaX = std::abs( pointA.x - pointB.x );
	deltaY = std::abs( pointA.y - pointB.y );

	//std::cout << " calculating points " << v1 <<
	//		", " << v2 << std::endl;
	//std::cout << " deltaX " << deltaX <<
	//		" deltaY " << deltaY << std::endl;

	distance = std::sqrt(
			std::pow( deltaX, 2 ) + std::pow( deltaY, 2 )
	);
	//std::cout << " distance is " << distance << std::endl;
	angleAlpha = std::acos(
			( std::pow( deltaX, 2 )
					+ std::pow( distance, 2 )
					- std::pow( deltaY, 2 )
			) / ( 2 * deltaX * distance ) ) * 180.0 / PI; 
	return angleAlpha;
}

//void update2DKalman( cv::KalmanFilter &kalmanFilter,
//		cv::Point &updatetablePoint )
void initialize2DKalman( kalman &kalmanFilter )
{
	kalmanFilter.KF.init( 4, 2, 0 );
	kalmanFilter.kalmanMeasurement = cv::Mat_< float > ( 2, 1 );
	//kalman.kalmanMeasurement;// = cv::Mat_< float >( 2, 1 );
	kalmanFilter.KF.transitionMatrix = *( cv::Mat_< float >( 4, 4 ) <<
			1, 0, 1, 0,
			0, 1, 0, 1,  
			0, 0, 1, 0,
			0, 0, 0, 1
	);
	kalmanFilter.kalmanMeasurement.setTo( cv::Scalar( 0 ) );

	kalmanFilter.KF.statePre.at< float >( 0 ) = 
			horizontalCameraResolution / 2;
	kalmanFilter.KF.statePre.at< float >( 1 ) =
			verticalCameraResolution / 2;
	kalmanFilter.KF.statePre.at< float >( 2 ) = 0;
	kalmanFilter.KF.statePre.at< float >( 3 ) = 0;
	
	cv::setIdentity( kalmanFilter.KF.measurementMatrix );
	cv::setIdentity( kalmanFilter.KF.processNoiseCov,
			cv::Scalar::all( 1e-4 )
	);
	cv::setIdentity( kalmanFilter.KF.measurementNoiseCov,
			cv::Scalar::all( 1e-1 )
	);
	cv::setIdentity( kalmanFilter.KF.errorCovPost,
			cv::Scalar::all( .1 )
	);

}
void update2DKalman( kalman &kalmanFilter, cv::Point &updatetablePoint )
{
	//predictionL = kalmanFilter.predict();
	kalmanFilter.kalmanPrediction = kalmanFilter.KF.predict();
	kalmanFilter.kalmanMeasurement( 0 ) = updatetablePoint.x;
	kalmanFilter.kalmanMeasurement( 1 ) = updatetablePoint.y;
	kalmanFilter.kalmanEstimated = 
			kalmanFilter.KF.correct( kalmanFilter.kalmanMeasurement );
	updatetablePoint = cv::Point(
			kalmanFilter.kalmanEstimated.at< float >( 0 ), 
			kalmanFilter.kalmanEstimated.at< float >( 1 )
	);
}


void calculateMovingMean( const cv::Mat &image,
		std::vector< double > &movingMean
)
{
	// TODO: decide which channel to chose from
	movingMean[ DATA.counter ] = cv::mean( image )[ 0 ];
}


void findPupilByEllipse( const cv::Mat &image,
		const cv::Rect &searchArea,
		cv::RotatedRect &boxAroundPupil
)
{
	cv::Mat imageCopy = image( searchArea ).clone();
	cv::Mat whiteMatrix = imageCopy.clone();
	cv::RotatedRect box;
	std::vector< std::vector< cv::Point > > ellipseContour;
	int lowerThreshold = 120;
	int lowerBoxThresholdHeight = 4;
	int lowerBoxThresholdWidth = 4;
	int upperBoxThresholdHeight = 30;
	int upperBoxThresholdWidth = 50;
	int ellipseRatio = 30;
	int darknessIndex;
	double minDarkness = DBL_MAX;
	double curDarkness;

	whiteMatrix.setTo( cv::Scalar( 0, 0, 0 ) );
	cv::threshold( imageCopy,
			imageCopy,
			lowerThreshold,
			255,
			CV_THRESH_BINARY
	);
	// invert the original image
	cv::bitwise_xor( imageCopy, whiteMatrix, imageCopy);
	cv::dilate( imageCopy,
			imageCopy,
			cv::Mat( cv::Size( 5, 5 ),
			CV_8UC1
			),
			cv::Point( 3, 3 )
	);
	
	cv::findContours( imageCopy,
			ellipseContour,
			CV_RETR_CCOMP,
			CV_CHAIN_APPROX_SIMPLE
	);

	//?
	//oldBox.size.width = 10;

	for( size_t m = 0; m < ellipseContour.size(); m++ )
	{			
		cv::Mat pointsf;
		cv::Mat( ellipseContour[ m ]).convertTo( pointsf, CV_32F );
		size_t ellipseCount = ellipseContour[ m ].size();
		// because with 5 or less points fitEllipse will fail
		if( ellipseCount < 6 )
		{
			continue;
		}

		box = cv::fitEllipse( pointsf );

		// if the box does not have a roughly elliptical shape
		if( MAX( box.size.width, box.size.height ) > 
				MIN( box.size.width, box.size.height ) * ellipseRatio )
		{
			continue;
		}
		
		if ( ( box.size.width >= upperBoxThresholdWidth ) ||
				( box.size.height >= upperBoxThresholdHeight)
		)
		{
			continue;
		}
		
		if ( ( box.size.width <= lowerBoxThresholdWidth ) || 
				( box.size.height <= lowerBoxThresholdHeight )
		)
		{
			continue;
		}

		//cv::ellipse( imageCopy,
		//		box.center,
		//		box.size * 0.5f,
		//		box.angle,
		//		0,
		//		360,
		//		cv::Scalar( 0, 255, 255 ),
		//		1,
		//		CV_AA
		//);

		// TODO: Select which ellipse we should return!
		//boxAroundPupil = box;
		//boxAroundPupil.center.x = box.center.x + searchArea.x;
		//boxAroundPupil.center.y = box.center.y + searchArea.y;
		
		curDarkness = calculateMeanOfEllipse( imageCopy,
				box.center );
		if ( curDarkness < minDarkness )
		{
				minDarkness = curDarkness;
				darknessIndex = i;
				boxAroundPupil = box;
				boxAroundPupil.center = cv::Point( 
						cvRound( circles[ darknessIndex ][ 0 ] )
								+ searchArea.x,
						cvRound( circles[ darknessIndex ][ 1 ] )
								+ searchArea.y
				);
		}
	}
	imageCopy.release();
	whiteMatrix.release();
}

void findPupilByCircle( const cv::Mat &image,
		const cv::Rect &searchArea,
		cv::Point &pupilCenter
)
{
	//cv::Mat compare;
	std::vector<std::vector<cv::Point> > contours;
	cv::Mat workingImage = image( searchArea ).clone();
	cv::Mat contourImage(
			workingImage.size(), CV_8UC1, cv::Scalar( 0, 0, 0 )
	);
	std::vector< cv::Vec3f > circles;
	cv::Point center;
	int radius;
	int darknessIndex;
	double minDarkness = DBL_MAX;
	double curDarkness;

	cv::erode( workingImage,
			workingImage,
			cv::Mat( cv::Size( 5, 5 ),
			CV_8UC1
			),
			cv::Point( 3, 3 )
	);
	cv::dilate( workingImage,
			workingImage,
			cv::Mat( cv::Size( 5, 5 ),
			CV_8UC1 ),
			cv::Point( 3, 3 )
	);

	//cv::Canny( workingImage, compare, 20, 20 * 3, 3 );
	cv::findContours( workingImage.clone(),
			contours,
			CV_RETR_EXTERNAL,
			CV_CHAIN_APPROX_NONE
	);

	for ( size_t idx = 0; idx < contours.size(); idx++)
	{
		cv::drawContours( contourImage,
				contours,
				idx,
				cv::Scalar( 255, 10, 100 ),
				CV_FILLED
		);
	}

	HoughCircles( contourImage,
			circles,
			CV_HOUGH_GRADIENT,
			2,
			horizontalCameraResolution / 2, 	// Minimum distance between detected centers
			10, 							// Upper threshold for the internal Canny edge detector
			30,								// Minimum distance between detected centers
			4,								// Minimum radio to be detected
			100								// Maximum radius to be detected
	);

	for( size_t i = 0; i < circles.size(); i++ )
	{
		center = cv::Point( cvRound( circles[ i ][ 0 ] ) + searchArea.x,
				cvRound( circles[ i ][ 1 ] ) + searchArea.y
		);
		radius = cvRound( circles[ i ][ 2 ] );
		
		//prediction = KFPR.predict();

		//measurementPR( 0 ) = centerR.x;
		//measurementPR( 1 ) = centerR.y;

		//estimatedPR = KFPR.correct( measurementPR );
		//center = cv::Point( estimatedPR.at<float>( 0 ), 
		//		estimatedPR.at<float>( 1 ) );
		
		// TODO: Select which circle we should return!
		// biggest?
		// the one that is the blackest? -> mean
		pupilCenter = center;
		curDarkness = calculateMeanOfCircle( workingImage,
				cv::Point( circles[ i ][ 0 ],
						circles[ i ][ 1 ] ),
				radius
		);
		if ( curDarkness < minDarkness )
		{
				minDarkness = curDarkness;
				darknessIndex = i;
		}
	}
	pupilCenter = cv::Point( 
			cvRound( circles[ darknessIndex ][ 0 ] ) + searchArea.x,
			cvRound( circles[ darknessIndex ][ 1 ] ) + searchArea.y
	);
	workingImage.release();
	contourImage.release();
}


void findClosestPoint( const std::vector< cv::Point2f > &array,
		const cv::Point &referencePoint,
		int &arrayIndex
)
{
	std::vector< cv::Point2f > dummyVector;
	//dummyVector.push_back( ( cv::Point2f ) referencePoint );
	dummyVector.push_back( referencePoint );
	double distance;
	int arrayIndexA, arrayIndexB;
	
	findClosestPair( array, 
		dummyVector,
		distance,
		arrayIndexA,
		arrayIndexB
	);
	arrayIndex = arrayIndexA;
}
void findEyeCorner( const cv::Mat &image,
		const cv::Rect &searchArea,
		cv::Point &eyeCorner,
		cv::Point &flandmarkCorner
)
{
	cv::Mat workingImage = image( searchArea ).clone();
	std::vector< cv::Point2f > corners;
	int index, darknessIndex;
	double minDarkness = DBL_MAX;
	double curDarkness;
	
	cv::erode( workingImage,
			workingImage,
			cv::Mat( cv::Size( 5, 5 ),
			CV_8UC1
			),
			cv::Point( 3, 3 )
	);
	cv::dilate( workingImage,
			workingImage,
			cv::Mat( cv::Size( 5, 5 ),
			CV_8UC1
			),
			cv::Point( 3, 3 )
	);
	//cv::Mat colorFeatB = rEye.clone();
	cv::cvtColor( workingImage, workingImage, CV_BGR2GRAY );
	cv::equalizeHist( workingImage, workingImage );
	cv::convertScaleAbs( workingImage, workingImage );

	// TODO: check if the second dilation is necessary
	cv::dilate( workingImage,
			workingImage,
			cv::Mat( cv::Size( 5, 5 ),
			CV_8UC1 ),
			cv::Point( 3, 3 )
	);
	
	cv::goodFeaturesToTrack( workingImage,
			corners,
			9,
			0.06,
			5
	);

	
	// adjust the offset of the search area
	for ( int i = 0; i < corners.size(); i++ )
	{
		curDarkness = calculateMeanOfCircle( workingImage, 
				corners[ i ], 5 );
		corners[ i ].x += searchArea.x;
		corners[ i ].y += searchArea.y;
		// check which corner candidate is the darkest
		if ( curDarkness < minDarkness )
		{
				minDarkness = curDarkness;
				darknessIndex = i;
		}
	}
	//findClosestPair( std::vector< cv::Point2f > &arrayA, 
	//	std::vector< cv::Point2f > &arrayB,
	//	double &minDistance,
	//	int &arrayAIndex,
	//	int &arrayBIndex,
	findClosestPoint( corners, flandmarkCorner, index );
	if ( index == darknessIndex )
	{
		eyeCorner = corners[ index ];
	}
	else
	{
		std::cerr << "The darkest candidate was NOT the "
				<< "one that was closest to the flandmark" << std::endl;
		eyeCorner = corners[ darknessIndex ];
	}
	

	workingImage.release();
}


void convertToHSVAndSplit( const cv::Mat &image,
		cv::Mat &hChannel,
		cv::Mat &sChannel,
		cv::Mat &vChannel
)
{
	std::vector< cv::Mat > channels( 3 );
	cv::Mat workingImage = image.clone();
	// INFO on HSV http://www.aishack.in/2010/07/tracking-colored-objects-in-opencv/
	cv::cvtColor( image, workingImage, CV_BGR2HSV, 3 );
	//cvInRangeS(imgHSV, cvScalar( 20, 100, 100), cvScalar(30, 255, 255), imgThreshed);

	cv::split( workingImage, channels );
	hChannel = channels[ 0 ];
	sChannel = channels[ 1 ];
	vChannel = channels[ 2 ];

	channels[ 0 ].release();
	channels[ 1 ].release();
	channels[ 2 ].release();
	workingImage.release();
}
void increaseFrameCounter()
{
	DATA.frame++;
}
int getFrameCounter()
{
	return DATA.frame;
}
int getCounter()
{
	return DATA.counter;
}
/*
void showStatistics()
{
	std::cout << "#############################" << std::endl;
	double gazeDistA, gazeDistB, gazeDistLeft, gazeDistRight;
	std::cout << "pL " << pupilLeft << std::endl;
	std::cout << "pR " << pupilRight << std::endl;
	std::cout << "corner left " << leftEstimate << std::endl;
	std::cout << "corner right " << rightEstimate << std::endl;
	gazeDistA = pupilLeft.x - leftEstimate.x;
	gazeDistB = pupilLeft.y - leftEstimate.y;
	gazeDistLeft = sqrt( gazeDistA * gazeDistA + gazeDistB * gazeDistB );
	
	// wait for 50 frames before setting the max/min values (at first
	// the pointers are too far apart.
	if ( count > 50 )
	{
		if ( gazeDistLeft < leftMinDist )
		{
			leftMinDist = gazeDistLeft;
		}
		if ( gazeDistLeft > leftMaxDist )
		{
			leftMaxDist = gazeDistLeft;
		}
		std::cout << "left max/min " << leftMaxDist << ", " << leftMinDist << std::endl;
	
		if ( std::abs( pupilLeft.x - leftEstimate.x ) > maxLX )
		{
			maxLX = std::abs( pupilLeft.x - leftEstimate.x );
		}				
		if ( std::abs( pupilLeft.y - leftEstimate.y ) > maxLY )
		{
			maxLY = std::abs( pupilLeft.y - leftEstimate.y );
		}
		if ( std::abs( pupilLeft.x - leftEstimate.x ) < minLX )
		{
			minLX = std::abs( pupilLeft.x - leftEstimate.x );
		}
		if ( std::abs( pupilLeft.y - leftEstimate.y ) < minLY )
		{
			minLY = std::abs( pupilLeft.y - leftEstimate.y );
		}
		///////////////////////////////////////////
		
		
		if ( std::abs( pupilRight.x - rightEstimate.x ) > maxRX )
		{
			maxRX = std::abs( pupilRight.x - rightEstimate.x );
		}				
		if ( std::abs( pupilRight.y - rightEstimate.y ) > maxRY )
		{
			maxRY = std::abs( pupilRight.y - rightEstimate.y );
		}
		if ( std::abs( pupilRight.x - rightEstimate.x ) < minRX )
		{
			minRX = std::abs( pupilRight.x - rightEstimate.x );
		}
		if ( std::abs( pupilRight.y - rightEstimate.y ) < minRY )
		{
			minRY = std::abs( pupilRight.y - rightEstimate.y);
		}

		gazeDistA = pupilRight.x - rightEstimate.x;
		gazeDistB = pupilRight.y - rightEstimate.y;
		gazeDistRight = sqrt( gazeDistA * gazeDistA + gazeDistB * gazeDistB );
	
	
		if ( gazeDistRight < rightMinDist )
		{
			rightMinDist = gazeDistRight;
		}
		if ( gazeDistLeft > rightMaxDist )
		{
			rightMaxDist = gazeDistRight;
		}
		std::cout << "right max/min " << rightMaxDist << ", " << rightMinDist << std::endl;
	}
	else
	{
		count++;
	}
	double pupilDist;
	double cornerDist;
	
	gazeDistA = pupilRight.x - pupilLeft.x;
	gazeDistB = pupilRight.y - pupilLeft.y;
	pupilDist = sqrt( gazeDistA * gazeDistA + gazeDistB * gazeDistB );
	gazeDistA = leftEstimate.x - rightEstimate.x;
	gazeDistB = leftEstimate.y - rightEstimate.y;
	cornerDist = sqrt( gazeDistA * gazeDistA + gazeDistB * gazeDistB );
	std::cout << "distance between pupils (test): " << calcDistance( pupilRight, pupilLeft ) << ", distance between corners " << calcDistance( leftEstimate, rightEstimate ) << std::endl;
	double leftAngle, rightAngle;
	if ( pupilLeft.x - leftEstimate.x != 0 )
	{
		leftAngle = (int) ( std::atan( ( pupilLeft.y - leftEstimate.y )/( pupilLeft.x - leftEstimate.x )) * 180 / PI );
	}
	else
	{
		std::cout << "leftAngle div by zero" << std::endl;
		leftAngle = (int) ( 0 );
	}
	
	if ( pupilRight.x - rightEstimate.x != 0 )
	{
		rightAngle = (int) ( std::atan( ( pupilRight.y - rightEstimate.y )/( pupilRight.x - rightEstimate.x )) * 180 / PI );
	}
	else
	{
		std::cout << "rightAngle div by zero" << std::endl;
		rightAngle = (int) 0;
	}
	
	angleBetweenPupils = ( std::atan( ( pupilRight.y - pupilLeft.y )/( pupilRight.x - pupilLeft.x )) * 180 / PI );
	angleBetweenCorners = ( std::atan( ( rightEstimate.y - leftEstimate.x )/( rightEstimate.x - leftEstimate.x )) * 180 / PI );
	
	middleCorner = cv::Point( ( leftEstimate.x + rightEstimate.x ) / 2, ( leftEstimate.y + rightEstimate.y ) / 2 );
	middlePupil = cv::Point( ( pupilLeft.x + pupilRight.x ) / 2, ( pupilLeft.y + pupilRight.y ) / 2 );
	
	std::cout << "angle between pupils: " << ang( pupilRight, pupilLeft ) << std::endl;

	std::cout << "angle between corners: " << ang( leftEstimate, rightEstimate ) << std::endl;
	
	//double angleC = std::atan2( pupilRight.y - pupilLeft.y, pupilRight.x - pupilLeft.x) * 180.0 / CV_PI;
    //if( angleC < 0) angleC += 360;
	//std::cout << "angle between pupils3: " << angleA << std::endl;
	//std::cout << "angle between pupils4: " << ang(pupilRight, pupilLeft ) << std::endl;
	
	std::cout << "Left eye: " << ang( pupilLeft, leftEstimate ) << " and distance " << calcDistance( pupilLeft, leftEstimate )  << std::endl;
	std::cout << "Right eye: " << ang( pupilRight, rightEstimate ) << " and distance " << calcDistance( pupilRight, rightEstimate ) << std::endl;
	std::cout << "middle pupil " << middlePupil << " middle corner " << middleCorner << std::endl;
	std::cout << "xDist to corner middle " << middlePupil.x - middleCorner.x << " yDist " << middlePupil.y - middleCorner.y << std::endl;
	if ( rightMaxDist > leftMaxDist )
	{
		maxXDist = leftMaxDist;
	}
	else
	{
		maxXDist = rightMaxDist;
	}
	
	if ( rightMinDist < leftMinDist )
	{
		maxYDist = leftMinDist;
	}
	else
	{
		maxYDist = rightMinDist;
	}
	std::cout << "Max dist " << maxXDist << " min dist " << maxYDist << std::endl;
	//( maxXDist - maxYDist )
	std::cout << "minLX " << minLX << " minLY " << minLY << " maxLX " << maxLX << " maxLY " << maxLY << std::endl;
	std::cout << "minRX " << minRX << " minRY " << minRY << " maxRX " << maxRX << " maxRY " << maxRY << std::endl;
	double displacementX = ( 1920 / 2 ) - ( ( middlePupil.x - middleCorner.x ) / std::abs( maxLX - minLX ) ) * 1920;
	double displacementY = ( 1080 / 2 ) - ( ( middlePupil.y - middleCorner.y ) / std::abs( maxLY - minLY ) ) * 1080;
	std::cout << "\t\tx displacement " << displacementX << " y " << displacementY << std::endl;
}

*/
void createFlandmarks( cv::Mat &image,
		cv::Rect &searchArea,
		
)
{
	int bbox[] = { searchArea.x,
			searchArea.y,
			searchArea.x + searchArea.width,
			searchArea.y + searchArea.height
	};

	// detect facial landmarks (output are x, y coordinates of detected landmarks)
	//double * landmarks = landmarks = (double*) malloc( 
	//			2 * DATA.fModel->data.options.M * sizeof( double ) 
	//);
	IplImage ipl_img = image.clone();
	flandmark_detect( &ipl_img, bbox, DATA.fModel, DATA.landmarks );
	//TODO: Destroy flandmarks
}

double calculateMeanOfCircle( const cv::Mat &image,
		const cv::Point &circleCenter,
		const int circleRadius
)
{
	assert( image.type == CV_8UC1 );
	
	cv::Mat maskA = image( cv::Rect( circleCenter.x - circleRadius,
			circleCenter.y - circleRadius,
			circleRadius * 2,
			circleRadius * 2
			) ).clone();
	double mean = -1.0;
	cv::Mat croppedImage = maskA.clone();
	maskA.setTo( cv::Scalar( 0 ) );
	cv::circle( maskA,
			circleCenter,
			circleRadius,
			cv::Scalar( 255, 255, 255 ),
			-1,
			1
	);
	
	cv::bitwise_and( croppedImage, maskA, croppedImage );
	mean = cv::mean( croppedImage )[ 0 ];
	return mean;
	maskA.relase();
	croppedImage.relase();
}


double calculateMeanOfEllipse( const cv::Mat &image,
		const cv::RotatedRect &boundingBox
)
{
	assert( image.type == CV_8UC1 );
	
	cv::Mat maskA = image( boundingBox.boundingRect() ).clone();
	double mean = -1.0;
	//maskA = Mat::zeros( 1, 1, CV_8U );
	cv::Mat croppedImage = maskA.clone();
	maskA.setTo( cv::Scalar( 0 ) );
	cv::ellipse( maskA,
			boundingBox.center,
			boundingBox.size * 0.5f,
			boundingBox.angle,
			0,
			360,
			cv::Scalar( 255, 255, 255 ),
			1,
			CV_AA
	);
	
	cv::bitwise_and( croppedImage, maskA, croppedImage );
	mean = cv::mean( croppedImage )[ 0 ];
	return mean;
	maskA.relase();
	croppedImage.relase();
}


void calculateCornerCenter( const cv::Point &leftCorner,
		const cv::Point &rightCorner,
		cv::Point &cornerCenter
)
{
	cornerCenter = calculateCenter( leftCorner, rightCorner );
}

void calculatePupilCenter(const cv::Point &leftPupil,
		const cv::Point &rightPupil,
		cv::Point &pupilCenter
)
{
	pupilCenter = calculateCenter( leftPupil, rightPupil );
}

cv::Point calculateCenter( const cv::Point &pointA,
		const cv::Point &pointB
)
{
	cv::Point returnPoint = cv::Point(
			cvRound( ( pointA.x + pointB.x ) / 2 ),
			cvRound( ( pointA.y + pointB.y ) / 2 )
	);
	return returnPoint;
}
void calculateCenterPoint( const cv::Point &leftPupil,
		const cv::Point &leftCorner,
		const cv::Point &rightPupil,
		const cv::Point &rightCorner,
		cv::Point cornerCenter,
		cv::Point pupilCenter
)
{
	calculatePupilCenter( leftPupil, rightPupil, pupilCenter );
	calculateCornerCenter( leftCorner, rightCorner, cornerCenter );
}
