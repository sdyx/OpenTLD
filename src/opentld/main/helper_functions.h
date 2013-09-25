#ifndef HELPER_FCT_H
#define HELPER_FCT_H

//#include <opencv2/imgproc/imgproc.hpp>
//#include <opencv2/highgui/highgui.hpp>
//#include <opencv2/objdetect/objdetect.hpp>
//#include <opencv2/video/tracking.hpp>

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/video/video.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/video/tracking.hpp>
#include "opencv2/core/core.hpp"

#include <vector>

// GLOBAL variables
const int ARRAY_SIZE = 5;
int GLOBAL_COUNT = 0;
const double PI = std::atan( 1.0 ) * 4;



struct kalman {
	cv::KalmanFilter KF;//( 4, 2, 0 );
	cv::Mat_< float > kalmanMeasurement;// ( 2, 1 );
	cv::Mat kalmanPrediction;
	cv::Mat kalmanEstimated;
};

void calculateMinMaxXY( //const cv::Point &pupilArray[],
		const std::vector< cv::Point> &pupilArray,
		//const cv::Point &referenceArray[],
		const std::vector< cv::Point> &referenceArray,
		double &minXVal,
		double &maxXVal,
		double &minYVal,
		double &maxYVal
);
void increaseCounter();

void updatePoint( const cv::Point &currentPoint,
		//cv::Point& previousPoints[]
		std::vector< cv::Point > previousPoints
);
/*
void mapGazeToScreen( //const cv::Point &gaze[],
		const std::vector< cv::Point > &ganze,
		//cv::Point &onScreen[],
		const std::vector< cv::Point > &onScreen,
		const double maxX,
		const double minX,
		const double maxY,
		const double minY,
		const int screenX,
		const int screenY
);
*/
void mapGazeToScreen( const std::vector< cv::Point > &gaze,
		std::vector< cv::Point > &onScreen,
		const double maxX,
		const double minX,
		const double maxY,
		const double minY,
		const int screenX,
		const int screenY
);

void findClosestPair( const std::vector< cv::Point2f > &arrayA,
		const std::vector< cv::Point2f > &arrayB,
		double &minDistance,
		int &arrayAIndex,
		int &arrayBIndex
);
float calculateDistance( const cv::Point &pointA, const cv::Point &pointB );
float calculateAngle( const cv::Point &pointA, const cv::Point &pointB );

void initialize2DKalman( kalman &kalmanFilter );
void update2DKalman( kalman &kalmanFilter, cv::Point &updatetablePoint );
void calculateMovingMean( const cv::Mat &image,
		//double &movingMean[]
		const std::vector< double > movingMean
);
void findPupilByEllipse( cv::Mat &image,
		cv::Rect &searchArea,
		cv::RotatedRect &boxAroundPupil
);

void findPupilByCircle( const cv::Mat &image,
		const cv::Rect &searchArea,
		cv::Point &pupilCenter
);
void findClosestPoint( const std::vector< cv::Point2f > &array,
		const cv::Point &referencePoint,
		int &arrayIndex
);
void findEyeCorner( const cv::Mat &image,
		const cv::Rect &searchArea,
		cv::Point &eyeCorner,
		cv::Point &flandmarkCorner
);
void convertToHSVAndSplit( const cv::Mat &image,
		cv::Mat &hChannel,
		cv::Mat &sChannel,
		cv::Mat &vChannel
);

void increaseFrameCounter();
int getFrameCounter();
int getCounter();
double calculateMeanOfCircle( const cv::Mat &image,
		const cv::Point &circleCenter,
		const int circleRadius
);
double calculateMeanOfEllipse( const cv::Mat &image,
		const cv::RotatedRect &boundingBox
);

void calculateCornerCenter( const cv::Point &leftCorner,
		const cv::Point &rightCorner,
		cv::Point &cornerCenter
);

void calculatePupilCenter(const cv::Point &leftPupil,
		const cv::Point &rightPupil,
		cv::Point &pupilCenter
);
cv::Point calculateCenter( const cv::Point &pointA,
		const cv::Point &pointB
);

void calculateCenterPoint( const cv::Point &leftPupil,
		const cv::Point &leftCorner,
		const cv::Point &rightPupil,
		const cv::Point &rightCorner,
		cv::Point cornerCenter,
		cv::Point pupilCenter
);

#endif
