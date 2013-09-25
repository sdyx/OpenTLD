#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <iostream>
#include <queue>
#include <stdio.h>

#include "constants.h"
#include "helpers.h"

#include "findEyeCorner.h"

extern bool debug;

////
//
// This code is from:
// 	https://github.com/trishume/eyeLike/blob/master/src/findEyeCorner.cpp
//
////

cv::Mat *leftCornerKernel;
cv::Mat *rightCornerKernel;

// not constant because stupid opencv type signatures

/*
float kEyeCornerKernel[4][6] = 
{
	{-1,-1,-1, 1, 1, 1},
	{-1,-1,-1,-1, 1, 1},
	{-1,-1,-1,-1, 0, 3},
	{ 1, 1, 1, 1, 1, 1},
};
*/
float kEyeCornerKernel[4][6] = 
{
	{-1,-1,-1, 1, 1, 1},
	{-1,-1,-1,-1, 1, 1},
	{-1,-1,-1,-1, 0, 3},
	{ 1, 1, 1, 1, 1, 2},
};
/*
float kEyeCornerKernel[4][6] = 
{
	{-1/24,-1/24,-1/24, 1/24, 1/24, 1/24 },
	{-1/24,-1/24,-1/24,-1/24, 1/24, 1/24 },
	{-1/24,-1/24,-1/24,-1/24,    0, 3/24 },
	{ 1/24, 1/24, 1/24, 1/24, 1/24, 1/24},
};
*/
void createCornerKernels()
{
  rightCornerKernel = new cv::Mat( 4, 6, CV_32F, kEyeCornerKernel );
  leftCornerKernel = new cv::Mat( 4, 6, CV_32F );
  // flip horizontally
  cv::flip( *rightCornerKernel, *leftCornerKernel, 1 );
}

void releaseCornerKernels()
{
	delete leftCornerKernel;
	delete rightCornerKernel;
}

// TODO implement these
cv::Mat eyeCornerMap(const cv::Mat &region,
	bool left,
	bool left2 )
{
	cv::Mat cornerMap;
	cv::Size sizeRegion = region.size();
	cv::Range colRange( sizeRegion.width / 4, sizeRegion.width * 3 / 4 );
	cv::Range rowRange( sizeRegion.height / 4, sizeRegion.height * 3 / 4 );
	cv::Mat miRegion( region, rowRange, colRange );

	createCornerKernels();
	// the next line throws a segmentation fault
	// this was because, createCornerKernels was not being called and therefor
	// has been empty
	cv::filter2D( miRegion, 
			cornerMap, 
			CV_32F,
			( left && !left2 ) || ( !left && !left2 ) 
					? *leftCornerKernel : *rightCornerKernel );
	//cv::filter2D( miRegion, cornerMap, -1, *leftCornerKernel );
	return cornerMap;
}

cv::Point2f findEyeCorner(cv::Mat region,
	bool left,
	bool left2 )
{
	if ( debug )
	{
		std::cout << "detecting eye corner" << std::endl;
	}
	cv::Point2f maxP2 = cv::Point2f( 0.0, 0.0 );
	try
	{
		cv::Mat bw;
		//cv::cvtColor( region, bw, CV_GRAY2BGR );
		//cv::cvtColor( bw, bw, CV_BGR2GRAY );
		
		cv::Mat cornerMap = eyeCornerMap( region, left, left2 );

		cv::Point maxP;
		cv::minMaxLoc( cornerMap, NULL, NULL, NULL, &maxP );
		maxP2 = findSubpixelEyeCorner( cornerMap, maxP );
		//cv::rectangle( region, maxP, 1, CV_RGB( 255, 155, 0), 2 );

		// GFTT
	//  std::vector<cv::Point2f> corners;
	//  cv::goodFeaturesToTrack(region, corners, 500, 0.005, 20);
	//  for (int i = 0; i < corners.size(); ++i) {
	//    cv::circle(region, corners[i], 2, 200);
	//  }
		if ( debug )
		{
			if ( region.cols >= 0 &&
					region.rows >= 0
			)
			{
				cv::circle( region, maxP2, 0.5, CV_RGB( 0, 0, 255), 2 );
				cv::imshow( "Corners", region );
			}
			else
			{
				std::cout << "can not display region because its size is"
				<< " the following rows: "
				<< region.rows << " cols: "
				<< region.cols
				<< std::endl;
			}
		}

	}
	catch( cv::Exception& e )
	{
		const char* err_msg = e.what();
		std::cout << "!!!Error in findEyeCorner\n" << err_msg << std::endl;
	}
	if ( debug )
	{	
		std::cout << "done detecting eye corner" << std::endl;
	}
	return maxP2;
}

cv::Point2f findSubpixelEyeCorner(
	cv::Mat region,
	cv::Point maxP )
{
	cv::Size sizeRegion = region.size();

  // Matrix dichotomy
  // Not useful, matrix becomes too small

  /*int offsetX = 0;
  if(maxP.x - sizeRegion.width / 4 <= 0) {
    offsetX = 0;
  } else if(maxP.x + sizeRegion.width / 4 >= sizeRegion.width) {
    offsetX = sizeRegion.width / 2 - 1;
  } else {
    offsetX = maxP.x - sizeRegion.width / 4;
  }
  int offsetY = 0;
  if(maxP.y - sizeRegion.height / 4 <= 0) {
    offsetY = 0;
  } else if(maxP.y + sizeRegion.height / 4 >= sizeRegion.height) {
    offsetY = sizeRegion.height / 2 - 1;
  } else {
    offsetY = maxP.y - sizeRegion.height / 4;
  }
  cv::Range colRange(offsetX, offsetX + sizeRegion.width / 2);
  cv::Range rowRange(offsetY, offsetY + sizeRegion.height / 2);

  cv::Mat miRegion(region, rowRange, colRange);


if(left){
    imshow("aa",miRegion);
  } else {
    imshow("aaa",miRegion);
  }*/

	cv::Mat cornerMap( sizeRegion.height * 10, sizeRegion.width * 10, CV_32F );
	cv::resize(region, cornerMap, cornerMap.size(), 0, 0, cv::INTER_CUBIC);
	cv::Point maxP2;
	cv::minMaxLoc( cornerMap, NULL,NULL,NULL,&maxP2 );
	return cv::Point2f( sizeRegion.width / 2 + maxP2.x / 10,
		sizeRegion.height / 2 + maxP2.y / 10 );
}
