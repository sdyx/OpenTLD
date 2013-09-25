#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <iostream>
#include <queue>
#include <stdio.h>

#include "constants.h"
#include "helpers.h"

extern bool debug;

////
//
// This code is from:
//	https://github.com/trishume/eyeLike/blob/master/src/findEyeCenter.cpp
//
////

// Pre-declarations
cv::Mat floodKillEdges(cv::Mat &mat);

#pragma mark Visualization
/*
template<typename T> mglData *matToData(const cv::Mat &mat) {
  mglData *data = new mglData(mat.cols,mat.rows);
  for (int y = 0; y < mat.rows; ++y) {
    const T *Mr = mat.ptr<T>(y);
    for (int x = 0; x < mat.cols; ++x) {
      data->Put(((mreal)Mr[x]),x,y);
    }
  }
  return data;
}

void plotVecField(const cv::Mat &gradientX, const cv::Mat &gradientY, const cv::Mat &img) {
  mglData *xData = matToData<double>(gradientX);
  mglData *yData = matToData<double>(gradientY);
  mglData *imgData = matToData<float>(img);
  
  mglGraph gr(0,gradientX.cols * 20, gradientY.rows * 20);
  gr.Vect(*xData, *yData);
  gr.Mesh(*imgData);
  gr.WriteFrame("vecField.png");
  
  delete xData;
  delete yData;
  delete imgData;
}*/

#pragma mark Helpers

cv::Point unscalePoint(cv::Point p, cv::Rect origSize) {
  float ratio = (((float)kFastEyeWidth)/origSize.width);
  int x = round(p.x / ratio);
  int y = round(p.y / ratio);
  return cv::Point(x,y);
}

void scaleToFastSize(const cv::Mat &src,cv::Mat &dst) {
  cv::resize(src, dst, cv::Size(kFastEyeWidth,(((float)kFastEyeWidth)/src.cols) * src.rows));
}

cv::Mat computeMatXGradient(const cv::Mat &mat) {
  cv::Mat out(mat.rows,mat.cols,CV_64F);
  
  for (int y = 0; y < mat.rows; ++y) {
    const uchar *Mr = mat.ptr<uchar>(y);
    double *Or = out.ptr<double>(y);
    
    Or[0] = Mr[1] - Mr[0];
    for (int x = 1; x < mat.cols - 1; ++x) {
      Or[x] = (Mr[x+1] - Mr[x-1])/2.0;
    }
    Or[mat.cols-1] = Mr[mat.cols-1] - Mr[mat.cols-2];
  }
  
  return out;
}

#pragma mark Main Algorithm

void testPossibleCentersFormula(
	int x,
	int y,
	unsigned char weight,
	double gx,
	double gy,
	cv::Mat &out )
{
	// for all possible centers
	for (int cy = 0; cy < out.rows; ++cy )
	{
		double *Or = out.ptr<double>( cy );
		for (int cx = 0; cx < out.cols; ++cx )
		{
			if (x == cx && y == cy)
			{
				continue;
			}
			// create a vector from the possible center to the gradient origin
			double dx = x - cx;
			double dy = y - cy;
			// normalize d
			double magnitude = sqrt( ( dx * dx ) + ( dy * dy ) );
			dx = dx / magnitude;
			dy = dy / magnitude;
			double dotProduct = dx * gx + dy * gy;
			dotProduct = std::max( 0.0, dotProduct );
			// square and multiply by the weight
			if ( kEnableWeight )
			{
				Or[cx] += dotProduct * dotProduct * ( weight / kWeightDivisor );
			}
			else
			{
				Or[cx] += dotProduct * dotProduct;
			}
		}
	}
}

cv::Point findEyeCenter( 
	cv::Mat face,
	cv::Rect eye,
	std::string debugWindow
)
{
	cv::Mat out;
	cv::Mat eyeROI;
	cv::Mat eyeROIUnscaled;
	cv::Mat mags;
	cv::Mat gradientY;
	cv::Mat gradientX;
	double gradientThresh;
	cv::Mat weight;
	cv::Point maxP;
	double maxVal;
	cv::Mat floodClone;
	
	if ( eye.width <= 0 &&
			eye.height <= 0 &&
			eye.x <= 0 &&
			eye.y <= 0 
	)
	{
		if ( debug )
		{
			std::cout << "Did not get a valid eye search area. Returning "
					<< "cv::Point( 0, 0 )" << std::endl;
		}
		return cv::Point( 0, 0 );
	}
	
	try
	{
		if ( debug )
		{
			std::cout << " findEyeCenter - 1 " << std::endl;
		}
		// TODO: FIX error between 1 and 2
		// OpenCV Error: Assertion failed (ssize.area() > 0) in resize, file /opt/opencv-2.4.5/modules/imgproc/src/imgwarp.cpp, line 1725
		//!!!Error in findEyeCenter
		///opt/opencv-2.4.5/modules/imgproc/src/imgwarp.cpp:1725: error: (-215) ssize.area() > 0 in function resize
		// OpenCV Error: Bad flag (parameter or structure field) (Unrecognized or unsupported array type) in cvGetMat, file /opt/opencv-2.4.5/modules/core/src/array.cpp, line 2482
		eyeROIUnscaled = face( eye );
		scaleToFastSize( eyeROIUnscaled, eyeROI );
		if ( debug )
		{
			// draw eye region
			cv::rectangle( face, eye, 1234 );
		}
		//-- Find the gradient
		gradientX = computeMatXGradient( eyeROI );
		gradientY = computeMatXGradient( eyeROI.t() ).t();
		//-- Normalize and threshold the gradient
		// compute all the magnitudes
		mags = matrixMagnitude( gradientX, gradientY );
		//compute the threshold
		gradientThresh = computeDynamicThreshold( mags, kGradientThreshold );
		//double gradientThresh = kGradientThreshold;
		//double gradientThresh = 0;
		//normalize
		if ( debug )
		{
			std::cout << " findEyeCenter - 2 " << std::endl;
		}
		for ( int y = 0; y < eyeROI.rows; ++y )
		{
			double *Xr = gradientX.ptr<double>( y ), *Yr = gradientY.ptr<double>( y );
			const double *Mr = mags.ptr<double>(y);
			for ( int x = 0; x < eyeROI.cols; ++x )
			{
				double gX = Xr[ x ], gY = Yr[ x ];
				double magnitude = Mr[ x ];
				if ( magnitude > gradientThresh )
				{
					Xr[ x ] = gX / magnitude;
					Yr[ x ] = gY / magnitude;
				}
				else
				{
					Xr[ x ] = 0.0;
					Yr[ x ] = 0.0;
				}
			}
		}
		if ( debug )
		{
			std::cout << " findEyeCenter - 3 " << std::endl;
			cv::imshow( debugWindow,gradientX );
		}
		//-- Create a blurred and inverted image for weighting
		cv::GaussianBlur( eyeROI, weight, cv::Size( kWeightBlurSize, kWeightBlurSize ), 0, 0 );
		if ( debug )
		{
			std::cout << " findEyeCenter - 4 " << std::endl;
		}
		for (int y = 0; y < weight.rows; ++y) 
		{
			unsigned char *row = weight.ptr<unsigned char>(y);
			for (int x = 0; x < weight.cols; ++x)
			{
		  		row[x] = ( 255 - row[ x ] );
			}
		}
		if ( debug )
		{
			std::cout << " findEyeCenter - 5 " << std::endl;
		}
		//-- Run the algorithm!
		cv::Mat outSum = cv::Mat::zeros( eyeROI.rows, eyeROI.cols, CV_64F );
		if ( debug )
		{
			std::cout << " findEyeCenter - 6 " << std::endl;
		}
		// for each possible center
		//printf( "Eye Size: %ix%i\n", outSum.cols, outSum.rows );
		for ( int y = 0; y < weight.rows ; ++y ) 
		{
			const unsigned char *Wr = weight.ptr<unsigned char>( y );
			const double *Xr =
					gradientX.ptr<double>(y), *Yr = gradientY.ptr<double>( y );
			for ( int x = 0; x < weight.cols; ++x )
			{
				double gX = Xr[ x ], gY = Yr[ x ];
				if (gX == 0.0 && gY == 0.0)
				{
					continue;
				}
				testPossibleCentersFormula( x, y, Wr[ x ], gX, gY, outSum );
			}
		}
		if ( debug )
		{
			std::cout << " findEyeCenter - 7 " << std::endl;
		}
		// scale all the values down, basically averaging them
		double numGradients = ( weight.rows * weight.cols );
		outSum.convertTo( out, CV_32F, 1.0 / numGradients );
		if ( debug )
		{
			std::cout << " findEyeCenter - 8 " << std::endl;
			cv::imshow( debugWindow, out );
		}
		
		//-- Find the maximum point
		cv::minMaxLoc( out, NULL, &maxVal, NULL, &maxP );
		//-- Flood fill the edges
		if ( debug )
		{
			std::cout << " findEyeCenter - 9 " << std::endl;
		}
		if ( kEnablePostProcess )
		{
			//double floodThresh = computeDynamicThreshold(out, 1.5);
			double floodThresh = maxVal * kPostProcessThreshold;
			cv::threshold( out,
					floodClone,
					floodThresh,
					0.0f,
					cv::THRESH_TOZERO
			);
			if( kPlotVectorField )
			{
				//plotVecField(gradientX, gradientY, floodClone);
				imwrite( "eyeFrame.png", eyeROIUnscaled );
			}
			cv::Mat mask = floodKillEdges(floodClone);
			//imshow( debugWindow + " Mask", mask );
			//imshow( debugWindow, out );
			// redo max
			cv::minMaxLoc( out, NULL, &maxVal, NULL, &maxP, mask );
		}
		
		if ( debug )
		{
			std::cout << " findEyeCenter - return value" << std::endl;
		}
		return unscalePoint( maxP, eye );
	}
	catch( cv::Exception& e )
	{
		const char* err_msg = e.what();
		std::cout << "!!!Error in findEyeCenter\n" << err_msg << std::endl;
	}

}

#pragma mark Postprocessing

bool floodShouldPushPoint(const cv::Point &np, const cv::Mat &mat) {
  return inMat(np, mat.rows, mat.cols);
}

// returns a mask
cv::Mat floodKillEdges(cv::Mat &mat) {
  rectangle(mat,cv::Rect(0,0,mat.cols,mat.rows),255);
  
  cv::Mat mask(mat.rows, mat.cols, CV_8U, 255);
  std::queue<cv::Point> toDo;
  toDo.push(cv::Point(0,0));
  while (!toDo.empty()) {
    cv::Point p = toDo.front();
    toDo.pop();
    if (mat.at<float>(p) == 0.0f) {
      continue;
    }
    // add in every direction
    cv::Point np(p.x + 1, p.y); // right
    if (floodShouldPushPoint(np, mat)) toDo.push(np);
    np.x = p.x - 1; np.y = p.y; // left
    if (floodShouldPushPoint(np, mat)) toDo.push(np);
    np.x = p.x; np.y = p.y + 1; // down
    if (floodShouldPushPoint(np, mat)) toDo.push(np);
    np.x = p.x; np.y = p.y - 1; // up
    if (floodShouldPushPoint(np, mat)) toDo.push(np);
    // kill it
    mat.at<float>(p) = 0.0f;
    mask.at<uchar>(p) = 0;
  }
  return mask;
}
