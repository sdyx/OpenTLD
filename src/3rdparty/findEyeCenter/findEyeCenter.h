#ifndef EYE_CENTER_H
#define EYE_CENTER_H

#include <opencv2/imgproc/imgproc.hpp>
//#include "imgproc.hpp"

//https://github.com/trishume/eyeLike/blob/master/src/findEyeCenter.h

cv::Point findEyeCenter(cv::Mat face, cv::Rect eye, std::string debugWindow);

#endif
