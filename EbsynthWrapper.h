#pragma once

#define NOMINMAX
#include <vector>
#include <string>

// cv::Mat forward declaration
namespace cv
{
	class Mat;
};

cv::Mat CallStyLit(cv::Mat sourceStyleMat, 
	               std::vector<cv::Mat> sources, 
	               std::vector<cv::Mat> targets, 
				   const float styLitMaxMP,
	               const float styleWeight, 
	               const std::string& exposeGuidePath, 
	               const std::string& stylitBackend, 
				   std::string& errorMessage);