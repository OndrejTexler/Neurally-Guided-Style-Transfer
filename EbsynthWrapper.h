#pragma once

#define NOMINMAX
#include <vector>
#include <string>

// cv::Mat forward declaration
namespace cv
{
	class Mat;
};

cv::Mat CallEbsynth(cv::Mat sourceStyleMat, 
	               std::vector<cv::Mat> sources, 
	               std::vector<cv::Mat> targets, 
				   const float patchBasedMaxMP,
	               const float styleWeight,
	               const std::string& patchBasedBackend,
				   std::string& errorMessage);