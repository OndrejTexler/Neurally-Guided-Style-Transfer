#pragma once
#include <vector>
#include <string>

// cv::Mat forward declaration
namespace cv
{
	class Mat;
};

std::pair<cv::Mat, cv::Mat> CreateGrayScaleGuide(cv::Mat style, cv::Mat target, int levelOfAbstraction);

void Recolor(cv::Mat& lumImage, cv::Mat& colorImage);

int SubsampleIfNecessary(cv::Mat& sourceStyleMat, std::vector<cv::Mat>& sources, std::vector<cv::Mat>& targets, const float patchBasedMaxMP);

cv::Mat UpsampleIfNecessaty(const std::vector<int>& finalNNF, const int NNF_height, const int NNF_width, const int subsampleCoefficient, const int patchsize, const cv::Mat& originalStyle, const int originalTargetRows, const int originalTargetCols);