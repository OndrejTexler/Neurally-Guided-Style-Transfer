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

void CreateColorRegionsGuide(cv::Mat& style, cv::Mat& target);

void ColorTransfer(cv::Mat& result, cv::Mat& reference);

std::vector<std::pair<cv::Mat, cv::Mat>> CreateContentResponseGuide(const cv::Mat& style, const cv::Mat& target);

cv::Mat RunUST(cv::Mat style, cv::Mat target, const std::string& tmpPath, const int subsampleCoeffForUST, const float UST_alpha, const bool useNewUST);

cv::Mat FakeRunningUST(const std::string& USTResultPath, const int USTShorterSide);

int SubsampleIfNecessary(cv::Mat& sourceStyleMat, std::vector<cv::Mat>& sources, std::vector<cv::Mat>& targets, const float styLitMaxMP);

cv::Mat UpsampleIfNecessaty(const std::vector<int>& finalNNF, const int NNF_height, const int NNF_width, const int subsampleCoefficient, const int patchsize, const cv::Mat& originalStyle, const int originalTargetRows, const int originalTargetCols);