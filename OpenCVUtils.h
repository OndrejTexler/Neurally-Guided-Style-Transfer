#pragma once

#include <cstdio>
#include <string>

// cv::Mat forward declaration
namespace cv
{
	class Mat;
};

cv::Mat cumsum(cv::Mat & src);

cv::Mat& ScanImageAndReduceC(cv::Mat& I, const unsigned char* const table);

cv::Mat GrayHistMatching(cv::Mat I, cv::Mat R);

void ResizeImageMaintainAspectRatio(cv::Mat& image, const int desiredSmallerSide);

cv::Mat Imread(const std::string& path, const bool swapRandB);

void Imwrite(const std::string& path, const cv::Mat& image, const bool swapRandB);

bool isOutOfImage(const cv::Mat& image, const int row, const int col);

/*
void imshowInWindow(const std::string& windowTitle, const cv::Mat& image);

void imshowInWindow(const std::string& windowTitle, const cv::Mat& image, int windowWidth, int windowHeight);

void imshowInWindow(const std::string& windowTitle, const cv::Mat& image, int windowWidth, int windowHeight, int moveX, int moveY);
*/