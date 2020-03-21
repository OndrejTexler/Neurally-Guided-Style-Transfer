#include "SynthesisUtils.h"

#include "opencv2/opencv.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "OpenCVUtils.h" // isOutOfImage

#include <omp.h>

using namespace cv;
using namespace std;

//Private functions
Mat2i upsampleNNF(const Mat2i& NNFSmall, const int coeficient);
Mat voting(const Mat& style, const Mat2i& NNF, const int NNF_patchsize);


// TODO: Actually, EBSYNTH needs 3-channel no more, remove the last COLOR_GRAY2BGR conversion 
pair<Mat, Mat> CreateGrayScaleGuide(Mat source, Mat target, int levelOfAbstraction)
{
	//Convert to gray-scale
	cv::cvtColor(source, source, cv::COLOR_BGR2GRAY);
	cv::cvtColor(target, target, cv::COLOR_BGR2GRAY);

	//TODO: do we want to blur it?
	// Blur the source 
	if (levelOfAbstraction > 1) {
		int kernelSize = (levelOfAbstraction * 2) - 1; // Make it odd
		cv::GaussianBlur(source, source, cv::Size(kernelSize, kernelSize), 0);
	}

	//TODO: Fix GrayHistMatching to support diferent size images
	// Match target histogram to be the same as source histogram
	if(source.size != target.size)
	{
		cout << "WARNING: GrayHistMatching cannot be used when style and target differ in size. Result could be thus slightly worse. (This limitation will be fixed soon)" << endl;
	}
	else
	{
		target = GrayHistMatching(target, source);
	}

	//Convert back to 3-channel, it is still grays-cale, but 3-channel
	cv::cvtColor(source, source, cv::COLOR_GRAY2BGR);
	cv::cvtColor(target, target, cv::COLOR_GRAY2BGR);

	return make_pair(source, target);
}


void Recolor(Mat & lumImage, Mat & colorImage) 
{
	cv::cvtColor(lumImage, lumImage, cv::COLOR_BGR2Lab);
	cv::cvtColor(colorImage, colorImage, cv::COLOR_BGR2Lab);

	{
		for (int row = 0; row < lumImage.rows; row++)
		{
			for (int col = 0; col < lumImage.cols; col++)
			{
				lumImage.at<cv::Vec3b>(row, col)[1] = colorImage.at<cv::Vec3b>(row, col)[1];
				lumImage.at<cv::Vec3b>(row, col)[2] = colorImage.at<cv::Vec3b>(row, col)[2];
			}
		}
	}
	cv::cvtColor(lumImage, lumImage, cv::COLOR_Lab2BGR);
}


// TODO: Check patchBasedMaxMP against the available GPU memory
int SubsampleIfNecessary(cv::Mat& sourceStyleMat, vector<cv::Mat>& sources, vector<cv::Mat>& targets, const float patchBasedMaxMP)
{	
	const float pixelsMP = std::max(((float)sources[0].rows / 1000.0f)*((float)sources[0].cols / 1000.0f), 
		                            ((float)targets[0].rows / 1000.0f)*((float)targets[0].cols / 1000.0f));

	if (patchBasedMaxMP == 0.0f) 
	{
		return 1;
	}

	int coefficient = (int)ceilf(std::sqrt(ceil(pixelsMP / patchBasedMaxMP)));
	
	if (coefficient <= 1) 
	{
		return coefficient;
	}

	cv::resize(sourceStyleMat, sourceStyleMat, Size(sourceStyleMat.cols / coefficient, sourceStyleMat.rows / coefficient));

	for (Mat& source : sources) 
	{
		cv::resize(source, source, Size(source.cols / coefficient, source.rows / coefficient));
	}

	for (Mat& target : targets)
	{
		cv::resize(target, target, Size(target.cols / coefficient, target.rows / coefficient));
	}

	return coefficient;
}

Mat UpsampleIfNecessaty(const std::vector<int>& finalNNF, const int NNF_height, const int NNF_width, const int subsampleCoefficient, const int patchSize, const Mat& originalStyle, const int originalTargetRows, const int originalTargetCols)
{
	Mat2i NNF_Small = Mat2i(NNF_height, NNF_width);
	{
		int finalNNFIndex = 0;
		for (int row = 0; row < NNF_height; row++)
		{
			for (int col = 0; col < NNF_width; col++)
			{
				NNF_Small.at<Vec2i>(row, col) = Vec2i(finalNNF[finalNNFIndex + 1], finalNNF[finalNNFIndex]);
				finalNNFIndex += 2;
			}
		}
	}
	
	Mat2i NNF = upsampleNNF(NNF_Small, subsampleCoefficient);

	Mat output = voting(originalStyle, NNF, patchSize);

	// Resize if the output is slightly smaller due to rounding
	if(output.rows != originalTargetRows || output.cols != originalTargetCols)
	{
		cv::resize(output, output, Size(originalTargetCols, originalTargetRows), cv::InterpolationFlags::INTER_CUBIC);
	}

	return output;
}

Mat2i upsampleNNF(const Mat2i& NNFSmall, const int coeficient)
{
	Mat2i NNFBig = Mat2i(NNFSmall.rows * coeficient, NNFSmall.cols * coeficient);

	#pragma omp parallel for
	for (int row = 0; row < NNFBig.rows; row++)
	{
		for (int col = 0; col < NNFBig.cols; col++)
		{
			Vec2i value = NNFSmall.at<Vec2i>(row / coeficient, col / coeficient);

			NNFBig.at<Vec2i>(row, col) = Vec2i(value[0] * coeficient + (row%coeficient),
				value[1] * coeficient + (col%coeficient));
		}
	}

	return NNFBig;
}

Mat voting(const Mat& style, const Mat2i& NNF, const int NNF_patchsize)
{
	Mat output = Mat::zeros(NNF.rows, NNF.cols, CV_8UC3);
	const int halfPatchsize = NNF_patchsize / 2;

	//Iterate throughout the NNF/output image
	#pragma omp parallel for
	for (int row = 0; row < output.rows; row++)
	{
		for (int col = 0; col < output.cols; col++)
		{
			short accumulatedB = 0; 
			short accumulatedG = 0;
			short accumulatedR = 0;
			short acumulatedPixelCount = 0;

			//Iterate throughout the patch
			for (int row_offset_in_patch = -1*halfPatchsize; row_offset_in_patch <= halfPatchsize; row_offset_in_patch++)
			{
				for (int col_offset_in_patch = -1*halfPatchsize; col_offset_in_patch <= halfPatchsize; col_offset_in_patch++)
				{
					const int row_in_NNF = row + row_offset_in_patch;
					const int col_in_NNF = col + col_offset_in_patch;

					if (isOutOfImage(output, row_in_NNF, col_in_NNF))
					{
						continue;
					}

					const Vec2i nearest = NNF.at<Vec2i>(row_in_NNF, col_in_NNF);
					const int row_in_style = nearest[0] - row_offset_in_patch;
					const int col_in_style = nearest[1] - col_offset_in_patch;

					const Vec3i stylePixel = style.at<Vec3b>(row_in_style, col_in_style);
					accumulatedB += stylePixel[0];
					accumulatedG += stylePixel[1];
					accumulatedR += stylePixel[2];
					acumulatedPixelCount++;
				}
			}

			output.at<Vec3b>(row, col) = Vec3b((float)accumulatedB / (float)acumulatedPixelCount, (float)accumulatedG / (float)acumulatedPixelCount, (float)accumulatedR / (float)acumulatedPixelCount);
		}
	}

	return output;
}