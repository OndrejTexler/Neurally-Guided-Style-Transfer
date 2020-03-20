#include "OpenCVUtils.h"

#include "opencv2/opencv.hpp"
#include "opencv2/imgproc/imgproc.hpp"

using namespace cv;

Mat cumsum(Mat & src)
{
	Mat result = Mat::zeros(Size(src.cols, src.rows), CV_32FC1);
	for (int i = 0; i < src.rows; ++i)
	{
		for (int j = 0; j < src.cols; ++j)
		{
			if (i == 0)
			{
				result.at<float>(i, j) = src.at<float>(i, j);
			}
			else
			{
				result.at<float>(i, j) = src.at<float>(i, j) + result.at<float>(i - 1, j);
			}

		}

	}

	return result;
}

Mat& ScanImageAndReduceC(Mat& I, const unsigned char* const table)
{
	// accept only char type matrices
	CV_Assert(I.depth() != sizeof(uchar));

	int channels = I.channels();

	int nRows = I.rows;
	int nCols = I.cols * channels;

	if (I.isContinuous())
	{
		nCols *= nRows;
		nRows = 1;
	}

	int i, j;
	uchar* p;
	for (i = 0; i < nRows; ++i)
	{
		p = I.ptr<uchar>(i);
		for (j = 0; j < nCols; ++j)
		{
			p[j] = table[p[j]];
		}
	}
	return I;
}

Mat GrayHistMatching(Mat I, Mat R)
{
	/* Histogram Matching of a gray image with a reference*/
	// accept two images I (input image)  and R (reference image)

	Mat Result;     // The Result image

	int L = 256;    // Establish the number of bins
	if (I.channels() != 1)
	{
		//cout << "Please use Gray image" << endl;
		return Mat::zeros(I.size(), CV_8UC1);
	}
	Mat G, S, F; //G is the reference CDF, S the CDF of the equlized given image, F is the map from S->G
	if (R.cols>1)
	{
		if (R.channels() != 1)
		{
			//cout << "Please use Gray reference" << endl;
			return Mat::zeros(I.size(), CV_8UC1);
		}


		Mat R_hist, Rp_hist; //R_hist the counts of pixels for each level, Rp_hist is the PDF of each level
							 /// Set the ranges ( for B,G,R) )
		float range[] = { 0, 256 };
		const float* histRange = { range };
		bool uniform = true; bool accumulate = false;
		//calcHist(const Mat* images, int nimages, const int* channels, InputArray mask, OutputArray hist, int dims, const int* histSize, const float**         ranges, bool uniform=true, bool accumulate=false )
		calcHist(&R, 1, 0, Mat(), R_hist, 1, &L, &histRange, uniform, accumulate);
		//Calc PDF of the image
		Rp_hist = R_hist / (I.rows*I.cols);
		//calc G=(L-1)*CDF(p)
		Mat CDF = cumsum(Rp_hist);
		G = (L - 1)*CDF;
		for (int i = 0; i<G.rows; i++)
			G.at<Point2f>(i, 0).x = (float)cvRound(G.at<Point2f>(i, 0).x);//round G
	}
	else
	{
		//else, the given R is the reference PDF
		Mat CDF = cumsum(R);
		G = (L - 1)*CDF;
		for (int i = 0; i<G.rows; i++)
			G.at<Point2f>(i, 0).x = (float)cvRound(G.at<Point2f>(i, 0).x);//round G
	}
	/// Establish the number of bins
	Mat S_hist, Sp_hist; //S_hist the counts of pixels for each level, Sp_hist is the PDF of each level
						 /// Set the ranges ( for B,G,R) )
	float range[] = { 0, 256 };
	const float* histRange = { range };

	bool uniform = true; bool accumulate = false;

	//calcHist(const Mat* images, int nimages, const int* channels, InputArray mask, OutputArray hist, int dims, const int* histSize, const float**         ranges, bool uniform=true, bool accumulate=false )
	calcHist(&I, 1, 0, Mat(), S_hist, 1, &L, &histRange, uniform, accumulate);

	//Calc PDF of the image
	Sp_hist = S_hist / (I.rows*I.cols);
	//calc s=(L-1)*CDF(p)
	Mat CDF = cumsum(Sp_hist);
	S = (L - 1)*CDF;
	for (int i = 0; i<S.rows; i++)
		S.at<Point2f>(i, 0).x = (float)cvRound(S.at<Point2f>(i, 0).x);//round S

	F = Mat::zeros(S.size(), CV_32F);
	int minIndex = -1;
	double T, min = 100000;
	for (int i = 0; i<S.rows; i++)
	{
		for (int j = 0; j<G.rows; j++)
		{
			T = abs(double(S.at<Point2f>(i, 0).x) - double(G.at<Point2f>(j, 0).x));
			if (T == 0)
			{
				minIndex = j;
				break;
			}
			else
				if (T<min)
				{
					minIndex = j;
					min = T;
				}
		}
		F.at<Point2f>(i, 0).x = (float)minIndex;
		minIndex = -1;
		min = 1000000;
	}
	uchar table[256];
	for (int i = 0; i<256; i++)
	{
		table[i] = (int)F.at<Point2f>(i, 0).x;
	}

	Result = ScanImageAndReduceC(I, table);

	return Result;
}

void ResizeImageMaintainAspectRatio(Mat& image, const int desiredSmallerSide)
{
	int smallerSide = std::min(image.cols, image.rows);

	if (desiredSmallerSide >= smallerSide) 
	{
		return;
	}

	cv::resize(image, image, cv::Size( ((float)image.cols/(float)smallerSide) * desiredSmallerSide,
									   ((float)image.rows/(float)smallerSide) * desiredSmallerSide) );
}

cv::Mat Imread(const std::string& path, const bool swapRandB)
{
	cv::Mat image = cv::imread(path);

	if (!image.empty() && swapRandB)
	{
		cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
	}
	return image;
}

void Imwrite(const std::string& path, const Mat& image, const bool swapRandB)
{
	if(swapRandB)
	{
		Mat swappedRandB;
		cvtColor(image, swappedRandB, cv::COLOR_BGR2RGB);
		cv::imwrite(path, swappedRandB);
	}
	else 
	{
		cv::imwrite(path, image);
	}
}

bool isOutOfImage(const Mat& image, const int row, const int col)
{
	return row < 0 ||
		col < 0 ||
		row >= image.rows ||
		col >= image.cols;
}

/*
void imshowInWindow(const std::string& windowTitle, const cv::Mat& image)
{
	cv::namedWindow(windowTitle, WINDOW_NORMAL);
	cv::resizeWindow(windowTitle, image.cols, image.rows);
	imshow(windowTitle, image);
	cv::waitKey(1);
}

void imshowInWindow(const std::string& windowTitle, const cv::Mat& image, int windowWidth, int windowHeight)
{
	cv::namedWindow(windowTitle, WINDOW_NORMAL);
	cv::resizeWindow(windowTitle, windowWidth, windowHeight);
	imshow(windowTitle, image);
	cv::waitKey(1);
}

void imshowInWindow(const std::string& windowTitle, const cv::Mat& image, int windowWidth, int windowHeight, int moveX, int moveY)
{
	cv::namedWindow(windowTitle, WINDOW_NORMAL);
	cv::resizeWindow(windowTitle, windowWidth, windowHeight);
	cv::moveWindow(windowTitle, moveX, moveY);
	imshow(windowTitle, image);
	cv::waitKey(1);
}
*/