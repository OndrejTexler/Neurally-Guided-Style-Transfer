#include "EbsynthWrapper.h"
#include "ebsynth.h"			// ebsynthRun
#include <utility>				// pair
#include <algorithm>			// std::min
#include "CreateGuideUtils.h"	// UpsampleIfNecessaty, SubsampleIfNecessary
#include "opencv2/opencv.hpp"   // cv::Mat
#include "TimeMeasure.h"		// TimeMeasure

using namespace std;

// Private function
int evalNumChannels(const unsigned char* data, const int numPixels);

// Private function
std::pair<int, int> pyramidLevelSize(const std::pair<int, int>& sizeBase, const int level);


/*
 *  All input images are passed as deep-copy, because they are modified (subsampled) if they are too big
*/
cv::Mat CallEbsynth(cv::Mat sourceStyleMat, vector<cv::Mat> sources, vector<cv::Mat> targets, const float patchBasedMaxMP, const float styleWeight, 
				   const std::string& patchBasedBackend, std::string& errorMessage)
{
	errorMessage.clear();
	cv::Mat originalStyle;
	sourceStyleMat.copyTo(originalStyle);
	const int originalTarget_rows = targets[0].rows;
	const int originalTarget_cols = targets[0].cols;

	int subsampleCoefficient = SubsampleIfNecessary(sourceStyleMat, sources, targets, patchBasedMaxMP);
	

	cout << "Patch based synthesis runs on resolution " << sources[0].cols << "x" << sources[0].rows << endl;
	
	struct Guide
	{
		float          weight;

		int            sourceWidth;
		int            sourceHeight;
		unsigned char* sourceData;

		int            targetWidth;
		int            targetHeight;
		unsigned char* targetData;

		int            numChannels;
	};

	const int sourceWidth = sources[0].cols;
	const int sourceHeight = sources[0].rows;
	const int targetWidth = targets[0].cols;
	const int targetHeight = targets[0].rows;

	//float styleWeight = NAN;

	std::vector<Guide> guides;

	float uniformityWeight = 3500; // No restrictions
	int patchSize = 5; // Must be >=3 AND must be odd number 
	int numPyramidLevels = -1; // Must be >= 1 OR -1 to compute levels automaticaly
	int numSearchVoteIters = 6; // Must be >= 0
	int numPatchMatchIters = 4; // Must be >= 0
	int stopThreshold = 5; // Must be >= 0

	const size_t numGuides = sources.size();

	//unsigned char* sourceStyleData = tryLoad(styleFileName, &sourceWidth, &sourceHeight);
	const int numStyleChannelsTotal = 3; //TODO: Is style really always 3-channel? //evalNumChannels(sourceStyleData, sourceWidth*sourceHeight);

	std::vector<unsigned char> sourceStyle(sourceWidth*sourceHeight*numStyleChannelsTotal);
	for (int xy = 0; xy < sourceWidth*sourceHeight; xy++)
	{
		// There is no ALPHA in sourceStyleMat.data
		const int step = 3;

		if (numStyleChannelsTotal > 0) { sourceStyle[xy*numStyleChannelsTotal + 0] = sourceStyleMat.data[xy * step + 0]; }
		if (numStyleChannelsTotal == 2) { sourceStyle[xy*numStyleChannelsTotal + 1] = sourceStyleMat.data[xy * step + 3]; }
		else if (numStyleChannelsTotal > 1) { sourceStyle[xy*numStyleChannelsTotal + 1] = sourceStyleMat.data[xy * step + 1]; }
		if (numStyleChannelsTotal > 2) { sourceStyle[xy*numStyleChannelsTotal + 2] = sourceStyleMat.data[xy * step + 2]; }
		if (numStyleChannelsTotal > 3) { sourceStyle[xy*numStyleChannelsTotal + 3] = sourceStyleMat.data[xy * step + 3]; }
	}

	int numGuideChannelsTotal = 0;

	for (int i = 0; i < numGuides; i++)
	{
		Guide guide;
		guide.sourceData = sources[i].data;
		guide.targetData = targets[i].data;
		guide.numChannels = std::max(evalNumChannels(guide.sourceData, sourceWidth*sourceHeight), evalNumChannels(guide.targetData, targetWidth*targetHeight));
		guide.weight = NAN;
		numGuideChannelsTotal += guide.numChannels;
		guides.push_back(guide);
	}

	if (numStyleChannelsTotal > EBSYNTH_MAX_STYLE_CHANNELS)
	{
		errorMessage = "Stylization failed, too much style channels.";
		return cv::Mat();
	}
	if (numGuideChannelsTotal > EBSYNTH_MAX_GUIDE_CHANNELS)
	{
		errorMessage = "Stylization failed, too much guide channels.";
		return cv::Mat();
	}

	std::vector<unsigned char> sourceGuides(sourceWidth*sourceHeight*numGuideChannelsTotal);
	for (int xy = 0; xy < sourceWidth*sourceHeight; xy++)
	{
		int c = 0;
		for (int i = 0; i < numGuides; i++)
		{
			// There is no ALPHA in guides[i].sourceData
			const int step = 3;

			const int numChannels = guides[i].numChannels;

			if (numChannels > 0) { sourceGuides[xy*numGuideChannelsTotal + c + 0] = guides[i].sourceData[xy * step + 0]; }
			if (numChannels == 2) { sourceGuides[xy*numGuideChannelsTotal + c + 1] = guides[i].sourceData[xy * step + 3]; }
			else if (numChannels > 1) { sourceGuides[xy*numGuideChannelsTotal + c + 1] = guides[i].sourceData[xy * step + 1]; }
			if (numChannels > 2) { sourceGuides[xy*numGuideChannelsTotal + c + 2] = guides[i].sourceData[xy * step + 2]; }
			if (numChannels > 3) { sourceGuides[xy*numGuideChannelsTotal + c + 3] = guides[i].sourceData[xy * step + 3]; }

			c += numChannels;
		}
	}

	std::vector<unsigned char> targetGuides(targetWidth*targetHeight*numGuideChannelsTotal);
	for (int xy = 0; xy < targetWidth*targetHeight; xy++)
	{
		int c = 0;
		for (int i = 0; i < numGuides; i++)
		{
			// There is no ALPHA in guides[i].targetData
			const int step = 3;

			const int numChannels = guides[i].numChannels;

			if (numChannels > 0) { targetGuides[xy*numGuideChannelsTotal + c + 0] = guides[i].targetData[xy * step + 0]; }
			if (numChannels == 2) { targetGuides[xy*numGuideChannelsTotal + c + 1] = guides[i].targetData[xy * step + 3]; }
			else if (numChannels > 1) { targetGuides[xy*numGuideChannelsTotal + c + 1] = guides[i].targetData[xy * step + 1]; }
			if (numChannels > 2) { targetGuides[xy*numGuideChannelsTotal + c + 2] = guides[i].targetData[xy * step + 2]; }
			if (numChannels > 3) { targetGuides[xy*numGuideChannelsTotal + c + 3] = guides[i].targetData[xy * step + 3]; }

			c += numChannels;
		}
	}

	std::vector<float> styleWeights(numStyleChannelsTotal);
	//if (isnan(styleWeight)) 
	//{ 
		//styleWeight = 1.0f; 
	//}
	for (int i = 0; i < numStyleChannelsTotal; i++) { styleWeights[i] = styleWeight / float(numStyleChannelsTotal); }

	for (int i = 0; i < numGuides; i++) { if (isnan(guides[i].weight)) { guides[i].weight = 1.0f / float(numGuides); } }

	std::vector<float> guideWeights(numGuideChannelsTotal);
	{
		int c = 0;
		for (int i = 0; i < numGuides; i++)
		{
			const int numChannels = guides[i].numChannels;

			for (int j = 0; j < numChannels; j++)
			{
				guideWeights[c + j] = guides[i].weight / float(numChannels);
			}

			c += numChannels;
		}
	}

	int maxPyramidLevels = 0;
	for (int level = 32; level >= 0; level--)
	{
		//TODO: Test if it is correct
		std::pair<int, int> pyr = pyramidLevelSize(std::min(std::pair<int, int>(sourceWidth, sourceHeight), std::pair<int, int>(targetWidth, targetHeight)), level);
		//if (min(pyramidLevelSize(std::min(V2i(sourceWidth, sourceHeight), V2i(targetWidth, targetHeight)), level)) >= (2 * patchSize + 1))
		if (std::min(pyr.first, pyr.second) >= (2 * patchSize + 1))
		{
			maxPyramidLevels = level + 1;
			break;
		}
	}

	if (numPyramidLevels == -1) 
	{ 
		numPyramidLevels = maxPyramidLevels; 
	}
	numPyramidLevels = std::min(numPyramidLevels, maxPyramidLevels);

	std::vector<int> numSearchVoteItersPerLevel(numPyramidLevels);
	std::vector<int> numPatchMatchItersPerLevel(numPyramidLevels);
	std::vector<int> stopThresholdPerLevel(numPyramidLevels);
	for (int i = 0; i < numPyramidLevels; i++)
	{
		numSearchVoteItersPerLevel[i] = numSearchVoteIters; //std::max(numSearchVoteIters - i, 2); //numSearchVoteIters;
		numPatchMatchItersPerLevel[i] = numPatchMatchIters;
		stopThresholdPerLevel[i] = stopThreshold;
	}

	cv::Mat output = cv::Mat(targetHeight, targetWidth, CV_8UC3);

	/*const unsigned char NOT_INITIALIZED_CHECK_VALUE = 99;
	for (int i = 0; i < targetWidth*targetHeight * numStyleChannelsTotal; i++)
	{
		output[i] = NOT_INITIALIZED_CHECK_VALUE;
	}*/

	// TODO
	/*if (!ebsynthBackendAvailable(EBSYNTH_BACKEND_CUDA))
	{
		printf("error: the CUDA backend is not available!\n");
		std::cin.get();
		return NULL;
	}*/

	int EBSYNTH_BACKEND;

	if (patchBasedBackend == "CUDA") 
	{
		EBSYNTH_BACKEND = EBSYNTH_BACKEND_CUDA;
	}
	else if (patchBasedBackend == "CPU")
	{
		EBSYNTH_BACKEND = EBSYNTH_BACKEND_CPU;
	}
	else if (patchBasedBackend == "AUTO") 
	{
		EBSYNTH_BACKEND = EBSYNTH_BACKEND_AUTO;
	}
	else 
	{
		errorMessage = "Internal error 9625";
		return cv::Mat();
	}

	std::vector<int> finalNNF(targetWidth*targetHeight * 2);

	//TimeMeasure ebSynthTimer;
	ebsynthRun(EBSYNTH_BACKEND,
		numStyleChannelsTotal,
		numGuideChannelsTotal,
		sourceWidth,
		sourceHeight,
		sourceStyle.data(),
		sourceGuides.data(),
		targetWidth,
		targetHeight,
		targetGuides.data(),
		NULL,
		styleWeights.data(),
		guideWeights.data(),
		uniformityWeight,
		patchSize,
		EBSYNTH_VOTEMODE_PLAIN,
		numPyramidLevels,
		numSearchVoteItersPerLevel.data(),
		numPatchMatchItersPerLevel.data(),
		stopThresholdPerLevel.data(),
		0,
		subsampleCoefficient > 1 ? finalNNF.data() : NULL, // col, row
		output.data);
	//float ebSynthTime = ebSynthTimer.elapsed_milliseconds() / 1000.0f;

	/*for (int i = 0; i < targetWidth*targetHeight * numStyleChannelsTotal; i++)
	{
		if (output[i] != NOT_INITIALIZED_CHECK_VALUE)
		{
			return output;
		}
	}*/

	//TimeMeasure upsampleTimer;
	if (subsampleCoefficient > 1)
	{
		output = UpsampleIfNecessaty(finalNNF, targetHeight, targetWidth, subsampleCoefficient, patchSize, originalStyle, originalTarget_rows, originalTarget_cols);
		cout << "NNF was upscaled " << subsampleCoefficient << " times" << endl;
	}
	else 
	{
		cout << "NNF does not need to be upscaled" << endl;
	}
	//float upsampleTime = upsampleTimer.elapsed_milliseconds() / 1000.0f;

	//system(string(string("ECHO") +  
	//	+ " ebsynth=" + std::to_string(ebSynthTime) 
	//	+ " upsample=" + std::to_string(upsampleTime) 
	//	+ " patchBasedMaxMP=" + std::to_string(patchBasedMaxMP) 
	//	+ " subsampleCoeff=" + std::to_string(subsampleCoefficient)
	//	+ " pyrLvls=" + std::to_string(numPyramidLevels)
	//	+ " & pause").c_str());

	//DEBUG_Save_All(exposeGuidePath, sourceStyleMat, sources, targets, output);

	return output;
}


int evalNumChannels(const unsigned char* data, const int numPixels)
{
	bool isGray = true;
	//bool hasAlpha = false;

	// There is no ALPHA in data
	const int step = 3;

	for (int xy = 0; xy < numPixels; xy++)
	{
		const unsigned char r = data[xy * step + 0];
		const unsigned char g = data[xy * step + 1];
		const unsigned char b = data[xy * step + 2];
		//const unsigned char a = data[xy * step + 3];

		if (!(r == g && g == b)) 
		{ 
			isGray = false; 
		}
		//if (a<255) { hasAlpha = true; }
	}

	const int numChannels = (isGray ? 1 : 3) /* + (hasAlpha ? 1 : 0)*/;

	return numChannels;
}

std::pair<int, int> pyramidLevelSize(const std::pair<int, int>& sizeBase, const int level)
{
	return std::pair<int, int>((float)sizeBase.first * pow(2.0f, -float(level)), (float)sizeBase.second * pow(2.0f, -float(level)));
}
