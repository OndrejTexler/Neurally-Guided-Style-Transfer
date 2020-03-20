#include <iostream>				// cout
#include "opencv2/opencv.hpp"	// cv::Mat
#include "cxxopts.hpp"			// cxxopts commandline parser
#include "OpenCVUtils.h"		// Imread
#include "CreateGuideUtils.h"
#include "EbsynthWrapper.h"

using namespace std;


/*
* Specifies argument options and parse argc/argv into a cxxopts::ParseResult object
*/
cxxopts::ParseResult Parse(int argc, char* argv[], std::string& errorMessage)
{
	errorMessage.clear();
	try
	{
		cxxopts::Options options(argv[0], " - Neurally-Guided-Style-Transfer");

		options
			.add_options()
			("style", "Style path", cxxopts::value<std::string>())
			("neural_result", "Path to the existing neural result image", cxxopts::value<std::string>())
			("target", "Target path", cxxopts::value<std::string>())
			("guide_by_target", "Guide patch based synthesis by the original target")
			("recolor_by_target", "Recolor the final result by the oritinal taarget")
			("patch_based_source_blur", "patch_based_source_blur", cxxopts::value<int>())
			("patch_based_style_weight", "patch_based_style_weight", cxxopts::value<float>())
			("patch_based_max_mp", "patch_based_max_mp", cxxopts::value<float>())
			("patch_based_backend", "patch_based_backend", cxxopts::value<std::string>())
			("out_path", "out_path", cxxopts::value<std::string>())
			("help", "help")
			;

		cxxopts::ParseResult result = options.parse(argc, argv);

		if (argc > 1)
		{
			cout << "Warning: " << argc - 1 << " argument(s) was/were not parsed." << endl;
		}

		return result;

	}
	catch (const cxxopts::OptionException & e)
	{
		stringstream ss;
		ss << "error parsing options: " << e.what();
		errorMessage = ss.str();
		return cxxopts::ParseResult();
	}
}


/*
* Creates additional gray-scale guidance channels (if guideByTarget is set)
* Creates guidance channel from neural result
* Calls patch based synthesis framework (EBSynth) with the aforementioned guidance channels
* Reolors the result (if recolorByTarget is set)
*/
cv::Mat Stylize(const cv::Mat& style, cv::Mat neural_result, const cv::Mat& target, bool guideByTarget, bool recolorByTarget,
	int patchBasedSourceBlur, float patchBasedMaxMP, float patchBasedStyleWeight, std::string patchBasedBackend, std::string& errorMessage)
{
	vector<cv::Mat> sources;
	vector<cv::Mat> targets;

	//### GREY SCALE GUIDE ###
	if (guideByTarget)
	{
		int levelOfAbstraction = 4;
		pair<cv::Mat, cv::Mat> grayScaleGuide = CreateGrayScaleGuide(style, target, levelOfAbstraction);

		sources.push_back(grayScaleGuide.first);
		targets.push_back(grayScaleGuide.second);
		//	Imwrite(exposeGuidePath + "GREY_SCALE_Source.png", grayScaleGuide.first, true);
		//	Imwrite(exposeGuidePath + "GREY_SCALE_Target.png", grayScaleGuide.second, true);
	}

	//### NEURAL GUIDE ###	
	cv::Mat sourceForEbsynth;
	style.copyTo(sourceForEbsynth);

	float neuralToStyleRatio = (float)neural_result.cols / (float)style.cols;
	// Both, the source and the target in neural guiding pair should have similar amount of blurriness. 
	cv::resize(sourceForEbsynth, sourceForEbsynth, cv::Size(style.cols * neuralToStyleRatio, style.rows * neuralToStyleRatio)); // Subsample source by the same coefficient as neural_result was subsampled.
	cv::resize(sourceForEbsynth, sourceForEbsynth, style.size(), cv::InterpolationFlags::INTER_CUBIC); // Then upsample it back in the same way as neural_result is upsampled.

	cv::resize(neural_result, neural_result, style.size(), cv::InterpolationFlags::INTER_CUBIC); // Upsample neural_result to the same size as target

	if (patchBasedSourceBlur > 1)
	{
		int kernelSize = (patchBasedSourceBlur * 2) - 1; // Make it odd
		cv::GaussianBlur(sourceForEbsynth, sourceForEbsynth, cv::Size(kernelSize, kernelSize), 0, 0);
	}

	sources.push_back(sourceForEbsynth);
	targets.push_back(neural_result);

	//### RECOLOR TARGET ###
	cv::Mat grayStyle;
	if (recolorByTarget) {
		style.copyTo(grayStyle);
		cv::cvtColor(grayStyle, grayStyle, cv::COLOR_BGR2GRAY);
		cv::equalizeHist(grayStyle, grayStyle);

		//Convert back to 3-channel, although it is still grays-cale
		cv::cvtColor(grayStyle, grayStyle, cv::COLOR_GRAY2BGR);
	}

	cv::Mat ebsynthOutput = CallEbsynth(recolorByTarget ? grayStyle : style, sources, targets, patchBasedMaxMP, patchBasedStyleWeight, patchBasedBackend, errorMessage);

	if (ebsynthOutput.empty())
	{
		if (errorMessage.empty())
		{
			errorMessage = "Unspecified patch_based_synthesis error. Try a smaller image or run it with argument --patch_based_backend \"CPU\".";
		}
		return cv::Mat();
	}

	if (recolorByTarget)
	{
		cv::Mat inColorMat;
		target.copyTo(inColorMat);
		Recolor(ebsynthOutput, inColorMat);
	}

	return ebsynthOutput;
}

/*
* Parse cxxopts::ParseResult (obtained by calling Parse funcion)
* Reads the input images
* Calls Stylize function
*/
cv::Mat ParseAndRun(const cxxopts::ParseResult& parseResult, string& errorMessage)
{
	string style_path;
	string neuralResultPath;
	string target_path;
	bool guideByTarget = false;
	bool recolorByTarget = false;
	int patchBasedSourceBlur = 4;
	float patchBasedStyleWeight = 1.0f;
	float patchBasedMaxMP = 4.0f;
	string patchBasedBackend = "AUTO";


	if (parseResult.count("style"))
	{
		style_path = parseResult["style"].as<string>();
	}
	else
	{
		errorMessage = "Argument --style must be specified";
		return cv::Mat();
	}

	if (parseResult.count("neural_result"))
	{
		neuralResultPath = parseResult["neural_result"].as<string>();
	}
	else
	{
		errorMessage = "Argument --neural_result must be specified";
		return cv::Mat();
	}

	if (parseResult.count("target"))
	{
		target_path = parseResult["target"].as<string>();
	}

	if (parseResult.count("guide_by_target"))
	{
		if (!parseResult.count("target")) 
		{
			errorMessage = string("Argument --target has to be specified if --guide_by_target is set");
			return cv::Mat();
		}
		guideByTarget = true;
	}

	if (parseResult.count("recolor_by_target"))
	{
		if (!parseResult.count("target"))
		{
			errorMessage = string("Argument --target has to be specified if --recolor_by_target is set");
			return cv::Mat();
		}
		recolorByTarget = true;
	}

	if (parseResult.count("patch_based_source_blur"))
	{
		patchBasedSourceBlur = parseResult["patch_based_source_blur"].as<int>();
	}

	if (parseResult.count("patch_based_style_weight"))
	{
		patchBasedStyleWeight = parseResult["patch_based_style_weight"].as<float>();
	}

	if (parseResult.count("patch_based_max_mp"))
	{
		patchBasedMaxMP = parseResult["patch_based_max_mp"].as<float>();
	}

	if (parseResult.count("patch_based_backend"))
	{
		patchBasedBackend = parseResult["patch_based_backend"].as<string>();
		if (patchBasedBackend != "AUTO" && patchBasedBackend != "CUDA" && patchBasedBackend != "CPU") 
		{
			errorMessage = string("Argument --patch_based_backend has to be \"AUTO\", \"CUDA\", or \"CPU\". Not ") + patchBasedBackend;
			return cv::Mat();
		}
	}

	const cv::Mat style = Imread(style_path, true);
	if (style.empty())
	{
		errorMessage = string("Failed to read style image: ") + style_path;
		return cv::Mat();
	}

	const cv::Mat neural_result = Imread(neuralResultPath, true);
	if (neural_result.empty())
	{
		errorMessage = string("Failed to read neural_result image: ") + neuralResultPath;
		return cv::Mat();
	}

	cv::Mat target = cv::Mat();
	if (parseResult.count("target"))
	{
		target = Imread(target_path, true);
		if (target.empty())
		{
			errorMessage = string("Failed to read target image: ") + target_path;
			return cv::Mat();
		}
	}

	cv::Mat output = Stylize(style, neural_result, target, guideByTarget, recolorByTarget,
		patchBasedSourceBlur, patchBasedMaxMP, patchBasedStyleWeight,
		patchBasedBackend, errorMessage);

	return output;
}


int main(int argc, char* argv[])
{
	std::string errorMessage;
	cxxopts::ParseResult parseResult = Parse(argc, argv, errorMessage);

	if (!errorMessage.empty())
	{
		cout << "Fatal parse error: " << errorMessage << endl;
		return -1;
	}

	if (parseResult.count("help"))
	{
		cout << "Everything you need to know is at: https://github.com/OndrejTexler/Neurally-Guided-Style-Transfer" << endl;
		return 0;
	}

	cv::Mat result = ParseAndRun(parseResult, errorMessage);

	if (result.empty())
	{
		cout << "FATAL ERROR: " << errorMessage << endl;
		return 0;
	}
	
	string out_path = parseResult["neural_result"].as<string>() + "_enhanced.png";
	if (parseResult.count("out_path"))
	{
		out_path = parseResult["out_path"].as<string>();
	}

	Imwrite(out_path, result, true);
	cout << "Result written to: " << out_path << endl;
	

	return 1;
}