#include <iostream>				//cout
#include "opencv2/opencv.hpp"	//cv::Mat
#include "StyLitBackEnd.h"		//Original, Fluids, Photo
#include "StyLitCoreCommands.h"	//namespace Usecase
#include "cxxopts.hpp"			//cxxopts commandline parser
#include "OpenCVUtils.h"		//Imread

using namespace std;

/*
struct my_pair
{
	std::string first;
	std::string second;
};

std::istream& operator>>(std::istream& stream, my_pair& pair)
{
	stream >> pair.first;
	stream >> std::skipws;
	stream >> pair.second;
	return stream;
}

if (result.count("guide"))
{
	for (int i = 0; i < result.count("guide"); i++)
	{
		my_pair guide_arg = result["guide"].as<my_pair>();
		std::cout << "Guide_arg: " << guide_arg.first << ", " << guide_arg.second << std::endl;
	}
}*/


cv::Mat Photo(const cv::Mat& style, const cv::Mat& target, std::string styLitPhotoCEPPath, bool useGrayScale, bool useColorRegions, bool useUST, std::string USTResultPath,
	bool exposeGuideImages, bool recolorResult, int levelOfAbstraction,
	int advanced_USTShorterSide, float advanced_USTAplha, bool advanced_UseNewUST, int advanced_USTSourceBlur,
	float advanced_StyLitMaxMP, float advanced_StyleWeight, std::string advanced_StylitBackend, std::string& errorMessage)
{
	const std::string exposeGuidePath = styLitPhotoCEPPath + "guides/";

	vector<cv::Mat> sources;
	vector<cv::Mat> targets;

	//### GREY SCALE GUIDE ###
	if (useGrayScale)
	{
		pair<cv::Mat, cv::Mat> grayScaleGuide = CreateGrayScaleGuide(style, target, levelOfAbstraction);

		sources.push_back(grayScaleGuide.first);
		targets.push_back(grayScaleGuide.second);

		if (exposeGuideImages)
		{
			Imwrite(exposeGuidePath + "GREY_SCALE_Source.png", grayScaleGuide.first, true);
			Imwrite(exposeGuidePath + "GREY_SCALE_Target.png", grayScaleGuide.second, true);
		}
	}

	//### COLOR REGIONS GUIDE ###	
	if (useColorRegions)
	{
		cv::Mat styleColorRegionsGuide;
		cv::Mat targetColorRegionsGuide;
		style.copyTo(styleColorRegionsGuide);
		target.copyTo(targetColorRegionsGuide);

		CreateColorRegionsGuide(styleColorRegionsGuide, targetColorRegionsGuide);

		sources.push_back(styleColorRegionsGuide);
		targets.push_back(targetColorRegionsGuide);

		if (exposeGuideImages)
		{
			Imwrite(exposeGuidePath + "COLOR_REGIONS_Source.png", styleColorRegionsGuide, true);
			Imwrite(exposeGuidePath + "COLOR_REGIONS_Target.png", targetColorRegionsGuide, true);
		}
	}


	//### UST GUIDE ###	
	if (useUST)
	{
		cv::Mat UST_result;
		if (!USTResultPath.empty())
		{
			UST_result = FakeRunningUST(USTResultPath, advanced_USTShorterSide);
			if (UST_result.empty())
			{
				errorMessage = "The FakeRunningUST() method failed.";
				return cv::Mat();
			}
		}
		else
		{
			UST_result = RunUST(style, target, styLitPhotoCEPPath, advanced_USTShorterSide, advanced_USTAplha, advanced_UseNewUST);
			if (UST_result.empty())
			{
				errorMessage = "The UST failed to run. Please, check UST installation and configuration.";
				return cv::Mat();
			}
		}

		cv::Mat sourceForStyLit;
		style.copyTo(sourceForStyLit);

		float USTSubsampleCoeff = (float)UST_result.cols / (float)target.cols;
		// Both, the source and the target in UST guiding pair should have similar amount of blurriness. 
		cv::resize(sourceForStyLit, sourceForStyLit, cv::Size(style.cols * USTSubsampleCoeff, style.rows * USTSubsampleCoeff)); // Subsample sourceForUST by the same coefficient as UST_result was subsampled.
		cv::resize(sourceForStyLit, sourceForStyLit, style.size(), cv::InterpolationFlags::INTER_CUBIC); // Then upsample it back in the same way as UST_result is upsampled.

		cv::resize(UST_result, UST_result, target.size(), cv::InterpolationFlags::INTER_CUBIC); // Upsample UST_result to the same size as target

		if (advanced_USTSourceBlur > 1)
		{
			int kernelSize = (advanced_USTSourceBlur * 2) - 1; // Make it odd
			cv::GaussianBlur(sourceForStyLit, sourceForStyLit, cv::Size(kernelSize, kernelSize), 0, 0);
		}

		//TODO: Maybe also modify the UST_result

		sources.push_back(sourceForStyLit);
		targets.push_back(UST_result);

		if (exposeGuideImages)
		{
			Imwrite(exposeGuidePath + "SENSEI_SOURCE.png", sourceForStyLit, true);
			Imwrite(exposeGuidePath + "SENSEI_TARGET.png", UST_result, true);
		}
	}

	//### RECOLOR TARGET ###
	cv::Mat grayStyle;
	if (recolorResult) {
		style.copyTo(grayStyle);
		//Convert target to gray-scale
		cv::cvtColor(grayStyle, grayStyle, cv::COLOR_BGR2GRAY);
		cv::equalizeHist(grayStyle, grayStyle);

		//Convert back to 3-channel, it is still grays-cale, but StyLit needs 3-channel
		cv::cvtColor(grayStyle, grayStyle, cv::COLOR_GRAY2BGR);
	}

	cv::Mat stylitOutput = CallStyLit(recolorResult ? grayStyle : style, sources, targets, advanced_StyLitMaxMP, advanced_StyleWeight, exposeGuidePath, advanced_StylitBackend, errorMessage);

	if (stylitOutput.empty())
	{
		if (errorMessage.empty())
		{
			errorMessage = "Unspecified StyLit back-end fail. Maybe your image is too big for your graphic card. Please try smaller image but restart Photoshop at first.";
		}
		return cv::Mat();
	}

	if (recolorResult)
	{
		cv::Mat inColorMat;
		target.copyTo(inColorMat);
		Recolor(stylitOutput, inColorMat);
	}

	//cv::destroyAllWindows();
	return stylitOutput;
}


cxxopts::ParseResult parse(int argc, char* argv[], std::string& errorMessage)
{
	errorMessage.clear();
	try
	{
		cxxopts::Options options(argv[0], " - StyLit: Commandline Interface");

		options
			.add_options()
			("usecase", "StyLit Usecase", cxxopts::value<std::string>())
			("style", "Style path", cxxopts::value<std::string>())
			("target", "target path", cxxopts::value<std::string>())
			("styLitPhotoCEPPath", "StyLit Photo CEP Path", cxxopts::value<std::string>())
			("useGrayScale", "Use GrayScale")
			("useColorRegions", "Use ColorRegions")
			("useUST", "Use UST")
			("USTResultPath", "Path to the existing UST result image", cxxopts::value<std::string>())
			("exposeGuideImages", "exposeGuideImages")
			("recolorResult", "recolorResult")
			("levelOfAbstraction", "levelOfAbstraction", cxxopts::value<int>())
			("advanced_USTShorterSide", "advanced_USTShorterSide", cxxopts::value<int>())
			("advanced_USTAplha", "advanced_USTAplha", cxxopts::value<float>())
			("advanced_UseNewUST", "advanced_UseNewUST")
			("advanced_USTSourceBlur", "advanced_USTSourceBlur", cxxopts::value<int>())
			("advanced_StyleWeight", "advanced_StyleWeight", cxxopts::value<float>())
			("advanced_StyLitMaxMP", "advanced_StyLitMaxMP", cxxopts::value<float>())
			("advanced_StylitBackend", "advanced_StylitBackend", cxxopts::value<std::string>())
			("out_path", "out_path", cxxopts::value<std::string>())

			("source1", "source1", cxxopts::value<std::string>())
			("target1", "target1", cxxopts::value<std::string>())
			("source2", "source2", cxxopts::value<std::string>())
			("target2", "target2", cxxopts::value<std::string>())
			("source3", "source3", cxxopts::value<std::string>())
			("target3", "target3", cxxopts::value<std::string>())
			("source4", "source4", cxxopts::value<std::string>())
			("target4", "target4", cxxopts::value<std::string>())

			("fluidStylePath", "fluidStylePath", cxxopts::value<std::string>())

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


cv::Mat PhotoUsecaseParseAndRun(const cxxopts::ParseResult& parseResult, string& errorMessage)
{
	string style_path;
	string target_path;
	string styLitPhotoCEPPath = "";
	bool useGrayScale = false;
	bool useColorRegions = false;
	bool useUST = false;
	string USTResultPath = "";
	bool exposeGuideImages = false;
	bool recolorResult = false;
	int levelOfAbstraction = 4;
	int advanced_USTShorterSide = 400;
	float advanced_USTAplha = 1.0f;
	bool advanced_UseNewUST = true;
	int advanced_USTSourceBlur = 4;
	float advanced_StyleWeight = 1.0f;
	float advanced_StyLitMaxMP = 4.0f;
	string advanced_StylitBackend = "AUTO";


	if (parseResult.count("style"))
	{
		style_path = parseResult["style"].as<string>();
	}
	else
	{
		errorMessage = "Fatal error: option --style must be specified in Phot usecase";
		return cv::Mat();
	}

	if (parseResult.count("target"))
	{
		target_path = parseResult["target"].as<string>();
	}
	else
	{
		errorMessage = "Fatal error: option --target must be specified in Phot usecase";
		return cv::Mat();
	}

	if (parseResult.count("styLitPhotoCEPPath"))
	{
		styLitPhotoCEPPath = parseResult["styLitPhotoCEPPath"].as<string>();
	}

	if (parseResult.count("useGrayScale"))
	{
		useGrayScale = true;
	}

	if (parseResult.count("useColorRegions"))
	{
		useColorRegions = true;
	}

	if (parseResult.count("useUST"))
	{
		useUST = true;
	}

	if (parseResult.count("USTResultPath"))
	{
		USTResultPath = parseResult["USTResultPath"].as<string>();
	}

	if (!useGrayScale && !useColorRegions && !useUST)
	{
		errorMessage = "Fatal error: at least one of following options must be specified: --useGrayScale, --useColorRegions, --useUST";
		return cv::Mat();
	}

	if (parseResult.count("exposeGuideImages"))
	{
		exposeGuideImages = true;
	}

	if (parseResult.count("recolorResult"))
	{
		recolorResult = true;
	}

	if (parseResult.count("levelOfAbstraction"))
	{
		levelOfAbstraction = parseResult["levelOfAbstraction"].as<int>();
	}

	if (parseResult.count("advanced_USTShorterSide"))
	{
		advanced_USTShorterSide = parseResult["advanced_USTShorterSide"].as<int>();
	}

	if (parseResult.count("advanced_USTAplha"))
	{
		advanced_USTAplha = parseResult["advanced_USTAplha"].as<float>();
	}

	if (parseResult.count("advanced_UseNewUST"))
	{
		advanced_UseNewUST = true;
	}

	if (parseResult.count("advanced_USTSourceBlur"))
	{
		advanced_USTSourceBlur = parseResult["advanced_USTSourceBlur"].as<int>();
	}

	if (parseResult.count("advanced_StyleWeight"))
	{
		advanced_StyleWeight = parseResult["advanced_StyleWeight"].as<float>();
	}

	if (parseResult.count("advanced_StyLitMaxMP"))
	{
		advanced_StyLitMaxMP = parseResult["advanced_StyLitMaxMP"].as<float>();
	}

	if (parseResult.count("advanced_StylitBackend"))
	{
		advanced_StylitBackend = parseResult["advanced_StylitBackend"].as<string>();
	}

	const cv::Mat style = Imread(style_path, true);
	if (style.empty())
	{
		errorMessage = string("Failed to read style image: ") + style_path;
		return cv::Mat();
	}

	const cv::Mat target = Imread(target_path, true);
	if (target.empty())
	{
		errorMessage = string("Failed to read target image: ") + target_path;
		return cv::Mat();
	}

	if (useUST)
	{
		if (styLitPhotoCEPPath.empty() && USTResultPath.empty())
		{
			errorMessage = "When --useUST parameter is present, either --styLitPhotoCEPPath or --USTResultPath must be specified";
			return cv::Mat();
		}
		if (!styLitPhotoCEPPath.empty() && !USTResultPath.empty())
		{
			errorMessage = "When --useUST parameter is present, exactly one of the --styLitPhotoCEPPath and --USTResultPath must be specified";
			return cv::Mat();
		}
	}

	cv::Mat stylitOutput = Photo(style, target, styLitPhotoCEPPath, useGrayScale, useColorRegions, useUST, USTResultPath, exposeGuideImages, recolorResult, levelOfAbstraction,
		advanced_USTShorterSide, advanced_USTAplha, advanced_UseNewUST, advanced_USTSourceBlur, advanced_StyLitMaxMP, advanced_StyleWeight,
		advanced_StylitBackend, errorMessage);

	return stylitOutput;
}


int main(int argc, char* argv[])
{
	/*string imagepath = "C:\\Users\\Ondrej\\Desktop\\stylit_plugin\\Paper_res\\wolf\\wolf1.jpg";
	cv::Mat image = Imread(imagepath, false);

	imshowInWindow("Image", image, 600, 400);
	cv::waitKey();

	return -99;*/

	// HACK BEGIN
	/*cv::Mat nnf_stylit = Imread("2019-03-03.png", false);

	cv::Mat nnf_random = cv::Mat::zeros(nnf_stylit.rows, nnf_stylit.cols, CV_8UC3);

	for (int row = 0; row < nnf_stylit.rows; row++)
	{
		for (int col = 0; col < nnf_stylit.cols; col++)
		{
			cv::Vec3b nn = nnf_stylit.at<cv::Vec3b>(row, col); // B G R

			// hash colors
			nnf_random.at<cv::Vec3b>(row, col) = cv::Vec3b(
				((nn[2] * nn[1] * 75 + nn[2] * 1234 + nn[1] * 147 + 495) % 255),
				((nn[2] * nn[1] * 15 + nn[2] * 6589 + nn[1] * 852 + 658) % 255),
				((nn[2] * nn[1] * 39 + nn[2] * 4584 + nn[1] * 369 + 364) % 255)
			);
		}
	}

	Imwrite("2019-03-03_hash.png", nnf_random, false);

	return -99;*/

	// HACK END


	std::string errorMessage;
	cxxopts::ParseResult parseResult = parse(argc, argv, errorMessage);

	if (!errorMessage.empty())
	{
		cout << "Fatal Parse Error - " << errorMessage << endl;
		return -1;
	}

	if (parseResult.count("help"))
	{
		cout << "TODO: Support 'help' parameter" << endl;
		return 0;
	}

	string usecase;
	if (parseResult.count("usecase"))
	{
		usecase = parseResult["usecase"].as<string>();
	}
	else
	{
		cout << "FATAL ERROR: parameter --usecase must be specified" << endl;
		return -1;
	}

	string out_path = "output.png";
	if (parseResult.count("out_path"))
	{
		out_path = parseResult["out_path"].as<string>();
	}

	cv::Mat stylitOutput;
	if (usecase == Usecase::Photo)
	{
		stylitOutput = PhotoUsecaseParseAndRun(parseResult, errorMessage);
	}
	else
	{
		cout << "FATAL ERROR: Unknown Usecase: " << usecase << endl;
		return -1;
	}

	if (stylitOutput.empty())
	{
		cout << "FATAL ERROR: " << errorMessage << endl;
	}
	else
	{
		Imwrite(out_path, stylitOutput, true);
		cout << "Result written to: " << out_path << endl;
	}

	return 0;
}