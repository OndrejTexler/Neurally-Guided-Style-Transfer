#include "CreateGuideUtils.h"

#include "opencv2/opencv.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "OpenCVUtils.h" // isOutOfImage

#include <omp.h>

using namespace cv;
using namespace std;

//Private functions
Mat1f SegmentImage(const Mat& mat, Mat& labels);
bool ImageMatHasResponse(const Mat& mat);
Mat2i upsampleNNF(const Mat2i& NNFSmall, const int coeficient);
Mat voting(const Mat& style, const Mat2i& NNF, const int NNF_patchsize);


// TODO: Actually, EBSYNTH needs 3-channel no more, remove the last COLOR_GRAY2BGR conversion 
pair<Mat, Mat> CreateGrayScaleGuide(Mat source, Mat target, int levelOfAbstraction)
{
	//Convert to gray-scale
	cv::cvtColor(source, source, cv::COLOR_BGR2GRAY);
	cv::cvtColor(target, target, cv::COLOR_BGR2GRAY);

	// Blur the source 
	if (levelOfAbstraction > 1) {
		int kernelSize = (levelOfAbstraction * 2) - 1; // Make it odd
		cv::GaussianBlur(source, source, cv::Size(kernelSize, kernelSize), 0);
	}

	// Equalize histograms
	//cv::equalizeHist(source, source);
	//cv::equalizeHist(target, target);

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

void CreateColorRegionsGuide(Mat& style, Mat& target)
{
	Mat style_labels;
	Mat target_labels;

	//DebugModalMsg("segmenting");
	Mat1f style_segments = SegmentImage(style, style_labels);
	Mat1f target_segments = SegmentImage(target, target_labels);
	//DebugModalMsg("segmented");

	//ostringstream oss;
	//
	//oss << "### STYLE_LABELS ###" << endl;
	//oss << style_segments << endl;
	//oss << "### TARGET_LABELS ###" << endl;
	//oss << target_segments << endl;
	//
	//string msg = oss.str();
	//
	//DebugModalMsg(msg);
	//
	//oss.str("");

	//oss << "### STYLE ###" << endl;
	//oss << style_segments << endl << endl;
	//oss << "### TARGET ###" << endl;
	//oss << target_segments << endl << endl;
	//
	//msg = oss.str();
	//
	//DebugModalMsg(msg);

	Mat flow;

	// Compute cost Mat by L2 distance
	Mat1f cost(style_segments.rows, target_segments.rows);
	{
		for (int row = 0; row < cost.rows; row++)
		{
			for (int col = 0; col < cost.cols; col++)
			{
				Vec3b style_segment_color((uchar)style_segments.at<float>(row, 1), (uchar)style_segments.at<float>(row, 2), (uchar)style_segments.at<float>(row, 3));
				Vec3b target_segment_color((uchar)target_segments.at<float>(col, 1), (uchar)target_segments.at<float>(col, 2), (uchar)target_segments.at<float>(col, 3));

				cost.at<float>(row, col) = (float)cv::norm(style_segment_color, target_segment_color);
			}
		}
	}

	float minimalWork = EMD(style_segments, target_segments, cv::DIST_USER, cost, 0, flow);

	//cout << "### FLOW ###" << endl;
	//cout << flow << endl << endl;

	Mat1f style_segments_interpolated;
	Mat1f target_segments_interpolated;
	style_segments.copyTo(style_segments_interpolated);
	target_segments.copyTo(target_segments_interpolated);


	for (int row = 0; row < flow.rows; row++)
	{
		float acc_R = style_segments.at<float>(row, 0) * style_segments.at<float>(row, 1);
		float acc_G = style_segments.at<float>(row, 0) * style_segments.at<float>(row, 2);
		float acc_B = style_segments.at<float>(row, 0) * style_segments.at<float>(row, 3);
		float weight = style_segments.at<float>(row, 0);

		for (int col = 0; col < flow.cols; col++)
		{
			if (flow.at<float>(row, col) == 0)
			{
				continue;
			}

			acc_R += flow.at<float>(row, col) * target_segments.at<float>(col, 1);
			acc_G += flow.at<float>(row, col) * target_segments.at<float>(col, 2);
			acc_B += flow.at<float>(row, col) * target_segments.at<float>(col, 3);
			weight += flow.at<float>(row, col);
		}

		style_segments_interpolated.at<float>(row, 0) = weight / 2;
		style_segments_interpolated.at<float>(row, 1) = acc_R / weight;
		style_segments_interpolated.at<float>(row, 2) = acc_G / weight;
		style_segments_interpolated.at<float>(row, 3) = acc_B / weight;
	}

	for (int col = 0; col < flow.cols; col++)
	{
		float acc_R = target_segments.at<float>(col, 0) * target_segments.at<float>(col, 1);
		float acc_G = target_segments.at<float>(col, 0) * target_segments.at<float>(col, 2);
		float acc_B = target_segments.at<float>(col, 0) * target_segments.at<float>(col, 3);
		float weight = target_segments.at<float>(col, 0);

		for (int row = 0; row < flow.rows; row++)
		{
			if (flow.at<float>(row, col) == 0)
			{
				continue;
			}

			acc_R += flow.at<float>(row, col) * style_segments.at<float>(row, 1);
			acc_G += flow.at<float>(row, col) * style_segments.at<float>(row, 2);
			acc_B += flow.at<float>(row, col) * style_segments.at<float>(row, 3);
			weight += flow.at<float>(row, col);
		}

		target_segments_interpolated.at<float>(col, 0) = weight / 2;
		target_segments_interpolated.at<float>(col, 1) = acc_R / weight;
		target_segments_interpolated.at<float>(col, 2) = acc_G / weight;
		target_segments_interpolated.at<float>(col, 3) = acc_B / weight;
	}

	//cout << "### STYLE_SEGMENTS_INTERPOLATED ###" << endl;
	//cout << style_segments_interpolated << endl << endl;
	//cout << "### TARGET_SEGMENTS_INTERPOLATED ###" << endl;
	//cout << target_segments_interpolated << endl << endl;

	for (int row = 0; row < style_labels.rows; row++)
	{
		for (int col = 0; col < style_labels.cols; col++)
		{
			uchar interpolated_R = (uchar)style_segments_interpolated.at<float>(style_labels.at<int>(row, col), 1);
			uchar interpolated_G = (uchar)style_segments_interpolated.at<float>(style_labels.at<int>(row, col), 2);
			uchar interpolated_B = (uchar)style_segments_interpolated.at<float>(style_labels.at<int>(row, col), 3);

			style.at<Vec3b>(row, col) = Vec3b(interpolated_R, interpolated_G, interpolated_B);
		}
	}

	for (int row = 0; row < target_labels.rows; row++)
	{
		for (int col = 0; col < target_labels.cols; col++)
		{
			uchar interpolated_R = (uchar)target_segments_interpolated.at<float>(target_labels.at<int>(row, col), 1);
			uchar interpolated_G = (uchar)target_segments_interpolated.at<float>(target_labels.at<int>(row, col), 2);
			uchar interpolated_B = (uchar)target_segments_interpolated.at<float>(target_labels.at<int>(row, col), 3);

			target.at<Vec3b>(row, col) = Vec3b(interpolated_R, interpolated_G, interpolated_B);
		}
	}
}


void ColorTransfer(Mat& result, Mat& reference)
{
	//Mat style_labels;
	//Mat target_labels;

	Mat1f style_segments(result.rows * result.cols, 1, 1.0f); //= SegmentImage(image, style_labels);
	Mat1f target_segments(reference.rows * reference.cols, 1, 1.0f); //= SegmentImage(reference, target_labels);

	//cout << "### STYLE_LABELS ###" << endl;
	//cout << style_labels << endl;
	//cout << "### TARGET_LABELS ###" << endl;
	//cout << target_labels << endl;



	//cout << "### STYLE ###" << endl;
	// << style_segments << endl << endl;
	//cout << "### TARGET ###" << endl;
	//cout << target_segments << endl << endl;

	Mat flow;

	// Compute cost Mat by L2 distance
	Mat1f* cost = new Mat1f(result.rows * result.cols, reference.rows * reference.cols);
	{
		for (int resultIndex = 0; resultIndex < cost->rows; resultIndex++)
		{
			for (int referenceIndex = 0; referenceIndex < cost->cols; referenceIndex++)
			{
				//Vec3b style_segment_color((uchar)style_segments.at<float>(row, 1), (uchar)style_segments.at<float>(row, 2), (uchar)style_segments.at<float>(row, 3));
				//Vec3b target_segment_color((uchar)target_segments.at<float>(col, 1), (uchar)target_segments.at<float>(col, 2), (uchar)target_segments.at<float>(col, 3));

				cost->at<float>(resultIndex, referenceIndex) = (float)cv::norm(  result.at<Vec3b>(resultIndex/result.cols, resultIndex%result.cols),
																				reference.at<Vec3b>(referenceIndex/reference.cols, referenceIndex%reference.cols));
			}
		}
	}

	float minimalWork = EMD(style_segments, target_segments, cv::DIST_USER, *cost, 0, flow);

	//cout << "### FLOW ###" << endl;
	//cout << flow << endl << endl;

	//Mat1f style_segments_interpolated;
	//Mat1f target_segments_interpolated;
	//style_segments.copyTo(style_segments_interpolated);
	//target_segments.copyTo(target_segments_interpolated);


	for (int row = 0; row < flow.rows; row++)
	{
		float acc_R = 0.0f; // style_segments.at<float>(row, 0) * style_segments.at<float>(row, 1);
		float acc_G = 0.0f; // style_segments.at<float>(row, 0) * style_segments.at<float>(row, 2);
		float acc_B = 0.0f; // style_segments.at<float>(row, 0) * style_segments.at<float>(row, 3);
		//float weight = 0.0f; // = style_segments.at<float>(row, 0);

		for (int col = 0; col < flow.cols; col++)
		{
			if (flow.at<float>(row, col) == 0)
			{
				continue;
			}

			acc_R += flow.at<float>(row, col) * reference.at<Vec3b>(col / reference.cols, col%reference.cols)[0]; //target_segments.at<float>(col, 1);
			acc_G += flow.at<float>(row, col) * reference.at<Vec3b>(col / reference.cols, col%reference.cols)[1]; //target_segments.at<float>(col, 2);
			acc_B += flow.at<float>(row, col) * reference.at<Vec3b>(col / reference.cols, col%reference.cols)[2]; //target_segments.at<float>(col, 3);
			//weight += flow.at<float>(row, col);
		}

		//style_segments_interpolated.at<float>(row, 0) = weight / 2;
		//style_segments_interpolated.at<float>(row, 1) = acc_R / weight;
		//style_segments_interpolated.at<float>(row, 2) = acc_G / weight;
		//style_segments_interpolated.at<float>(row, 3) = acc_B / weight;

		result.at<Vec3b>(row / result.cols, row%result.cols) = Vec3b(acc_R, acc_G, acc_B);
	}

	/*for (int col = 0; col < flow.cols; col++)
	{
		float acc_R = target_segments.at<float>(col, 0) * target_segments.at<float>(col, 1);
		float acc_G = target_segments.at<float>(col, 0) * target_segments.at<float>(col, 2);
		float acc_B = target_segments.at<float>(col, 0) * target_segments.at<float>(col, 3);
		float weight = target_segments.at<float>(col, 0);

		for (int row = 0; row < flow.rows; row++)
		{
			if (flow.at<float>(row, col) == 0)
			{
				continue;
			}

			acc_R += flow.at<float>(row, col) * style_segments.at<float>(row, 1);
			acc_G += flow.at<float>(row, col) * style_segments.at<float>(row, 2);
			acc_B += flow.at<float>(row, col) * style_segments.at<float>(row, 3);
			weight += flow.at<float>(row, col);
		}

		target_segments_interpolated.at<float>(col, 0) = weight / 2;
		target_segments_interpolated.at<float>(col, 1) = acc_R / weight;
		target_segments_interpolated.at<float>(col, 2) = acc_G / weight;
		target_segments_interpolated.at<float>(col, 3) = acc_B / weight;
	}*/

	//cout << "### STYLE_SEGMENTS_INTERPOLATED ###" << endl;
	//cout << style_segments_interpolated << endl << endl;
	//cout << "### TARGET_SEGMENTS_INTERPOLATED ###" << endl;
	//cout << target_segments_interpolated << endl << endl;

	/*for (int row = 0; row < style_labels.rows; row++)
	{
		for (int col = 0; col < style_labels.cols; col++)
		{
			uchar interpolated_R = (uchar)style_segments_interpolated.at<float>(style_labels.at<int>(row, col), 1);
			uchar interpolated_G = (uchar)style_segments_interpolated.at<float>(style_labels.at<int>(row, col), 2);
			uchar interpolated_B = (uchar)style_segments_interpolated.at<float>(style_labels.at<int>(row, col), 3);

			result.at<Vec3b>(row, col) = Vec3b(interpolated_R, interpolated_G, interpolated_B);
		}
	}

	for (int row = 0; row < target_labels.rows; row++)
	{
		for (int col = 0; col < target_labels.cols; col++)
		{
			uchar interpolated_R = (uchar)target_segments_interpolated.at<float>(target_labels.at<int>(row, col), 1);
			uchar interpolated_G = (uchar)target_segments_interpolated.at<float>(target_labels.at<int>(row, col), 2);
			uchar interpolated_B = (uchar)target_segments_interpolated.at<float>(target_labels.at<int>(row, col), 3);

			reference.at<Vec3b>(row, col) = Vec3b(interpolated_R, interpolated_G, interpolated_B);
		}
	}*/
}


vector<pair<Mat, Mat>> CreateContentResponseGuide(const Mat& style, const Mat& target) 
{
	Imwrite("C:\\Users\\texler\\Desktop\\scenesegmentation\\input\\style.png", style, true);
	system("docker run --rm -v C:\\Users\\texler\\Desktop\\scenesegmentation:/workspace/scenesegmentation scene-parsing:latest /bin/bash -c \"cd /workspace/scenesegmentation; python example.py input/style.png\"");
	cv::Mat style_fc_mountain = Imread("C:\\Users\\texler\\Desktop\\scenesegmentation\\dense\\segmentation_mountain.png", true);
	cv::Mat style_fc_sky = Imread("C:\\Users\\texler\\Desktop\\scenesegmentation\\dense\\segmentation_sky.png", true);
    cv::Mat style_fc_tree = Imread("C:\\Users\\texler\\Desktop\\scenesegmentation\\dense\\segmentation_tree.png", true);
	cv::Mat style_fc_water = Imread("C:\\Users\\texler\\Desktop\\scenesegmentation\\dense\\segmentation_water.png", true);

	Imwrite("C:\\Users\\texler\\Desktop\\scenesegmentation\\input\\target.png", target, true);
	system("docker run --rm -v C:\\Users\\texler\\Desktop\\scenesegmentation:/workspace/scenesegmentation scene-parsing:latest /bin/bash -c \"cd /workspace/scenesegmentation; python example.py input/target.png\"");
	cv::Mat target_fc_mountain = Imread("C:\\Users\\texler\\Desktop\\scenesegmentation\\dense\\segmentation_mountain.png", true);
	cv::Mat target_fc_sky = Imread("C:\\Users\\texler\\Desktop\\scenesegmentation\\dense\\segmentation_sky.png", true);
	cv::Mat target_fc_tree = Imread("C:\\Users\\texler\\Desktop\\scenesegmentation\\dense\\segmentation_tree.png", true);
	cv::Mat target_fc_water = Imread("C:\\Users\\texler\\Desktop\\scenesegmentation\\dense\\segmentation_water.png", true);

	vector<cv::Mat> styleFcs;
	vector<cv::Mat> targetFcs;

	styleFcs.push_back(style_fc_mountain);
	styleFcs.push_back(style_fc_sky);
	//styleFcs.push_back(style_fc_tree);
	//styleFcs.push_back(style_fc_water);

	targetFcs.push_back(target_fc_mountain);
	targetFcs.push_back(target_fc_sky);
	//targetFcs.push_back(target_fc_tree);
	//targetFcs.push_back(target_fc_water);

	vector<pair<Mat, Mat>> responseGuides;

	for (size_t i = 0; i < styleFcs.size(); i++)
	{
		if (!ImageMatHasResponse(styleFcs[i]) || !ImageMatHasResponse(targetFcs[i]))
		{
			continue;
		}

		responseGuides.push_back(make_pair(styleFcs[i], targetFcs[i]));
	}

	return responseGuides;
}


Mat RunUST(Mat style, Mat target, const string& styLitPhotoCEPPath, const int USTShorterSide, const float UST_alpha, const bool useNewUST)
{
	const string style_PNG_path = styLitPhotoCEPPath + "guides/x_UST_input_style.png";
	const string target_PNG_path = styLitPhotoCEPPath + "guides/x_UST_input_target.png";
	const string result_PNG_path = useNewUST ? 
								   (styLitPhotoCEPPath + "guides/x_UST_input_target_stylized_x_UST_input_style.png") :
								   (styLitPhotoCEPPath + "guides/x_UST_input_target_x_UST_input_style.png");

	// Subsample input for UST 
	if (USTShorterSide <= 0)
	{
		ResizeImageMaintainAspectRatio(style,  400);
		ResizeImageMaintainAspectRatio(target, 400);
	}
	else
	{
		ResizeImageMaintainAspectRatio(style,  USTShorterSide);
		ResizeImageMaintainAspectRatio(target, USTShorterSide);
	}

	Imwrite(style_PNG_path,  style,  true);
	Imwrite(target_PNG_path, target, true);

	const string style_PNG_path_local = "../guides/x_UST_input_style.png";
	const string target_PNG_path_local = "../guides/x_UST_input_target.png";

	string command;
	if (useNewUST)
	{
		command = "cd " + styLitPhotoCEPPath + "/UST_new " + "&& " +
			"echo Running new UST ... && " +
			"python test_fwct_isn.py --style_size 0 --content_size 0 --alpha " + std::to_string(UST_alpha) + " " + 
			"--style=" + style_PNG_path_local + " --content=" + target_PNG_path_local + " --save_ext png --output_dir ../guides > log.txt 2>&1";
	}
	else
	{
		command = "cd " + styLitPhotoCEPPath + "/UST_old " + "&& " +
			"echo Running old UST ... && " +
			"python stylize.py --checkpoints models/relu5_1 models/relu4_1 models/relu3_1 models/relu2_1 models/relu1_1 --relu-targets relu5_1 relu4_1 relu3_1 relu2_1 relu1_1 --style-size 0 --content-size 0 --alpha " + std::to_string(UST_alpha) + " " +
			"--style-path " + style_PNG_path_local + " --content-path " + target_PNG_path_local + " --out-path ../guides > log.txt 2>&1";
	}
	

	system(command.c_str());

	cv::Mat UST_result = Imread(result_PNG_path, true);

	//std::remove(style_PNG_path.c_str());
	//std::remove(target_PNG_path.c_str());
	//std::remove(result_PNG_path.c_str());

	return UST_result;
}


Mat FakeRunningUST(const string& USTResultPath, const int USTShorterSide)
{
	cv::Mat UST_result = Imread(USTResultPath, true);

	// Fake: Subsample input for UST 
	if (USTShorterSide <= 0)
	{
		ResizeImageMaintainAspectRatio(UST_result, 400);
	}
	else
	{
		ResizeImageMaintainAspectRatio(UST_result, USTShorterSide);
	}

	return UST_result;
}


Mat1f SegmentImage(const Mat& mat, Mat& labels)
{
	return Mat1f();
	/*
	IplImage img(mat);

	vector<vector<int>> ilabels;
	ilabels.resize(img.height);
	for (int i = 0; i < img.height; i++) {
		ilabels[i].resize(img.width,-1);
	}
	int regionCount = MeanShift(mat, ilabels);
	{
		ostringstream oss;
		oss << "number of labels: " << regionCount;
		//DebugModalMsg(oss.str());
	}
	vector<int> color_0(regionCount, 0);
	vector<int> color_1(regionCount, 0);
	vector<int> color_2(regionCount, 0);

	vector<int> colorCount(regionCount, 0);

	for (int i = 0; i < img.height; i++)
	{
		for (int j = 0; j < img.width; j++)
		{
			int cl = ilabels[i][j];
			color_0[cl] += ((uchar *)(img.imageData + i*img.widthStep))[j*img.nChannels + 0];
			color_1[cl] += ((uchar *)(img.imageData + i*img.widthStep))[j*img.nChannels + 1];
			color_2[cl] += ((uchar *)(img.imageData + i*img.widthStep))[j*img.nChannels + 2];

			colorCount[cl]++;
		}
	}

	Mat1f segments = Mat1f(regionCount, 4);

	for (int region = 0; region < regionCount; region++)
	{
		segments.at<float>(region, 0) = (float)colorCount[region];
		segments.at<float>(region, 1) = (float)(color_0[region] / colorCount[region]);
		segments.at<float>(region, 2) = (float)(color_1[region] / colorCount[region]);
		segments.at<float>(region, 3) = (float)(color_2[region] / colorCount[region]);
	}

	// Draw AVG color
	for (int i = 0; i < img.height; i++)
	{
		for (int j = 0; j < img.width; j++)
		{
			int region = ilabels[i][j];

			((uchar *)(img.imageData + i*img.widthStep))[j*img.nChannels + 0] = (uchar)segments.at<float>(region, 1);
			((uchar *)(img.imageData + i*img.widthStep))[j*img.nChannels + 1] = (uchar)segments.at<float>(region, 2);
			((uchar *)(img.imageData + i*img.widthStep))[j*img.nChannels + 2] = (uchar)segments.at<float>(region, 3);
		}
	}


	labels = Mat(img.height, img.width, CV_32S);
	for (int i = 0; i < img.height; i++)
	{
		for (int j = 0; j < img.width; j++)
		{
			labels.at<int>(i, j) = ilabels[i][j];
		}
	}

	return segments;
	*/
}

bool ImageMatHasResponse(const Mat& mat)
{
	if (mat.type() != CV_8UC3)
	{
		cout << "ImageMatHasResponse works on CV_8UC3 Mats only " << endl;
		throw(mat);
	}

	for (int row = 0; row < mat.rows; row++)
	{
		for (int col = 0; col < mat.cols; col++)
		{
			if (mat.at<Vec3b>(row, col) != Vec3b(0, 0, 0)) {
				return true;
			}
		}
	}

	return false;
}

// TODO: Check styLitMaxMP against the available GPU memory
int SubsampleIfNecessary(cv::Mat& sourceStyleMat, vector<cv::Mat>& sources, vector<cv::Mat>& targets, const float styLitMaxMP)
{	
	const float pixelsMP = std::max(((float)sources[0].rows / 1000.0f)*((float)sources[0].cols / 1000.0f), 
		                            ((float)targets[0].rows / 1000.0f)*((float)targets[0].cols / 1000.0f));

	if (styLitMaxMP == 0.0f) 
	{
		return 1;
	}

	int coefficient = (int)ceilf(std::sqrt(ceil(pixelsMP / styLitMaxMP)));
	
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