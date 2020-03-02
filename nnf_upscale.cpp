#include "cxxopts.hpp"			//cxxopts commandline parser


bool isOutOfImage(const Mat& image, const int row, const int col)
{
	return row < 0 ||
		col < 0 ||
		row >= image.rows ||
		col >= image.cols;
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
			for (int row_offset_in_patch = -1 * halfPatchsize; row_offset_in_patch <= halfPatchsize; row_offset_in_patch++)
			{
				for (int col_offset_in_patch = -1 * halfPatchsize; col_offset_in_patch <= halfPatchsize; col_offset_in_patch++)
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


Mat2i upsampleNNF(const Mat2i& NNFSmall, const int coeficient)
{
	Mat2i NNFBig = Mat2i(NNFSmall.rows * coeficient, NNFSmall.cols * coeficient);

#pragma omp parallel for
	for (int row = 0; row < NNFBig.rows; row++)
	{
		for (int col = 0; col < NNFBig.cols; col++)
		{
			Vec2i value = NNFSmall.at<Vec2i>(row / coeficient, col / coeficient);

			NNFBig.at<Vec2i>(row, col) = Vec2i(value[0] * coeficient + (row % coeficient),
				value[1] * coeficient + (col % coeficient));
		}
	}

	return NNFBig;
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
	if (output.rows != originalTargetRows || output.cols != originalTargetCols)
	{
		cv::resize(output, output, Size(originalTargetCols, originalTargetRows), cv::InterpolationFlags::INTER_CUBIC);
	}

	return output;
}



int main() 
{


}