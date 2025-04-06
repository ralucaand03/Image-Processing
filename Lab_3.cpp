#include "stdafx.h"
#include "common.h"
#include <opencv2/core/utils/logger.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <cmath>
#include <limits>
#include <stack>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <vector>
#include <fstream>
using namespace std;
using namespace cv;
wchar_t* projectPath;
//Lab3
void showHistogram(const string& name, vector<int> hist, const int hist_cols,const int hist_height) {
	Mat imgHist(hist_height, hist_cols, CV_8UC3, CV_RGB(255, 255, 255));
	// constructs a white image
	//computes histogram maximum
	int max_hist = 0;
	for (int i = 0; i < hist_cols; i++)
		if (hist[i] > max_hist)
			max_hist = hist[i];
	double scale = 1.0;
	scale = (double)hist_height / max_hist;
	int baseline = hist_height - 1;
	for (int x = 0; x < hist_cols; x++) {
		Point p1 = Point(x, baseline);
		Point p2 = Point(x, baseline - cvRound(hist[x] * scale));
		line(imgHist, p1, p2, CV_RGB(255, 0, 255)); // histogram bins
		// colored in magenta
	}
	imshow(name, imgHist);
}
vector<int> calc_hist(Mat_ < uchar> img) {
	vector<int> hist(256, 0);
	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			hist[img(i, j)]++;
		}
	}
	return hist;
}
vector<float> compute_pdf(Mat_<uchar> img) { // PDF - Probability Density Function
	vector<float> pdf(256, 0.0f);
	vector<int> hist = calc_hist(img);
	int total = img.rows * img.cols;
	for (int i = 0; i < 256; i++) {
		pdf[i] = (float)hist[i] / total;
	}
	return pdf;
}
vector<int> hist_custom_bins(Mat_<uchar> img, int m) {
	vector<int> hist(m, 0);
	int bin_size = 256 / m;
	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			int g = img(i, j); //gray intensiry lvl DIV size
			hist[g*m /256]++;
		}
	}
	return hist;
}
Mat_<uchar> multilevel_thresholding(Mat_<uchar> img) {
	vector<int> hist = calc_hist(img);
	vector<float> pdf = compute_pdf(img);

	int WH = 5;
	float TH = 0.0003f;

	Mat_<uchar> result = img.clone();
	vector<int> maxima;
	maxima.push_back(0);

	for (int k = WH; k < 256 - WH; k++) {
		float sum = 0;
		for (int i = -WH; i <= WH; i++) {
			sum += pdf[k + i];
		}
		float avg = sum / (2 * WH + 1);
		bool isLocalMax = true;
		for (int i = -WH; i <= WH; i++) {
			if (pdf[k] < pdf[k + i]) {
				isLocalMax = false;
				break;
			}
		}
		if (pdf[k] > avg + TH && isLocalMax) {
			maxima.push_back(k);
			k += WH;
		}
	}
    
	maxima.push_back(255); 

	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			int old_pixel = img(i, j);
			int new_pixel = findClosestHistogramMaxima(maxima, old_pixel);
			// 
            //int minDist = abs(pixel - maxima[0]);
            // int closest = maxima[0];
			// for (int m = 1; m < maxima.size(); m++) {
			// 	int dist = abs(pixel - maxima[m]);
			// 	if (dist < minDist) {
			// 		closest = maxima[m];
			// 		minDist = dist;
			// 	}
			// }
			result(i, j) = new_pixel; // new_pixel == closest
		}
	}
	return result;
}
int findClosestHistogramMaxima(vector<int> maxima, int pixel) { 
	int minDist = abs(pixel - maxima[0]);
	int closest = maxima[0]; 
	for (int m = 0; m < maxima.size(); m++) {
		int currDist = abs(pixel - maxima[m]);
		if (minDist > currDist) {
			minDist = currDist;
			closest = maxima[m];
		}
	}
	return closest;
}
void floydSteinbergDithering(Mat_<uchar> img) {
	vector<int> hist = calc_hist(img);
	vector<float> pdf = compute_pdf(img);

	int WH = 5;
	float TH = 0.0003;

	vector<int> maxima; 
	maxima.push_back(0);

	for (int k = WH; k < 256 - WH; k++) {
		float sum = 0;
		for (int i = -WH; i <= WH; i++) {
			sum += pdf[k + i];
		}
		float avg = sum / (2 * WH + 1);
		bool isLocalMax = true;
		for (int i = -WH; i <= WH; i++) {
			if (pdf[k] < pdf[k + i]) {
				isLocalMax = false;
				break;
			}
		}
		if (pdf[k] > avg + TH && isLocalMax) {
			maxima.push_back(k);
			k += WH;
		}
	}
   
	maxima.push_back(255);

    //So far same, from here it is different
	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			int old_pixel = img(i, j);
			int new_pixel = findClosestHistogramMaxima(maxima, old_pixel);
			img(i, j) = new_pixel;
            
            //from here diff - just the errors

			int error = old_pixel - new_pixel;

			if (isInside(img, i, j + 1))
				img(i, j + 1) +=  (7 * error) / 16;
			
			if (isInside(img, i + 1, j - 1))
				img(i + 1, j - 1) +=  (3 * error) / 16;

			if (isInside(img, i + 1, j ))
				img(i + 1, j - 1) += (5 * error) / 16;

			if (isInside(img, i + 1, j + 1))
				img(i + 1, j + 1) += error / 16;
		}
	}

	imshow("Image created with MultiThreading", img);
	waitKey(0);
}
bool isInside(Mat img, int i, int j) {
	return (i >= 0 && i < img.rows && j >= 0 && j < img.cols);
}

void main() {
	////------------------------Lab_6
	// Mat_<uchar> img = imread("Images/triangle_up.bmp", IMREAD_GRAYSCALE);
	// if (img.empty())
	// {
	// 	cerr << "Could not open or find the image!\n";
	// 	return;
	// }
	// imshow("image", img);
	// waitKey(0);
	// vector<Point> contour = borderTrace(img);
	// Mat_<Vec3b> colorImg;
	// cvtColor(img, colorImg, COLOR_GRAY2BGR);
	// for (Point pt : contour)
	// 	colorImg(pt.y, pt.x) = Vec3b(0, 0, 255);
	// imshow("Traced Border", colorImg);
	// waitKey(0);
	// vector<int> chain = buildChainCode(contour);
	// cout<<"Chain Code: "<<"\n";
	// printVector(chain);
	// vector<int> der = buildDerivativeCode(chain);
	// cout << "Derivative Code: " << "\n";
	// printVector(der);
	// Point start;
	// vector<int> chain_rec;
	// readChainCode("Images/reconstruct.txt", start, chain_rec);
	// reconstructs(start, chain_rec);

	////------------------------Lab_5
	//Mat_<uchar> img = imread("Images/shapes.bmp", IMREAD_GRAYSCALE);
	//Mat_<int> labels = Mat::zeros(img.size(), CV_32SC1);
	//imshow("img", img);
	//waitKey(0);
	////
	//two_pass(img, labels,8);
	//bfs(img, labels, 8);
	//dfs(img, labels, 8);
	//bfs_visualize(img, labels, 8);
	
	////------------------------Lab_4
	//loadAndSetupImage("Images/oval_obl.bmp");
	//waitKey(0);

	////------------------------Lab_3
	// Mat_<uchar> img = imread("Images/cameraman.bmp", IMREAD_GRAYSCALE);
	// if (img.empty()) return;
	// Mat_<Vec3b> colorimg = imread("Images/Lena_24bits.bmp");
	// imshow("Original", colorimg);
	// vector<int> hist = calc_hist(img);
	// showHistogram("Histogram", hist, 256, 400);
	// waitKey(0);
	// int m = 128; 
	// vector<int> custom_hist = hist_custom_bins(img, m);
	// showHistogram("Custom Bins", hist, 100, 300);
	// waitKey(0);
	// Mat simplified = multilevel_thresholding(img);
	// imshow("Multilevel Thresholding", simplified);
	// waitKey(0);
	// Mat hue = floydSteinbergDithering(colorimg);
	// imshow("Multilevel Thresholding", hue);
	// waitKey(0);

    ////------------------------Lab_2
	// split_channels();
	// convert_to_grayscale();
	// convert_grayscale_to_BW();
	// Mat_<Vec3b> img = imread("Images/Lena_24bits.bmp");
	// imshow("Original", img);
	// waitKey(0);
	// Mat_<Vec3b> hsvimg = computeHSV(img);
	// imshow("HSV", hsvimg);
	// waitKey(0);
	// Mat_<Vec3b> rgbimg(hsvimg.rows, hsvimg.cols);
	// cvtColor(hsvimg, rgbimg, COLOR_HSV2BGR);
	// imshow("RGB", rgbimg);
	// waitKey(0);
	    
	////------------------------Lab_1
	// negative_image();
	// additive_factor(15);
	// multiplicative_factor(15);
	// create_img();
	// create3x3float();
	// symethric_img();


}