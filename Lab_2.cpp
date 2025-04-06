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
//Lab_2
void split_channels() {
	Mat img = imread("Images/Lena_24bits.bmp");
    if (img.empty()) return;
    Mat_<uchar> blue(img.rows, img.cols);
    Mat_<uchar> green(img.rows, img.cols);
    Mat_<uchar> red(img.rows, img.cols);
    for (int i = 0; i < img.rows; i++) {
        for (int j = 0; j < img.cols; j++) {
            Vec3b pixel = img.at<Vec3b>(i, j);
            blue(i, j) = pixel[0];
            green(i, j) = pixel[1];
            red(i, j) = pixel[2];
        }
    }
    imshow("Blue Channel", blue);
    imshow("Green Channel", green);
    imshow("Red Channel", red);
    waitKey(0);
}
void convert_to_grayscale() {
	Mat img = imread("Images/Lena_24bits.bmp");
	if (img.empty()) return;
	Mat_<uchar> grayImg(img.rows, img.cols);
	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			Vec3b pixel = img.at<Vec3b>(i, j);
			uchar B = pixel[0];
			uchar G = pixel[1];
			uchar R = pixel[2];
			int average = (R + G + B) / 3;
			grayImg(i, j) = static_cast<uchar>(average);
		}
	}
	imshow("Grayscale", grayImg);
	waitKey(0);
}
void convert_grayscale_to_BW() {
	Mat_<uchar> img = imread("Images/Lena_24bits.bmp", IMREAD_GRAYSCALE);
	if (img.empty()) return;
	int thresholdValue;
	std::cout << "Enter threshold : ";
	std::cin >> thresholdValue;
	if (thresholdValue < 0) thresholdValue = 0;
	if (thresholdValue > 255) thresholdValue = 255;

	Mat_<uchar> bwImg(img.rows, img.cols);
	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			uchar grayVal = img(i, j);
			if (grayVal >= thresholdValue) {
				bwImg(i, j) = 255; 
			}
			else {
				bwImg(i, j) = 0;  
			}
		}
	}
	imshow("Original" , img);
	imshow("Binary Image ", bwImg);
	waitKey(0);
}
Mat_ <Vec3b>  computeHSV(Mat_<Vec3b>& img) {
	Mat_<Vec3b> hsvImg (img.rows, img.cols);
	
	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			Vec3b pixel = img.at<Vec3b>(i, j);
			float b = pixel[0] / 255.0f;
			float g = pixel[1] / 255.0f;
			float r = pixel[2] / 255.0f;

			float max_val = max(r, max(g, b));
			float min_val = min(r, min(g, b));
			float C = max_val - min_val;
			float V = max_val;

			float Sat =0;
			if (V != 0)
				Sat = C / V;

			float Hue = 0;
			if (C != 0) {
				if (max_val == r) {
					Hue = 60 * (g - b) / C;
				}
				else if (max_val == g) {
					Hue = 120 + 60 * (b - r) / C;
				}
				else { 
					Hue = 240 + 60 * (r - g) / C;
				}
			}
			else {
				Hue = 0;
			}
			if (Hue < 0)
				Hue += 360;
			hsvImg(i, j) = Vec3b(Hue * 180 / 360, Sat * 255, V * 255);
		}
	}
	return hsvImg;
}
bool isInside(Mat img, int i, int j) {
	return (i >= 0 && i < img.rows && j >= 0 && j < img.cols);
}


void main() {
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

}