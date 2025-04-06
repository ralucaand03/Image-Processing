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
//Lab_1
void negative_image() {
	Mat_<uchar> img = imread("Images/cameraman.bmp", IMREAD_GRAYSCALE); // if u don t put IMREAD GRAYSCALE it will be 3 times wide
	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			img(i, j) = 255 - img(i, j);
		}
	}
	imshow("negative image", img);
	waitKey(0);
}
void additive_factor(int factor) {
	Mat_<uchar> img = imread("Images/cameraman.bmp", IMREAD_GRAYSCALE);
	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			int t = img(i, j) + factor;
			if (t > 255) img(i, j) = 255;
			else if (t < 0) img(i, j) = 0;
			else  img(i, j) = t;
		}
	}
	imshow("Additive Image", img);
	waitKey(0);
}
void multiplicative_factor(int factor) {
	Mat_<uchar> img = imread("Images/cameraman.bmp", IMREAD_GRAYSCALE);
	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			int t = img(i, j) * factor;
			if (t > 255) img(i, j) = 255;
			else if (t < 0) img(i, j) = 0;
			else  img(i, j) = t;
		}
	}
	imshow("Multiplicative Image", img);
	waitKey(0);
}
void create_img() {
	Mat img(256, 256, CV_8UC3);

	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			if (i < 128 && j < 128) img.at< Vec3b >(i, j) = Vec3b(255, 255, 255);
			else if (i >= 128 && j < 128) img.at< Vec3b >(i, j) = Vec3b(0, 0, 255); //red
			else if (i < 128 && j >= 128) img.at< Vec3b >(i, j) = Vec3b(0, 255, 0); //blue
			else  img.at<Vec3b>(i, j) = Vec3b(0, 255, 255); //yellow
		}
	}
	imshow("Created Image", img);
	waitKey(0);
}
void create3x3float() {
	float vals[9] = { 1.0f, 2.0f, 3.0f,
					  0.0f, 1.0f, 4.0f,
					  5.0f, 6.0f, 0.0f };
	Mat M(3, 3, CV_32FC1, vals);
	std::cout << "Matrix M:" << std::endl << M << std::endl;

	double det = determinant(M);
	std::cout << "Determinant of M: " << det << std::endl;

	if (det != 0) {
		Mat M_inv;
		invert(M, M_inv);
		std::cout << "Inverse of M:" << std::endl << M_inv << std::endl;
	}
	else {
		std::cout << "Determinant is zero => Matrix cannot be inverted." << std::endl;
	}

}
void symethric_img() {
	Mat_<uchar> img = imread("Images/cameraman.bmp", IMREAD_GRAYSCALE);
	if (img.empty()) {
		return;
	}
	Mat_<uchar> img2(img.rows, 2 * img.cols);
	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			img2(i, j) = img(i, j);
			img2(i, img2.cols - 1 - j) = img(i, j);
		}
	}
	imshow("Sym Image", img2);
	waitKey(0);
}


void main() {
	    
	////------------------------Lab_1
	// negative_image();
	// additive_factor(15);
	// multiplicative_factor(15);
	// create_img();
	// create3x3float();
	// symethric_img();


}