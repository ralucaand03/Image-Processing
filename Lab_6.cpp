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
//Lab6
vector<Point> borderTrace(const Mat_<uchar>& img)
{
	int di[8] = { 0, -1, -1, -1, 0,  1,  1, 1 };
	int dj[8] = { 1,  1,  0, -1, -1, -1, 0, 1 };
	vector<Point> border;
	int start_i=-1, start_j=-1;
	for (int i = 0; i < img.rows && start_i < 0; i++)
	{
		for (int j = 0; j < img.cols && start_j < 0; j++)
		{
			if (img(i, j) == 0)
			{
				start_i = i;
				start_j = j;
			}
		}
	}

	if (start_i < 0)
		return border;

	Point P0(start_j, start_i);

	border.push_back(P0);
	int dir = 7;

	bool ok = false;
	int startDir;

	Point current = P0;
	Point P1;

	if (dir % 2 == 0) 
		startDir = (dir + 7) % 8;
	else 
		startDir = (dir + 6) % 8;

	for (int k = 0; k < 8; k++)
	{
		int checkDir = (startDir + k) % 8;
		int ni = start_i + di[checkDir];
		int nj = start_j + dj[checkDir];
		if (ni >= 0 && ni < img.rows && nj >= 0 && nj < img.cols)
		{
			if (img(ni, nj) == 0) 
			{
				P1 = Point(nj, ni);
				dir = checkDir;
				ok = true;
				break;
			}
		}
	}
	if (!ok) return border;

	border.push_back(P1);

	Point prev = P0;
	current = P1;

	while (1) {
		if (dir % 2 == 0)
			startDir = (dir + 7) % 8;
		else
			startDir = (dir + 6) % 8;

		ok = false;

		for (int k = 0; k < 8; k++)
		{
			int checkDir = (startDir + k) % 8;
			int ni = current.y + di[checkDir];
			int nj = current.x + dj[checkDir];
			if (ni >= 0 && ni < img.rows && nj >= 0 && nj < img.cols)
			{
				if (img(ni, nj) == 0)
				{
					prev = current;
					current = Point(nj, ni);
					
					dir = checkDir;
					ok = true;
					break;
				}
			}
		}
		if (current== P0)
			break;
		border.push_back(current);
		if (!ok)  
			break;
	}
	return border;

}
vector<int> buildChainCode(vector<Point>& border)
{
	vector<int> chain;
	for (int i = 0; i + 1 < border.size(); i++)
	{
		int dr = border[i + 1].y - border[i].y;
		int dc = border[i + 1].x - border[i].x;
		int d = -1;
		if (dr == 0 && dc == 1) d = 0;
		else if (dr == -1 && dc == 1) d = 1;
		else if (dr == -1 && dc == 0) d = 2;
		else if (dr == -1 && dc == -1) d = 3;
		else if (dr == 0 && dc == -1) d = 4;
		else if (dr == 1 && dc == -1) d = 5;
		else if (dr == 1 && dc == 0) d = 6;
		else if (dr == 1 && dc == 1) d = 7;
		chain.push_back(d);
	}
	return chain;
}
void printVector(const vector<int>& v)
{
	for (int val : v)
		cout << val << " ";
	cout << "\n";
}
vector<int> buildDerivativeCode( vector<int>& chain)
{
	vector<int> deriv;
	for (int i = 0; i + 1 < chain.size(); i++)
	{
		int diff = (chain[i + 1] - chain[i] + 8) % 8;
		deriv.push_back(diff);
	}
	return deriv;
}
void readChainCode(const String& path, Point& start, vector<int>& chain) {
	ifstream fin(path);
	fin >> start.y >> start.x;

	int number;
	fin >> number;

	chain.clear();

	int code;
	while (fin >> code) {
		chain.push_back(code);
	}
	fin.close();
}
void reconstructs(Point start, vector<int> chain) {
	Mat_<uchar> img = imread("Images/gray_background.bmp", IMREAD_GRAYSCALE);
	img(start.y, start.x) = 0;
	int di[8] = { 0, -1, -1, -1, 0,  1,  1, 1 };
	int dj[8] = { 1,  1,  0, -1, -1, -1, 0, 1 };
	Point current = start;
	for (int c : chain)
	{
		current.y += di[c];
		current.x += dj[c];
		img(current.y, current.x) = 0;

	}
	imshow("Reconstruction", img);
	waitKey(0);

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
}