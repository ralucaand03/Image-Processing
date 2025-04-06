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
//Lab4
vector<Point> pixels(Mat_<uchar>& img, int label_val) {
	vector<Point> pixels;
	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			int pixel = img(i, j);
			if (pixel == label_val) {
				pixels.push_back(Point(j, i));
			}
		}
	}
	return pixels;
}

int Area(vector<Point>& pixels) {
	return pixels.size();
}

Point CenterMass(vector<Point>& pixels) {
	long long sum_of_i = 0;
	long long sum_of_j = 0;

	for (int i = 0; i < pixels.size(); i++) {
		sum_of_i += pixels[i].y;
		sum_of_j += pixels[i].x;
	}

	double i_mass = (sum_of_i * 1.0) / Area(pixels);
	double j_mass = (sum_of_j * 1.0) / Area(pixels);
    return Point(j_mass,i_mass);
}
double Elongation(vector<Point>& pixels) {
    Point center = CenterMass(pixels);
    double center_y = center.y; // i_mass = row (y)
    double center_x = center.x; // j_mass = col (x)

    double sum_xy = 0.0, sum_y = 0.0, sum_x = 0.0;

    for (int i = 0; i < pixels.size(); i++) {
        double dy = pixels[i].y - center_y; // row diff
        double dx = pixels[i].x - center_x; // col diff

        sum_y += dy * dy;
        sum_x += dx * dx;
        sum_xy += dy * dx;
    }

    double angle = 0.5 * atan2(2.0 * sum_xy, sum_x - sum_y);
    return angle;
}
int Perimeter(Mat_<uchar>& img, int label_val) {
	int perimeter  = 0;
	for (int y = 0; y < img.rows; y++) {
		for (int x = 0; x < img.cols; x++) {
			if (img(y, x) == label_val) {
				bool isOnContour = false;

				for (int dy = -1; dy <= 1 && !isOnContour; dy++) {
                    for (int dx = -1; dx <= 1 && !isOnContour; dx++) {
                        if (dx != 0 || dy == 0) {
                            int ny = y + dy;
                            int nx = x + dx;
                            if (ny < 0 || ny >= img.rows || nx < 0 || nx >= img.cols || img(ny, nx) != label_val) {
                                isOnContour = true;
                            }
                        }
                    }
                }
				if (isOnContour)
                perimeter ++;
			}
		}
	}
	return perimeter ;
}
void Projection(Mat_<uchar>& img, int label_val) {
	vector<int> proj_h(img.cols, 0); // vertical bars
	vector<int> proj_v(img.rows, 0); // horizontal bars

    for (int y = 0; y < img.rows; y++) {
        for (int x = 0; x < img.cols; x++) {
            if (img(y, x) == label_val) {
                proj_h[x]++;
                proj_v[y]++;
            }
        }
    }

    // Create white canvases
    Mat projection_h(img.rows, img.cols, CV_8UC1, Scalar(255));
    Mat projection_v(img.rows, img.cols, CV_8UC1, Scalar(255));

    // Draw horizontal projection (as vertical bars from bottom up)
    for (int x = 0; x < img.cols; x++) {
        int barHeight = proj_h[x];
        for (int y = img.rows - 1; y >= img.rows - barHeight && y >= 0; y--) {
            projection_h.at<uchar>(y, x) = 0;
        }
    }

    // Draw vertical projection (as horizontal bars from left to right)
    for (int y = 0; y < img.rows; y++) {
        int barWidth = proj_v[y];
        for (int x = 0; x < barWidth && x < img.cols; x++) {
            projection_v.at<uchar>(y, x) = 0;
        }
    }

    imshow("Horizontal Projection", projection_h);
    imshow("Vertical Projection", projection_v);
}
void util(Mat_<uchar>& img, int label_val) {
	vector<Point> objPixels = pixels(img, label_val);

	Mat_<Vec3b> clone_img;
	cvtColor(img, clone_img, COLOR_GRAY2BGR);
	int area = Area(objPixels);

    Point p = CenterMass(objPixels);
    double i_mass, j_mass;
    i_mass=p.x;
    j_mass=p.y;
	double axis = Elongation(objPixels );
	int perimeter = Perimeter(img, label_val);
	//Out
	cout << "Area: " << area << endl;
	cout << "Perimeter: " << perimeter << endl;
	cout << "Center_of_mass: (" << j_mass << ", " << i_mass << ")" << endl;
	cout << "Axis of elongation: " << axis << endl;
	
	//draw
	//---perimeter
	Mat_<Vec3b> pimg(img.size(), Vec3b(255, 255, 255));
	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			if (img(i, j) == label_val) {
				bool border = false;

				for (int window_i = -1; window_i <= 1; window_i++) {
					for (int window_j = -1; window_j <= 1; window_j++) {
						if (!(window_i == 0 && window_j == 0)) {
							int neighbour_i = i + window_i;
							int neighbour_j = j + window_j;

							if (neighbour_i < 0 || neighbour_i >= img.rows ||
								neighbour_j < 0 || neighbour_j >= img.cols) {
								border = true;
							}
							else {
								if (img(neighbour_i, neighbour_j) != label_val) {
									border = true;
								}
							}
						}
						if (border)
							break;
					}
					if (border)
						break;
				}

				if (border)
					pimg(i, j) = Vec3b(0, 0, 255);
				else
					pimg(i, j) = Vec3b(100, 100, 100);
			}
		}
	}
	imshow("Perimeter", pimg);

	//---center of mass
	Mat_<Vec3b> center_of_mass_img = clone_img.clone();
	Point cMassP((int)round(j_mass), (int)round(i_mass));
	line(center_of_mass_img, Point(cMassP.x - 10, cMassP.y), Point(cMassP.x + 10, cMassP.y), Scalar(0, 0, 255), 2);
	line(center_of_mass_img, Point(cMassP.x, cMassP.y - 10), Point(cMassP.x, cMassP.y + 10), Scalar(0, 0, 255), 2);
	imshow("Center of mass", center_of_mass_img);

	//---axis of enlg
	Mat_<Vec3b> axisimg = clone_img.clone();
	double length = 75.0;
	Point a1((int)round(j_mass - length * cos(axis)), (int)round(i_mass - length * sin(axis)));
	Point a2((int)round(j_mass + length * cos(axis)), (int)round(i_mass + length * sin(axis)));
	line(axisimg, a1, a2, Scalar(255, 0, 0), 2);
	imshow("Axis of elongation", axisimg);

	//---projectin
	Projection(img, label_val);

}
static void onMouse(int event, int x, int y, int flags, void* param) {
	if (event != EVENT_LBUTTONDOWN)
		return;
	Mat_<uchar> img = *(Mat_<uchar>*)param;
	int label_val = img(y, x);
	Vec3b color = img(x, y);
	Mat_<uchar> bw(img.size());
	util(img, label_val);

}
void loadAndSetupImage(const string& imagePath) {

	Mat_<uchar> img = imread(imagePath, IMREAD_GRAYSCALE);
	namedWindow("Image", WINDOW_AUTOSIZE);
	imshow("Image", img);
	setMouseCallback("Image", onMouse, &img);

	waitKey(0);

}

void main() {
	 
	////------------------------Lab_4
	//loadAndSetupImage("Images/oval_obl.bmp");
	//waitKey(0);
 }