// OpenCVApplication.cpp : Defines the entry point for the console application.
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

void geometrical_features(Mat_<uchar> img) {
	Mat_<Vec3b> modif_img(img.rows, img.cols);
	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			modif_img(i, j)[0] = 255;
			modif_img(i, j)[1] = 255;
			modif_img(i, j)[2] = 255;
		}
	}
	imshow("Original Image", img);
	waitKey(0);

	int area = 0;
	int center_row = 0;
	int center_col = 0;
	int perimeter = 0;
	bool edge = false;

	int* rows = new int[img.rows]();
	int* cols = new int[img.cols]();

	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			if (img(i, j) == 0) {
				//compute the area of the object
				area++;
				modif_img(i, j)[0] = 0;
				modif_img(i, j)[1] = 0;
				modif_img(i, j)[2] = 0;

				//compute the center of mass
				center_row += i;
				center_col += j;


				if (isInside(img, i, j - 1) && (img(i, j - 1) == 255)) {
					perimeter++; 
					edge = true;
				}
				else {
					if (isInside(img, i, j + 1) && (img(i - 1, j - 1) == 255)) {
						perimeter++;
						edge = true;
					}
					else {
						if (isInside(img, i + 1, j) && (img(i + 1, j) == 255)) {
							perimeter++;
							edge = true;
						}
						else
							if (isInside(img, i + 1, j) && (img(i, j + 1) == 255)){
								perimeter++;
								edge = true;
							}
					}
				}
				if (edge) {
					modif_img(i, j)[0] = 0;
					modif_img(i, j)[1] = 0;
					modif_img(i, j)[2] = 255;
				}
				edge = false;

				rows[i]++;
				cols[j]++;
			}
		}
	}

	center_row /= area;
	center_col /= area;

	for (int i = center_col - 10; i <= center_col + 10; i++) {
		modif_img(center_row, i)[0] = 255; 
		modif_img(center_row, i)[1] = 255;
		modif_img(center_row, i)[2] = 255;
	}

	for (int i = center_row - 10; i <= center_row + 10; i++) {
		modif_img(i, center_col)[0] = 255;
		modif_img(i, center_col)[1] = 255;
		modif_img(i, center_col)[2] = 255;
	}

	int num = 0; 
	int den = 0, den1 = 0, den2 = 0;
	int dif_col = 0, dif_row = 0;

	int c_max = 0, c_min = img.cols;
	int r_max = 0, r_min = img.rows;
	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			if (img(i, j) == 0) {
				dif_col = j - center_col;
				dif_row = i - center_row;
				num += dif_row * dif_col;
				den += dif_col * dif_col - dif_row * dif_row;

				if (i < r_min)
					r_min = i;
				if (i > r_max)
					r_max = i;
				if (j < c_min)
					c_min = j;
				if (j > c_max)
					c_max = j;
			}
		}
	}

	num = num * 2;

	double ang = atan2(num, den) * 0.5;
	cout << ang;
	int p1_x = center_col + 100 * cos(ang);
	int p1_y = center_row + 100 * sin(ang);

	int p2_x = center_col - 100 * cos(ang);
	int p2_y = center_row - 100 * sin(ang);

	Point p1(p1_x, p1_y);
	Point p2(p2_x, p2_y);

	line(modif_img, p1, p2, (255, 255, 255));

	float circularity = (4 * PI * area) / (perimeter * perimeter);
	cout << "Circularity " << circularity << endl;

	Point b1(c_min, r_min);
	Point b2(c_min, r_max);
	Point b3(c_max, r_max);
	Point b4(c_max, r_min);

	line(modif_img, b1, b2, (255, 0, 0));
	line(modif_img, b2, b3, (255, 0, 0));
	line(modif_img, b3, b4, (255, 0, 0));
	line(modif_img, b4, b1, (255, 0, 0));

	imshow("Modif Img", modif_img);
	
	float aspect_ration = (c_max - c_min + 1.f) / (r_max - r_min + 1);
	cout << "Aspect Ratio " << aspect_ration << endl;


	Mat_<Vec3b> projection(img.size());

	projection.setTo(255);

	for (int i = 0; i < projection.rows; i++) {
		int cur_row = rows[i];
		for (int k = 0; k < cur_row; k++) {
			int j = projection.cols - k - 1;

			projection(i, j)[0] = 0;
			projection(i, j)[1] = 0;
			projection(i, j)[2] = 255;
		}
	}

	for (int j = 0; j < projection.cols; j++) {
		int cur_col = cols[j];
		for (int k = 0; k < cur_col; k++) {
			int i = projection.rows - k - 1;
			projection(i, j)[0] = 0;
			projection(i, j)[1] = 255;
			projection(i, j)[2] = 0;
		}
	}

	imshow("Projection", projection);

	waitKey(0);

}
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
//Lab5
Mat_<Vec3b> generate_color(Mat_<int>& labels)
{
	srand(time(0));
	Mat_<Vec3b> colorImg = Mat::zeros(labels.size(), CV_8UC3);
	vector<Vec3b> labelColorVec;
	srand(0);
	labelColorVec.push_back({ 255, 255, 255 });
	for (int i = 1; i < 1000; i++) {
		uchar r = rand() % 256;
		uchar g = rand() % 256;
		uchar b = rand() % 256;
		labelColorVec.push_back(Vec3b(b, g, r));
	}
	for (int i = 0; i < labels.rows; i++) {
		for (int j = 0; j < labels.cols; j++) {
			int label = labels(i, j);

			colorImg(i, j) = labelColorVec[label];  
			
		}
	}
	return colorImg;
} 
void bfs(Mat_<uchar>& img, Mat_<int>& labels, int u)
{
	int di8[8] = { -1,-1,-1,0,0,1,1,1 };
	int dj8[8] = { -1,0,1,-1,1,-1,0,1 };
	int di4[4] = { -1,0,1,0 };
	int dj4[4] = { 0,-1,0,1 };
	int label = 0;
	for (int i = 0; i < img.rows; i++){
		for (int j = 0; j < img.cols; j++){
			if (img(i, j) == 0 && labels(i, j) == 0){
				label++;
				queue< pair<int, int> > Q;
				labels(i, j) = label;
				Q.push(make_pair(i, j));

				while (!Q.empty())
				{
					pair<int, int> p = Q.front();
					Q.pop();

					for (int t = 0; t < u; t++) {
						int ni, nj;
						if (u == 8) {
							ni = p.first + di8[t];
							nj = p.second + dj8[t];
						}
						else {
							ni = p.first + di4[t];
							nj = p.second + dj4[t];

						}
						if (ni >= 0 && ni < img.rows && nj >= 0 && nj < img.cols &&
							img(ni, nj) == 0 && labels(ni, nj) == 0) {
							labels(ni, nj) = label;
							Q.push(pair<int, int>(ni, nj));
						}

					}
				}
			}
		}
	}
	Mat_<Vec3b> color_result = generate_color(labels);
	imshow("BFS", color_result);
	waitKey(0);

} 
void two_pass(Mat_<uchar>& img, Mat_<int>& labels,int u) {
	int di4[4] = { -1, 0, 1, 0 };
	int dj4[4] = { 0, -1, 0, 1 };
	int di8[8] = { -1,-1,-1,0,0,1,1,1 };
	int dj8[8] = { -1,0,1,-1,1,-1,0,1 };
	int label = 0;
	vector<vector<int>> edges(1000);

	//First pass
	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			if (img(i, j) == 0 && labels(i, j) == 0) { 
				vector<int> L;
				for (int t = 0; t < u; t++) {
					int ni = (u == 8) ? i + di8[t] : i + di4[t];
					int nj = (u == 8) ? j + dj8[t] : j + dj4[t];
					if (ni >= 0 && ni < img.rows && nj >= 0 && nj < img.cols && labels(ni, nj) > 0) {
						L.push_back(labels(ni, nj));
					}
				}
				if (L.size() == 0) {
					label++;
					labels(i, j) = label;
				}
				else {
					int x = L[0];   
					for (int i = 1; i < L.size(); i++) {
						if (L[i] < x) {
							x = L[i];
						}
					}
					labels(i, j) = x;
					for (int y = 0; y < L.size(); y++) {
						if (L[y] != x) {
							if (x >= edges.size()) edges.resize(x + 1);
							if (L[y] >= edges.size()) edges.resize(L[y] + 1);
							edges[x].push_back(L[y]);
							edges[L[y]].push_back(x);
						}
					}
				}
			}
		}
	}

	Mat_<Vec3b> color_result = generate_color(labels);
	imshow("First Pass", color_result);
	waitKey(0);

	// Second pass
	vector<int> newlabels(label + 1, 0);
	int newlabel = 0;

	for (int i = 1; i <= label; i++) {
		if (newlabels[i] == 0) {
			newlabel++;
			queue<int> Q;
			newlabels[i] = newlabel;
			Q.push(i);

			while (!Q.empty()) {
				int x = Q.front();
				Q.pop();

				for (int y : edges[x]) {
					if (newlabels[y] == 0) {
						newlabels[y] = newlabel;
						Q.push(y);
					}
				}
			}
		}
	}

	// Update the labels matrix
	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			if (labels(i, j) > 0) {
				labels(i, j) = newlabels[labels(i, j)];
			}
		}
	}

	Mat_<Vec3b> final_result = generate_color(labels);
	imshow("Final Result", final_result);
	waitKey(0);
}
void bfs_visualize(Mat_<uchar>& img, Mat_<int>& labels, int u)
{
	int di8[8] = { -1,-1,-1,0,0,1,1,1 };
	int dj8[8] = { -1,0,1,-1,1,-1,0,1 };
	int di4[4] = { -1,0,1,0 };
	int dj4[4] = { 0,-1,0,1 };
	int label = 0;

	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			if (img(i, j) == 0 && labels(i, j) == 0) { 
				label++;
				queue<pair<int, int>> Q;
				labels(i, j) = label;
				Q.push(make_pair(i, j));

				while (!Q.empty()) {
					pair<int, int> p = Q.front();
					Q.pop();

					
					int N = (u == 8) ? 8 : 4;
					for (int t = 0; t < N; t++) {
						int ni = (u == 8) ? p.first + di8[t] : p.first + di4[t];
						int nj = (u == 8) ? p.second + dj8[t] : p.second + dj4[t];

						if (ni >= 0 && ni < img.rows && nj >= 0 && nj < img.cols &&
							img(ni, nj) == 0 && labels(ni, nj) == 0) {
							labels(ni, nj) = label;
							Q.push(make_pair(ni, nj));
						}
					}
				}
				
				/*Mat_<Vec3b> intermediate_result = generate_color(labels);
				imshow("Intermediate Results", intermediate_result);
				waitKey(0);*/  
				
			}
		}
	}

	Mat_<Vec3b> color_result = generate_color(labels);
	imshow("Final Result", color_result);
	waitKey(0); 
}
void dfs(Mat_<uchar>& img, Mat_<int>& labels, int u)
{
	int di8[8] = { -1,-1,-1,0,0,1,1,1 };
	int dj8[8] = { -1,0,1,-1,1,-1,0,1 };
	int di4[4] = { -1,0,1,0 };
	int dj4[4] = { 0,-1,0,1 };
	int label = 0;

	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			if (img(i, j) == 0 && labels(i, j) == 0) {   
				label++;
				stack<pair<int, int>> S;
				labels(i, j) = label;
				S.push(make_pair(i, j));

				while (!S.empty()) {
					pair<int, int> p = S.top();
					S.pop();
					int r = p.first;
					int c = p.second;

					int N = (u == 8) ? 8 : 4;
					for (int t = 0; t < N; t++) {
						int ni = (u == 8) ? r + di8[t] : r + di4[t];
						int nj = (u == 8) ? c + dj8[t] : c + dj4[t];

						if (ni >= 0 && ni < img.rows && nj >= 0 && nj < img.cols &&
							img(ni, nj) == 0 && labels(ni, nj) == 0) {
							labels(ni, nj) = label;
							S.push(make_pair(ni, nj));
						}
					}
				}
				
			}
		}
	}
	Mat_<Vec3b> color_result = generate_color(labels);
	imshow("DFS", color_result);
	waitKey(0);  
}
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