//Andronescu Raluca
//Image Processing Laboratory
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
//Lab_1 -- Colocviu 1
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
			grayImg(i, j) =  (average);
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
	imshow("Original", img);
	imshow("Binary Image ", bwImg);
	waitKey(0);
}
Mat_ <Vec3b>  computeHSV(Mat_<Vec3b>& img) {
	//if (img.empty()) return;
	Mat_<Vec3b> hsvImg(img.rows, img.cols);

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

			float Sat = 0;
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
void showHistogram(const string& name, vector<int> hist, const int hist_cols, const int hist_height) {

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
vector<float> compute_pdf(Mat_<uchar> img) {
	vector<float> pdf(256, 0.0f);
	int totalPixels = img.rows * img.cols;
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

			hist[g * m / 256]++;
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
			uchar pixel = img(i, j);
			int closest = maxima[0];
			int minDist = abs(pixel - maxima[0]);

			for (int m = 1; m < maxima.size(); m++) {
				int dist = abs(pixel - maxima[m]);
				if (dist < minDist) {
					closest = maxima[m];
					minDist = dist;
				}
			}
			result(i, j) = closest;
		}
	}
	return result;
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
void CenterMass(vector<Point>& pixels, double& i_mass, double& j_mass) {
	long long sum_of_i = 0;
	long long sum_of_j = 0;

	for (int i = 0; i < pixels.size(); i++) {
		sum_of_i += pixels[i].y;
		sum_of_j += pixels[i].x;
	}

	i_mass = (sum_of_i * 1.0) / Area(pixels);
	j_mass = (sum_of_j * 1.0) / Area(pixels);
}
double Elongation(vector<Point>& pixels, double i_mass, double j_mass) {
	double sum_iijj = 0;
	double sum_ii_at_2 = 0;
	double sum_jj_at_2 = 0;

	for (int i = 0; i < pixels.size(); i++) {
		double dif_currentP_mass_i = pixels[i].y - i_mass;
		double dif_currentP_mass_j = pixels[i].x - j_mass;
		sum_ii_at_2 += dif_currentP_mass_i * dif_currentP_mass_i;
		sum_jj_at_2 += dif_currentP_mass_j * dif_currentP_mass_j;
		sum_iijj += dif_currentP_mass_i * dif_currentP_mass_j;
	}

	double angle = 0.5 * (atan2(2.0 * sum_iijj, (sum_jj_at_2 - sum_ii_at_2)));

	return angle;
}
int Perimeter(Mat_<uchar>& img, int label_val) {
	int cnt = 0;

	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			if (img(i, j) == label_val) {

				bool isOnContour = false;

				for (int window_i = -1; window_i <= 1; window_i++) {
					for (int window_j = -1; window_j <= 1; window_j++) {
						if (!(window_i == 0 && window_j == 0)) {

							int neighbour_i = i + window_i;
							int neighbour_j = j + window_j;

							if (neighbour_i < 0 || neighbour_i >= img.rows || neighbour_j < 0 || neighbour_j >= img.cols) {
								isOnContour = true;
							}
							else {
								if (img(neighbour_i, neighbour_j) != label_val) {
									isOnContour = true;
								}
							}
						}
						if (isOnContour)
							break;
					}
					if (isOnContour)
						break;
				}
				if (isOnContour)
					cnt++;
			}
		}
	}
	return cnt;
}
void Projection(Mat_<uchar>& img, int label_val) {
	vector<int> proj_h(img.cols, 0);
	vector<int> proj_v(img.rows, 0);

	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			if (img(i, j) == label_val) {
				proj_h[j]++;
				proj_v[i]++;
			}
		}
	}

	Mat_<uchar> projection_h_img(img.rows, img.cols, (uchar)255);
	Mat_<uchar> projection_v_img(img.rows, img.cols, (uchar)255);

	for (int j = 0; j < img.cols; j++) {
		for (int i = img.rows - 1; i >= img.rows - proj_h[j] && i >= 0; i--) {
			projection_h_img(i, j) = 0;
		}
	}

	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < proj_v[i] && j < img.cols; j++) {
			projection_v_img(i, j) = 0;
		}
	}

	imshow("Horizontal Projection", projection_h_img);
	imshow("Vertical Projection", projection_v_img);
}
void util(Mat_<uchar>& img, int label_val) {
	vector<Point> objPixels = pixels(img, label_val);

	Mat_<Vec3b> clone_img;
	cvtColor(img, clone_img, COLOR_GRAY2BGR);
	int area = Area(objPixels);

	double i_mass, j_mass;
	CenterMass(objPixels, i_mass, j_mass);

	double axis = Elongation(objPixels, i_mass, j_mass);
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
	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			if (img(i, j) == 0 && labels(i, j) == 0) {
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
void two_pass(Mat_<uchar>& img, Mat_<int>& labels, int u) {
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
	int start_i = -1, start_j = -1;
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
		if (current == P0)
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
vector<int> buildDerivativeCode(vector<int>& chain)
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
//Lab7 -- Colocviu 2
Mat_<uchar> dilation(Mat_<uchar> src, Mat_<uchar> str_el) {
	Mat_<uchar> dst = src.clone();

	for (int i = 0; i < src.rows; i++) {
		for (int j = 0; j < src.cols; j++) {
			if (src(i, j) == 0) {
				for (int u = 0; u < str_el.rows; u++) {
					for (int v = 0; v < str_el.cols; v++) {
						if (str_el(u, v) == 0) {
							//neigbhour under element (u,v)
							int new_i = i + u - str_el.rows / 2;
							int new_j = j + v - str_el.cols / 2;
							if (isInside(src, new_i, new_j)) {
								dst(new_i, new_j) = 0;
							}
						}
					}
				}
			}
		}
	}

	return dst;
}
Mat_<uchar> erosion(Mat_<uchar> src, Mat_<uchar> str_el) {
	Mat_<uchar> dst = src.clone(); 
	for (int i = 0; i < src.rows; i++) {
		for (int j = 0; j < src.cols; j++) {
			if (src(i, j) == 0) {  // only check black pixels
				bool erode = false;

				for (int u = 0; u < str_el.rows; u++) {
					for (int v = 0; v < str_el.cols; v++) {
						if (str_el(u, v) == 0) {
							int new_i = i + u - str_el.rows / 2;
							int new_j = j + v - str_el.cols / 2;
							// if outside OR white under the kernel → erode!
							if (!isInside(src, new_i, new_j) || src(new_i, new_j) == 255) {
								erode = true;
								break;
							}
						}
					}
					if (erode) break;
				} 
				// if erosion needed, remove the pixel (make it white)
				if (erode)
					dst(i, j) = 255;
			}
		}
	}

	return dst;
}
//Lab8
float mean(Mat_<uchar>& img) {
	int M = img.rows * img.cols;
	float result = 0;
	vector<int> histo = calc_hist(img);
	for (int i = 0; i < histo.size(); i++)
	{
		result += (histo[i] * i);
	}
	return (result / (M * 1.0));
}
float standard_deviation(Mat_<uchar>& img, float mean_intensity) {
	int M = img.rows * img.cols;
	float result = 0;
	vector<int> histo = calc_hist(img);
	for (int i = 0; i < histo.size(); i++){
		result += (((i - mean_intensity) * (i - mean_intensity) * histo[i]) * 1.0);
	}
	result = (result / (M * 1.0));
	result = sqrt(result);
	return result;
}
vector<int> cumulative_histogram(Mat_<uchar>& img) {
	vector<int> histo = calc_hist(img);
	vector<int> cpdf(256);
	cpdf[0] = histo[0];

	for (int i = 1; i < histo.size(); i++)
	{
		cpdf[i] = cpdf[i - 1] + histo[i];
	} 
	return cpdf;
}
Mat_<uchar>  thresholding(Mat_<uchar>& img) {
  //🔵Step 1: Find intensity range
	int i_min = INT_MAX;
	int i_max = INT_MIN;
	vector<int>  histo = calc_hist(img);
	for (int i = 0; i <  histo.size(); i++)
	{
		if ( histo[i] > 0)
		{
			i_min = min(i_min, i);
			i_max = max(i_max, i);
		}
	}
	
	//🔵 Step 2: Initialize threshold
	float T = (i_min + i_max) / 2.0;
	float last_T = 0;
	
  //🔵Step 3: Iteratively find the best threshold
	while (1) {
		float m1 = 0, m2 = 0, n1 = 0, n2 = 0;
		for (int i = i_min; i <= i_max; i++){
			if (i < T){
				m1 += ( histo[i] * i);
				n1 += histo[i];
			}	else {
				m2 += (histo[i] * i);
				n2 +=histo[i];
			}
		}
		m1 = m1 / n1;
		m2 = m2 / n2;
		last_T = T;
		T = (m1 + m2) / 2.0;
		if (abs(T - last_T) <= 0.0) {
			break;
		}
	}
	
  //🔵 Step 4: Apply threshold to image
	Mat_<uchar> result(img.size(), uchar(255));
	for (int i = 0; i < img.rows; i++){
		for (int j = 0; j < img.cols; j++){
			if (img(i, j) < T){
				result(i, j) = 0;
			}	else {
				result(i, j) = 255;
			}
		}
	}
	
	return result;
}
Mat_<uchar> histogram_stretching_shrinking(Mat_<uchar>& img, int min_val, int max_val) {
	Mat_<uchar> result(img.size(), uchar(255));
	//🔵Step 1: Find intensity min and max
	int i_min = INT_MAX;
	int i_max = INT_MIN;
	vector<int> histo = calc_hist(img);
	for (int i = 0; i < histo.size(); i++) {
		if (histo[i] > 0) {
			i_min = min(i_min, i);
			i_max = max(i_max, i);
		}
	}
	//🔵Step 2: Map each pixel to the new range
	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			result(i, j) = ((img(i, j) - i_min) * (max_val - min_val)) / (i_max - i_min) + min_val;
		}
	}

	return result;
}
//Lab9.1
Mat_<float> convolution(Mat_<uchar> img, Mat_<float> H) {
	Mat_<float> dst(img.rows, img.cols);
	float s = 0; 
	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) { 
			s = 0;
			for (int u = 0; u < H.rows; u++) {
				for (int v = 0; v < H.cols; v++) {
					int ii = i + u - (H.rows / 2);
					int jj = j + v - (H.cols / 2); 
					if (isInside(img, ii, jj)) {
						s += img(ii, jj) * H(u, v);
					}
				}
			} 
			dst(i, j) = s;
		}
	} 
	return dst;
}
Mat_<uchar> normalization(Mat_<float> img, Mat_<float> H) {
	Mat_<uchar> dst(img.rows, img.cols);
	float pos = 0, neg = 0;
	float a, b;
	for (int u = 0; u < H.rows; u++) {
		for (int v = 0; v < H.cols; v++) {
			if (H(u, v) > 0)
				pos += H(u, v);
			else
				neg += H(u, v);
		}
	}
	b = pos * 255;
	a = neg * 255;
	for (int i = 0; i < img.rows; i++)
		for (int j = 0; j < img.cols; j++)
			dst(i, j) = (img(i, j) - a) * 255 / (b - a);
	return dst;
}
//Lab9.2
void centering_transform(Mat img) {
	//expects floating point image
	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			img.at<float>(i, j) = ((i + j) & 1) ? -img.at<float>(i, j) : img.at<float>(i, j);
		}
	}
}
Mat generic_frequency_domain_filter(Mat src,int x) {
	//convert input image to float image
	Mat srcf;
	src.convertTo(srcf, CV_32FC1);
	//centering transformation
	centering_transform(srcf);
	//perform forward transform with complex image output
	Mat fourier;
	dft(srcf, fourier, DFT_COMPLEX_OUTPUT);
	//split into real and imaginary channels
	Mat channels[] = { Mat::zeros(src.size(), CV_32F), Mat::zeros(src.size(), CV_32F) };
	split(fourier, channels); // channels[0] = Re(DFT(I)), channels[1] = Im(DFT(I))
	//calculate magnitude and phase in floating point images mag and phi
	Mat mag, phi;
	magnitude(channels[0], channels[1], mag);
	phase(channels[0], channels[1], phi);
	//display the phase and magnitude images here

	//insert filtering operations on Fourier coefficients here
	// ...... calculate X'uv
	if (x == 0) {
		int R = 20;
		for (int i = 0; i < mag.rows; i++) {
			for (int j = 0; j < mag.cols; j++) {
				float dist = pow((mag.cols / 2) - i, 2) + pow((mag.rows / 2) - j, 2);
				if (dist > pow(R, 2)) {
					mag.at<float>(i, j) = 0;
				}
			}
		}
	}
	else {
		float A = 50.0f;   
		for (int i = 0; i < mag.rows; i++) {
			for (int j = 0; j < mag.cols; j++) {
				float du = (float)(i - (mag.rows / 2));
				float dv = (float)(j - (mag.cols / 2));
				float dist2 = (du * du + dv * dv);
				float gaussian = 1.0f - exp(-dist2 / (A * A));
				mag.at<float>(i, j) *= gaussian;
			}
		}

	}
	//store in real part in channels[0] and imaginary part in channels[1]
	// ...... start here
	for (int i = 0; i < mag.rows; i++) {
		for (int j = 0; j < mag.cols; j++) {
			channels[0].at<float>(i, j) = mag.at<float>(i, j) * cos(phi.at<float>(i, j));
			channels[1].at<float>(i, j) = mag.at<float>(i, j) * sin(phi.at<float>(i, j));
		}
	}
	//perform inverse transform and put results in dstf
	Mat dst, dstf;
	merge(channels, 2, fourier);
	dft(fourier, dstf, DFT_INVERSE | DFT_REAL_OUTPUT | DFT_SCALE);
	//inverse centering transformation
	centering_transform(dstf);
	//normalize the result and put in the destination image
	normalize(dstf, dst, 0, 255, NORM_MINMAX, CV_8UC1);
	//=================================================
	//  ...... Display Magnitude  
	Mat mag_disp;
	mag += Scalar::all(1);
	log(mag, mag_disp);
	normalize(mag_disp, mag_disp, 0, 255, NORM_MINMAX);
	mag_disp.convertTo(mag_disp, CV_8UC1);
	imshow("Magnitude", mag_disp);

	//  ...... Display Phase  
	Mat phi_disp;
	normalize(phi, phi_disp, 0, 255, NORM_MINMAX);
	phi_disp.convertTo(phi_disp, CV_8UC1);
	imshow("Phase", phi_disp);

	//Note: normalizing distorts the resut while enhancing the image display in the range [0,255].
	//For exact results (see Practical work 3) the normalization should be replaced with convertion:
	//dstf.convertTo(dst, CV_8UC1);
	return dst;
}
//Lab10 
Mat_<uchar >median_filter(Mat_<uchar> img,int w) {
	double t = (double)getTickCount();
	Mat_ <uchar>  dst(img.rows, img.cols);
	int half = w / 2;
	for (int i = 0; i < img.rows ; i++) {
		for (int j = 0; j < img.cols ; j++) {
			vector<uchar> neighbours;
			for (int wi = -half; wi <= half; wi++) {
				for (int wj = -half; wj <= half; wj++) {
					int i2 = i + wi;
					int j2 = j + wj;
					if (isInside(img, i2, j2)) {
						neighbours.push_back(img(i2, j2));
					}
				}
			}
			sort(neighbours.begin(), neighbours.end());
			dst(i, j) = neighbours[neighbours.size()/2];
		}
	}
	t = ((double)getTickCount() - t) / getTickFrequency();
	printf("Time for Median Filtered = %.3f [ms]\n", t * 1000);
	return dst;
}
Mat_< uchar> gaussian_2D(Mat_<uchar> img,int w) {
	double t = (double)getTickCount();
	Mat_<uchar> result = img.clone(); 
	float standard_dev = w / 6.0f; 
	int half = w / 2;
	double sum = 0;
	float coeff = 1.0f / (2.0f * CV_PI * pow(standard_dev, 2)); 
	Mat_<float> kernel(w, w);
	for (int i = 0; i < w; i++) {
		for (int j = 0; j < w; j++) {
			float x = i - half;
			float y = j - half;
			kernel(i, j) = coeff * exp(-(pow(x, 2) + pow(y, 2)) / (2 * pow(standard_dev, 2)));
		}
	}
	 
	Mat_<float> conv_img = convolution(img, kernel);
	result = normalization(conv_img, kernel);
	t = ((double)getTickCount() - t) / getTickFrequency();
	printf("Time for Gaussian 2D = %.3f [ms]\n", t * 1000);
	return result;
} 
Mat_<uchar> gaussian_1D(Mat_<uchar> img,int w, int x) {
	double t = (double)getTickCount();
	Mat_<uchar> result ;
	float standard_dev = w / 6.0f;
	int half = w / 2;
	double sum = 0;
	float coeff = 2.0f * CV_PI;
	Mat_<float> H;

	if (x == 1) {
		Mat_<float> kernel(1, w);
		for (int i = 0; i < w; i++) {
			float x = i - half;

			kernel(0,i) = exp(-(pow(x, 2))) / (sqrt(coeff) * standard_dev);
		}
		H = kernel.clone();
	} 
	else {
		Mat_<float> kernel(w, 1);
		for (int j = 0; j < w; j++) {
			float y = j - half;
			kernel(j,0) = exp(-(pow(y, 2))) / (sqrt(coeff) * standard_dev);
		}
		H = kernel.clone();
	}

	Mat_<float> conv_img = convolution(img, H);
	result = normalization(conv_img, H);

	t = ((double)getTickCount() - t) / getTickFrequency();
	printf("Time for Gaussian 1D = %.3f [ms]\n", t * 1000);
	return result;
}
//Lab11
Mat_ <uchar> canny_edge_detection(Mat_<uchar> img)  {
	Mat_<float> sk_x = (Mat_<float>(3, 3) << -1, 0, 1, -2, 0, 2, -1, 0, 1);
	Mat_<float> sk_y = (Mat_<float>(3, 3) << 1, 2, 1, 0, 0, 0, -1, -2, -1);
	Mat_<float> imgx = convolution(img, sk_x);
	Mat_<float> imgy = convolution(img, sk_y);
	imshow("IMG", img);
	imshow("IMG  x", abs(imgx)/255);
	imshow("IMG  y", abs(imgy)/255);
	waitKey(0);
	int r = img.rows;
	int c = img.cols;
	Mat_<float> mag(r, c);
	Mat_<float> phi(r, c);
	for (int i = 0; i < r; i++) {
		for (int j = 0; j < c; j++) {
			mag(i, j) = sqrt(pow(imgx(i, j), 2) + pow(imgy(i, j), 2));
			phi(i, j) = atan2(imgy(i, j), imgx(i, j));
		}
	}

	Mat_<uchar> dir(r, c);
	for (int i = 0; i < r; i++) {
		for (int j = 0; j < c; j++) {
			float ang = phi(i, j);
			if (phi(i, j) <= 0) ang = phi(i, j) + 2 * CV_PI;
			dir(i, j) =  (ang * (8 / (2 * CV_PI))) + 0.5 ;
			dir(i, j) = dir(i, j) % 8;
		}
	}

	int di[] = { 0,-1,-1,-1,0,1,1,1 };
	int dj[] = { 1,1,0,-1,-1,-1,0,1 };

	Mat_<float> mt(r, c);
	mt = mag.clone();
	for (int i = 0; i < r; i++) {
		for (int j = 0; j < c; j++) {
			int ni = i + di[dir(i, j)];
			int nj = j + dj[dir(i, j)];

			int opi = i - di[dir(i, j)];
			int opj = j - dj[dir(i, j)];

			if (isInside(mt, ni, nj) && mag(ni, nj) > mag(i, j))
				mt(i, j) = 0;
			if (isInside(mt, opi, opj) && mag(opi, opj) > mag(i, j))
				mt(i, j) = 0;
		}
	}
	return mt;
}
Mat_<uchar> edge_linking(Mat_<uchar> mt, int t1, int t2) {
	Mat_<uchar> edges = Mat_<uchar>::zeros(mt.size());

	for (int i = 0; i < mt.rows; i++) {
		for (int j = 0; j < mt.cols; j++) {
			if (mt(i, j) >= t2) {
				edges(i, j) = 255;  
			}
			else if (mt(i, j) >= t1) {
				edges(i, j) = 128; 
			}
		}
	}

	queue<Point> Q;
	for (int i = 0; i < edges.rows; i++) {
		for (int j = 0; j < edges.cols; j++) {
			if (edges(i, j) == 255) {
				Q.push(Point(j, i));
			}
		}
	}

	int dx[8] = { -1, -1, -1, 0, 0, 1, 1, 1 };
	int dy[8] = { -1, 0, 1, -1, 1, -1, 0, 1 };

	while (!Q.empty()) {
		Point p = Q.front();
		Q.pop();

		for (int n = 0; n < 8; n++) {
			int ni = p.y + dy[n];
			int nj = p.x + dx[n];

			if (isInside(mt, ni, nj) && edges(ni, nj) == 128) {
				edges(ni, nj) = 255;
				Q.push(Point(nj, ni));

			}
		}
	}
	for (int i = 0; i < edges.rows; i++) {
		for (int j = 0; j < edges.cols; j++) {
			if (edges(i, j) == 128) {
				edges(i, j) = 0; 
			}
		}
	}

	return edges;

}
void main() {
	cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_FATAL);
	//-------------------------Lab11
	/*
	Mat_<uchar> img = imread("Images/saturn.bmp", IMREAD_GRAYSCALE);
	Mat_<uchar> mt = canny_edge_detection(img);
	imshow("Mag T", mt);
	waitKey(0);
	int t1 = 100;
	int t2 = 170;
	Mat_<uchar> edges = edge_linking(mt, t1, t2);
	imshow("Edges", edges);
	waitKey(0);
	*/
	//-------------------------Lab10
	/*
	//median
	Mat_<uchar> img = imread("Images/portrait_Salt&Pepper1.bmp", IMREAD_GRAYSCALE);
	imshow("Original Image", img);
	waitKey(0); 
	Mat_<uchar> mf = median_filter(img,3);
	imshow("Median Filtered Image", mf);
	waitKey(0);
	//gaussian2D
	Mat_<uchar> img2 = imread("Images/portrait_Gauss2.bmp", IMREAD_GRAYSCALE);
	imshow("Original Image2", img2);
	waitKey(0);
	Mat_<uchar> gs = gaussian_2D(img2, 5);
	imshow("Gaussian 2D Image", gs);
	waitKey(0);
	//1D
	Mat_<uchar> img3 = imread("Images/portrait_Gauss1.bmp", IMREAD_GRAYSCALE);
	imshow("Original Image3", img3);
	waitKey(0);
	Mat_<uchar> forx = gaussian_1D(img3, 5,1);
	Mat_<uchar> fory = gaussian_1D(forx, 5, 0);
	imshow("Gaussian 1D Image", fory);
	waitKey(0);*/
	//-------------------------Lab_9.1
	/*Mat_<uchar> img = imread("Images/cameraman.bmp", IMREAD_GRAYSCALE);
	Mat_<float> mean_filter = (Mat_<double>(3, 3) << 1, 1, 1, 1, 1, 1, 1, 1, 1);
	Mat_<float> conv_res = convolution(img, mean_filter);
	Mat_<uchar> new_img_blurred = normalization(conv_res, mean_filter);
	imshow("Original Image", img);
	imshow("Mean Filtered Image", new_img_blurred);
	waitKey(0);

	Mat_<float> mean_filter5(5, 5);
	mean_filter5.setTo(1);
	conv_res = convolution(img, mean_filter5);
	Mat_<uchar> new_img = normalization(conv_res, mean_filter5);
	imshow("Original Image", img);
	imshow("Mean Filtered 5x5", new_img);
	waitKey(0); 

	Mat_<float> gaussian_filter = (Mat_<double>(3, 3) << 1, 2, 1, 2, 4, 2, 1, 2, 1);
	conv_res = convolution(img, gaussian_filter);
	new_img = normalization(conv_res, gaussian_filter);
	imshow("Original Image", img);
	imshow("Gaussian Filtered Image", new_img);
	waitKey(0);

	Mat_<float> laplace_filter = (Mat_<double>(3, 3) << 0, -1, 0, -1, 4, -1, 0, -1, 0);
	conv_res = convolution(img, laplace_filter);
	new_img = normalization(conv_res, laplace_filter);
	imshow("Original Image", img);
	imshow("Laplace Filtered Image", new_img);
	waitKey(0);

	conv_res = convolution(new_img_blurred, laplace_filter);
	new_img = normalization(conv_res, laplace_filter);
	imshow("Original Image", img);
	imshow("Laplace Filtered on Blurred Image", new_img);
	waitKey(0);

	Mat_<float> high_pass_filter = (Mat_<double>(3, 3) << 0, -1, 0, -1, 5, -1, 0, -1, 0);
	conv_res = convolution(img, high_pass_filter);
	new_img = normalization(conv_res, high_pass_filter);
	imshow("Original Image", img);
	imshow("High Pass Filtered Image", new_img);
	waitKey(0); */
	//-------------------------Lab_9.2
	/*Mat_<uchar> img = imread("Images/cameraman.bmp", IMREAD_GRAYSCALE);
	auto dst = generic_frequency_domain_filter(img,1);
	imshow("Image before  ", img);
	waitKey(0); 
	imshow("Image after  ", dst);
	waitKey(0);*/
	//-------------------------Lab_8
	/*Mat_<uchar> img = imread("Images/balloons.bmp", IMREAD_GRAYSCALE);
	float mean_intensity = mean(img);
	cout << "Mean intensity: " << mean_intensity << endl; 
	float standarddev= standard_deviation(img, mean_intensity);
	cout << "Standard Deviation: " << standarddev << endl;
	vector<int> cumulative = cumulative_histogram(img);
	showHistogram("cumulative_histogram", cumulative, 256, 400);
	waitKey(0);
	Mat_<uchar> img2 = imread("Images/eight.bmp", IMREAD_GRAYSCALE);
	imshow("Image before thresholding", img2);
	waitKey(0);
	Mat_<uchar> img_result = thresholding(img2);
	imshow("Image after thresholding", img_result);
	waitKey(0);
	Mat_<uchar> s1 = imread("Images/Hawkes_Bay_NZ.bmp", IMREAD_GRAYSCALE);
	Mat_<uchar> s11  = histogram_stretching_shrinking(s1, 50, 200);
	Mat s1_resized, s11_resized;
	resize(s1, s1_resized, Size(), 0.5, 0.5);     // 50% scale
	resize(s11, s11_resized, Size(), 0.5, 0.5);
	imshow("original1", s1_resized);
	imshow("stretching", s11_resized);
	waitKey(0);
	Mat_<uchar> s2 = imread("Images/wheel.bmp", IMREAD_GRAYSCALE);
	Mat_<uchar> s22 = histogram_stretching_shrinking(s2, 50, 200);
	imshow("original2 ", s2);
	imshow("shrinking", s22);
	waitKey(0);*/
	//-------------------------Lab_7
	/*Mat_<uchar> img = imread("Images/1_Dilate/wdg2ded1_bw.bmp", IMREAD_GRAYSCALE);
	uchar data[] = {
		255,   0, 255,
		  0,   0,   0,
		255,   0,   0
	};
	Mat_<uchar> str_el(3, 3, data);
	Mat_<uchar> img2 = dilation(img, str_el);
	imshow("Original", img );
	imshow("Dilation", img2);
	waitKey(0);
	Mat_<uchar> img3 = erosion(img2, str_el);
	imshow("Erosion", img3);
	waitKey(0);*/
	////------------------------Lab_6
	/*
	 Mat_<uchar> img = imread("Images/triangle_up.bmp", IMREAD_GRAYSCALE);
	 if (img.empty())
	 {
	 	cerr << "Could not open or find the image!\n";
	 	return;
	 }
	 imshow("image", img);
	 waitKey(0);
	 vector<Point> contour = borderTrace(img);
	 Mat_<Vec3b> colorImg;
	 cvtColor(img, colorImg, COLOR_GRAY2BGR);
	 for (Point pt : contour)
	 	colorImg(pt.y, pt.x) = Vec3b(0, 0, 255);
	 imshow("Traced Border", colorImg);
	 waitKey(0);
	 vector<int> chain = buildChainCode(contour);
	 cout<<"Chain Code: "<<"\n";
	 printVector(chain);
	 vector<int> der = buildDerivativeCode(chain);
	 cout << "Derivative Code: " << "\n";
	 printVector(der);
	 Point start;
	 vector<int> chain_rec;
	 readChainCode("Images/reconstruct.txt", start, chain_rec);
	 reconstructs(start, chain_rec);
	*/
	////------------------------Lab_5
	/*Mat_<uchar> img = imread("Images/shapes.bmp", IMREAD_GRAYSCALE);
	Mat_<int> labels = Mat::zeros(img.size(), CV_32SC1);
	imshow("img", img);
	waitKey(0); 
	two_pass(img, labels,8);
	bfs(img, labels, 8);
	dfs(img, labels, 8);
	bfs_visualize(img, labels, 8);*/
	////------------------------Lab_4
	/*loadAndSetupImage("Images/oval_obl.bmp");
	waitKey(0);*/
	////------------------------Lab_3
	 /*Mat_<uchar> img = imread("Images/cameraman.bmp", IMREAD_GRAYSCALE);
	 if (img.empty()) return;
	 Mat_<Vec3b> colorimg = imread("Images/Lena_24bits.bmp");
	 imshow("Original", img);
	 vector<int> hist = calc_hist(img);
	 showHistogram("Histogram", hist, 256, 400);
	 waitKey(0);
	 int m = 128; 
	 vector<int> custom_hist = hist_custom_bins(img, m);
	 showHistogram("Custom Bins", hist, 100, 300);
	 waitKey(0); 
	 Mat simplified = multilevel_thresholding(img);
	 imshow("Multilevel Thresholding", simplified);
	 waitKey(0);
	 Mat hue = multilevel_thresholding(colorimg);
	 imshow("Multilevel Thresholding", hue);
	 waitKey(0);*/
	////------------------------Lab_2
	/*split_channels();
	convert_to_grayscale();
	convert_grayscale_to_BW();
	Mat_<Vec3b> img = imread("Images/Lena_24bits.bmp");
	imshow("Original", img);
	waitKey(0);
	Mat_<Vec3b> hsvimg = computeHSV(img);
	imshow("HSV", hsvimg);
	waitKey(0);
	Mat_<Vec3b> rgbimg(hsvimg.rows, hsvimg.cols);
	cvtColor(hsvimg, rgbimg, COLOR_HSV2BGR);
	imshow("RGB", rgbimg);
	waitKey(0);*/
	////------------------------Lab_1
	/*negative_image();
	 additive_factor(15);
	 multiplicative_factor(15);
	 create_img();
	 create3x3float();
	 symethric_img();*/
}
