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
		for (int j = 0; j < img.cols; j++){           //for all points P inside
			if (img(i, j) == 0 && labels(i, j) == 0){ //black and unlabeled
				label++;                              //new label
				queue< pair<int, int> > Q;            //create queue of pairs 
				labels(i, j) = label;                 //label
				Q.push(make_pair(i, j));              //add in queue

				while (!Q.empty())                    //while queue
				{
					pair<int, int> p = Q.front();   
					Q.pop();                          //pop
					for (int t = 0; t < u; t++) {     //for all neighbours P(ni,nj)
						int ni, nj;
						if (u == 8) {
							ni = p.first + di8[t];
							nj = p.second + dj8[t];
						}
						else {
							ni = p.first + di4[t];
							nj = p.second + dj4[t];
						}
						if (ni >= 0 && ni < img.rows && nj >= 0 && nj < img.cols &&     //P isInside
							img(ni, nj) == 0 && labels(ni, nj) == 0) {                  //black and unlabeled
							labels(ni, nj) = label;                                     //label
							Q.push(pair<int, int>(ni, nj));                             //add in queue
						}

					}
				}                                                                       //loop
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
	vector<vector<int>> edges(1000);                    //edges

	//First pass
	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			if (img(i, j) == 0 && labels(i, j) == 0) {  //black and unabled
				vector<int> L;                          //create vector L -> store neighboor labels in L
				for (int t = 0; t < u; t++) {           //for all neighbours
					int ni , nj ;
                    if (u == 8) {
                        ni = i + di8[t];
                        nj = j + dj8[t];
                    }
                    else {
                        ni = i + di4[t];
                        nj = j + dj4[t];
                    }
					if (ni >= 0 && ni < img.rows && nj >= 0 && nj < img.cols    //isInside 
                        && labels(ni, nj) > 0) {                                //label >0
						L.push_back(labels(ni, nj));                            //-> add in L
					}
				}
				if (L.size() == 0) {        //L empty
					label++;                //new label
					labels(i, j) = label;   //label
				}
				else {
					int x = L[0];   
					for (int i = 1; i < L.size(); i++) {
						if (L[i] < x) { 
							x = L[i];        //x smallest label
						}
					}
					labels(i, j) = x;        //assign smallest label from neighbors
                    for(int y : L)   {       //for all neighbour labels
						if (y != x) {        //label != x 
							edges[x].push_back(y);
							edges[y].push_back(x);
						}
					}
				}
			}
		}
	}

	Mat_<Vec3b> color_result = generate_color(labels);
	imshow("First Pass", color_result);
	waitKey(0);

	//Second pass
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
		for (int j = 0; j < img.cols; j++) {             //for all points P inside
			if (img(i, j) == 0 && labels(i, j) == 0) {   //black and unlabeled
				label++;                                 //new label
				stack<pair<int, int>> S;                 //create stack of pairs
				labels(i, j) = label;                    //label
				S.push(make_pair(i, j));                 //add in stack

				while (!S.empty()) {                    //while stack
					pair<int, int> p = S.top();     
					S.pop();                            //pop
					for (int t = 0; t < u; t++) {       //for all neighbours P(ni,nj)
						int ni, nj;
						if (u == 8) {
							ni = p.first + di8[t];
							nj = p.second + dj8[t];
						}
						else {
							ni = p.first + di4[t];
							nj = p.second + dj4[t];
						}
						if (ni >= 0 && ni < img.rows && nj >= 0 && nj < img.cols &&   //P is inside
							img(ni, nj) == 0 && labels(ni, nj) == 0) {                //black and unlabeled
							labels(ni, nj) = label;                                   //label
							S.push(make_pair(ni, nj));                                //add to stack
						}
					}                                                                 //loop
				}
				
			}
		}
	}
	Mat_<Vec3b> color_result = generate_color(labels);
	imshow("DFS", color_result);
	waitKey(0);  
}

void main() {
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
 
}
