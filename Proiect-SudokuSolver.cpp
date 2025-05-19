// SUDOKU SOLVER - project
// Andronescu Raluca
// UTCN - CTI en
// Group 30431
// 2025

#include "stdafx.h"
#include "common.h"
#include <opencv2/core/utils/logger.hpp>
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
#include <fstream>
using namespace std;
using namespace cv;
wchar_t* projectPath;

void testOpenImage()
{
    char fname[MAX_PATH];
    while (openFileDlg(fname))
    {
        Mat src;
        src = imread(fname);
        imshow("image", src);
        waitKey();
    }
}

void testOpenImagesFld()
{
    char folderName[MAX_PATH];
    if (openFolderDlg(folderName) == 0)
        return;
    char fname[MAX_PATH];
    FileGetter fg(folderName, "bmp");
    while (fg.getNextAbsFile(fname))
    {
        Mat src;
        src = imread(fname);
        imshow(fg.getFoundFileName(), src);
        if (waitKey() == 27) //ESC pressed
            break;
    }
}

void testImageOpenAndSave()
{
    _wchdir(projectPath);

    Mat src, dst;

    src = imread("Images/Lena_24bits.bmp", IMREAD_COLOR);	// Read the image

    if (!src.data)	// Check for invalid input
    {
        printf("Could not open or find the image\n");
        return;
    }

    // Get the image resolution
    Size src_size = Size(src.cols, src.rows);

    // Display window
    const char* WIN_SRC = "Src"; //window for the source image
    namedWindow(WIN_SRC, WINDOW_AUTOSIZE);
    moveWindow(WIN_SRC, 0, 0);

    const char* WIN_DST = "Dst"; //window for the destination (processed) image
    namedWindow(WIN_DST, WINDOW_AUTOSIZE);
    moveWindow(WIN_DST, src_size.width + 10, 0);

    cvtColor(src, dst, COLOR_BGR2GRAY); //converts the source image to a grayscale one

    imwrite("Images/Lena_24bits_gray.bmp", dst); //writes the destination to file

    imshow(WIN_SRC, src);
    imshow(WIN_DST, dst);

    waitKey(0);
}
//-----------------------------------Sudoku Solver
// extrag cells si compar cu o poza de referinta
// 
// Functions implementations
// Gaussian 2D filter (from Lab)
bool isInside(Mat img, int i, int j) {
    return (i >= 0 && i < img.rows && j >= 0 && j < img.cols);
}
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
Mat_< uchar> gaussian_2D(Mat_<uchar> img, int w) {
    
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

    return result;
}
vector<vector<Point>> borderTrace(const Mat_<uchar>& img) {
    int di[8] = { 0, -1, -1, -1, 0,  1,  1, 1 };
    int dj[8] = { 1,  1,  0, -1, -1, -1, 0, 1 };
    vector<vector<Point>> allBorders;
    Mat_<uchar> visited = Mat_<uchar>::zeros(img.size());


    return allBorders;
}
void drawCustomContours(Mat& image, const vector<Point>& border, Scalar color = Scalar(0, 255, 0), int thickness = 2) {
    for (int i = 0; i < border.size() - 1; i++) {
        line(image, border[i], border[i + 1], color, thickness);
    }

    if (!border.empty()) {
        line(image, border.back(), border[0], color, thickness); // Close the contour
    }
}
void printVector(const vector<Point>& v)
{
    for (Point val : v)
        cout << val.x << " " << val.y << " ";
    cout << "\n";
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
Mat_<uchar> multilevel_thresholding(Mat_<uchar> img) {
    vector<int> hist = calc_hist(img);
    vector<float> pdf = compute_pdf(img);

    int WH = 5;
    float TH = 0.005f;

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


//---------------Load image
Mat localizeSudoku(Mat& sudokuImage)
{
    Mat gray;
    cvtColor(sudokuImage, gray, COLOR_BGR2GRAY);
    imshow("Gray Image", gray);
    waitKey(0);
    
    //-----Blur  
    Mat_<uchar> blurred = gaussian_2D(gray, 7); 
    //Mat blurred2;
    //GaussianBlur(gray, blurred2, Size(5, 5), 0); //!
    imshow("Blurred Image", blurred);
    waitKey(0);

    //-----Treshhold  
    // Mat_<uchar> thresh = adaptive_thresholding(blurred, 21, 4.0);
    Mat_<uchar> thresh = multilevel_thresholding(blurred);
    //Mat thresh1;
    //adaptiveThreshold(blurred, thresh1, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY_INV, 11, 2); //! 
    imshow("Threshold Image", thresh);
    waitKey(0);

    //-----Contours  
    //vector<vector<Point>> contours2;
    //findContours(thresh, contours2, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE); //! 
    //Mat contourImage2 = sudokuImage.clone();
    //drawContours(contourImage2, contours2, -1, Scalar(0, 255, 0), 2);
    //imshow("All Contours imp", contourImage2);
     Mat contourImage2 = sudokuImage.clone();
    vector<vector<Point>> contours = borderTrace(thresh);
    drawContours(contourImage2, contours, -1, Scalar(0, 255, 0), 2);
    /*Mat contourImage = sudokuImage.clone();
    drawCustomContours(contourImage, border, Scalar(0, 255, 0), 2);
    if (contourImage.empty()) {
        cout << "Contour image is empty!" << endl;
        return sudokuImage;
    }
*/
    imshow("All Contours", contourImage2);
    waitKey(0);

    double maxArea = 0;
    vector<Point> biggest;

    for (auto& contour : contours)
    {
        double area = contourArea(contour);
        if (area > maxArea)
        {
            vector<Point> approx;
            approxPolyDP(contour, approx, 0.02 * arcLength(contour, true), true);
            if (approx.size() == 4)
            {
                biggest = approx;
                maxArea = area;
            }
        }
    }

    Mat boardOutline = sudokuImage.clone();
    if (biggest.size() == 4)
    {
        vector<vector<Point>> drawBiggest = { biggest };
        drawContours(boardOutline, drawBiggest, -1, Scalar(0, 0, 255), 5);
    }
    imshow("Biggest Contour", boardOutline);
    waitKey(0);

    if (biggest.size() != 4)
    {
        printf("No Sudoku grid detected.\n");
        return sudokuImage;
    }

    Point2f src[4];
    Point2f dst[4];

    sort(biggest.begin(), biggest.end(), [](Point a, Point b) { return a.y < b.y; });
    if (biggest[0].x < biggest[1].x)
    {
        src[0] = biggest[0];
        src[1] = biggest[1];
    }
    else
    {
        src[0] = biggest[1];
        src[1] = biggest[0];
    }

    if (biggest[2].x < biggest[3].x)
    {
        src[2] = biggest[2];
        src[3] = biggest[3];
    }
    else
    {
        src[2] = biggest[3];
        src[3] = biggest[2];
    }

    dst[0] = Point2f(0, 0);
    dst[1] = Point2f(800, 0);
    dst[2] = Point2f(0, 800);
    dst[3] = Point2f(800, 800);

    Mat matrix = getPerspectiveTransform(src, dst); 
    Mat warped;
    //warpPerspective(sudokuImage, warped, matrix, Size(800, 800));

    imshow("Warped Sudoku Board", warped);
    waitKey(0);

    return warped;
}

void loadSudokuImage(Mat& sudokuImage)
{
    _wchdir(projectPath);
    _wchdir(L"Images");

    char fname[MAX_PATH];
    while (openFileDlg(fname))
    {
        sudokuImage = imread(fname, IMREAD_COLOR);
        if (sudokuImage.empty())
        {
            printf("Could not open or find the Sudoku image!\n");
            continue;
        }
        resize(sudokuImage, sudokuImage, Size(600, 600));
        imshow("Loaded Sudoku Image", sudokuImage);
        waitKey(0);
        localizeSudoku(sudokuImage);
        break;
    }
}

int main()
{
    cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_FATAL);
    projectPath = _wgetcwd(0, 0);

    int op;
    do
    {
        system("cls");
        destroyAllWindows();
        printf("Sudoku solver:\n");
        printf("1. Load Sudoku Image\n");
        printf("0. Exit\n");
        printf("Option: ");
        scanf("%d", &op);

        if (op == 1)
        {
            Mat sudokuImage;
            loadSudokuImage(sudokuImage);

        }

    } while (op != 0);

    return 0;
}