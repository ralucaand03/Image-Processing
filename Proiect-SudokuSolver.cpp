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
#include <opencv2/ml.hpp>
#include <opencv2/ml/ml.hpp>

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
//---------------Blur
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
    //for (int i = 0; i < img.rows; i++) {
    //    for (int j = 0; j < img.cols; j++) {
    //        if (result(i, j) < 128) result(i, j)  =  (0.3* result(i, j));
    //    }
    //}
    return result;
} 
//---------------Thresholding
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
            // Invert black and white:
            result(i, j) = 255 - closest;
        }
    }

    return result;
}
Mat_<uchar>  thresholding(Mat_<uchar>& img) {

    int i_min = INT_MAX;
    int i_max = INT_MIN;

    vector<int>  histo = calc_hist(img);

    for (int i = 0; i < histo.size(); i++)
    {
        if (histo[i] > 0)
        {
            i_min = min(i_min, i);
            i_max = max(i_max, i);
        }
    }
    float T = (i_min + i_max) / 2.0;
    float last_T = 0;

    while (1) {
        float m1 = 0, m2 = 0, n1 = 0, n2 = 0;


        for (int i = i_min; i <= i_max; i++)
        {
            if (i < T)
            {
                m1 += (histo[i] * i);
                n1 += histo[i];
            }
            else
            {
                m2 += (histo[i] * i);
                n2 += histo[i];
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

    Mat_<uchar> result(img.size(), uchar(255));
    for (int i = 0; i < img.rows; i++)
    {
        for (int j = 0; j < img.cols; j++)
        {
            if (img(i, j) < T)
            {
                result(i, j) = 255;
            }
            else
            {
                result(i, j) = 0;
            }
        }
    }
    return result;
}
//---------------Border trace
vector<vector<Point>> borderTrace(const Mat_<uchar>& img) {
    int di[8] = { 0, -1, -1, -1, 0,  1,  1, 1 };
    int dj[8] = { 1,  1,  0, -1, -1, -1, 0, 1 };

    vector<vector<Point>> allBorders;
    Mat_<uchar> visited = Mat_<uchar>::zeros(img.size());

    for (int i = 0; i < img.rows; i++) {
        for (int j = 0; j < img.cols; j++) {
            if (img(i, j) == 255 && visited(i, j) == 0) {
                vector<Point> border;

                Point startPoint(j, i);
                Point currentPoint = startPoint;
                border.push_back(currentPoint);
                visited(i, j) = 1;

                int dir = 0;

                bool borderComplete = false;
                while (!borderComplete) {
                    bool foundNext = false;

                    for (int k = 0; k < 8; k++) {
                        int newDir = (dir + k) % 8;

                        int ni = currentPoint.y + di[newDir];
                        int nj = currentPoint.x + dj[newDir];

                        if (ni >= 0 && ni < img.rows && nj >= 0 && nj < img.cols && img(ni, nj) == 255) {
                            currentPoint = Point(nj, ni);
                            border.push_back(currentPoint);
                            visited(ni, nj) = 1;

                            dir = (newDir + 5) % 8;
                            foundNext = true;
                            break;
                        }
                    }
                    if (!foundNext || (border.size() > 2 && currentPoint == startPoint)) {
                        borderComplete = true;
                    }
                }
                if (border.size() > 50) {
                    allBorders.push_back(border);
                }
            }
        }
    } 
    return allBorders;
}
Mat drawBorders(const Mat& img, const vector<vector<Point>>& borders) {
    int thickness = 2;
    Mat result = Mat::zeros(img.size(), CV_8UC3);

    if (img.channels() == 3) {
        img.copyTo(result);
    }
    else {
        cvtColor(img, result, COLOR_GRAY2BGR);
    }

    srand(time(NULL));

    for (size_t i = 0; i < borders.size(); i++) {
        uchar r = rand() % 256;
        uchar g = rand() % 256;
        uchar b = rand() % 256;
        Scalar color(b, g, r);
        vector<Point> contour = borders[i];
        for (size_t j = 0; j < contour.size() - 1; j++) {
            line(result, contour[j], contour[j + 1], color, thickness);
        }
        if (contour.size() > 1) {
            line(result, contour[contour.size() - 1], contour[0], color, thickness);
        }
    }
    return result;
}
//---------------Wrap board
Mat warpSudokuBoard(const Mat& inputImage, const vector<Point>& corners, Size outputSize = Size(800, 800)) {
    if (corners.size() != 4) {
        cerr << "Error: corners vector must contain exactly 4 points." << endl;
        return inputImage.clone();
    }

    Point2f src[4];
    for (int i = 0; i < 4; i++) {
        src[i] = corners[i];
    }

    Point2f dst[4] = {
        Point2f(0, 0),
        Point2f(outputSize.width, 0),
        Point2f(0, outputSize.height),
        Point2f(outputSize.width, outputSize.height)
    };

    Mat matrix = getPerspectiveTransform(src, dst);

    Mat warped;
    warpPerspective(inputImage, warped, matrix, outputSize);

    return warped;
}

//---------------Localize
Mat_<uchar> localizeSudoku(Mat& sudokuImage)
{
    Mat gray;
    cvtColor(sudokuImage, gray, COLOR_BGR2GRAY);
    imshow("Gray Image", gray);
    waitKey(0);

    //-----Blur  
    Mat_<uchar> blurred = gaussian_2D(gray, 4);
    //Mat blurred2;
    //GaussianBlur(gray, blurred2, Size(5, 5), 0); //!
    imshow("Blurred Image", blurred);
    waitKey(0);

    //-----Treshhold  
    // Mat_<uchar> thresh = adaptive_thresholding(blurred, 21, 4.0);
    Mat_<uchar> thresh = thresholding(blurred);
    //Mat thresh1;
    //adaptiveThreshold(blurred, thresh1, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY_INV, 11, 2); //! 
    imshow("Threshold Image", thresh);
    waitKey(0);

    //-----Contours 
    vector<vector<Point>> contours = borderTrace(thresh);
    Mat contourImage2 = drawBorders(sudokuImage, contours);
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

     
    if (biggest.size() == 4)
    {
        vector<vector<Point>> drawBiggest = { biggest };
         Mat boardOutline = drawBorders(sudokuImage, drawBiggest );
         imshow("Biggest Contour", boardOutline);
        waitKey(0);
    }
    

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
    Mat warped = warpSudokuBoard(sudokuImage, biggest);
    imshow("Warped Sudoku Board", warped);
    waitKey(0);

    return warped;
}
//---------------Cells
void showSudokuCells(const Mat_<uchar>& sudokuBoard) {
    int cellHeight = sudokuBoard.rows / 9;
    int cellWidth = sudokuBoard.cols / 9;

    for (int i = 0; i < 9; i++) {
        for (int j = 0; j < 9; j++) {
            // Extract cell using simple row/col slicing
            Mat_<uchar> cell = sudokuBoard(Range(i * cellHeight, (i + 1) * cellHeight),
                Range(j * cellWidth, (j + 1) * cellWidth));

            // Show the cell
            imshow("Cell " + to_string(i) + "," + to_string(j), cell);
            waitKey(100); // small pause to show windows properly
        }
    }
}
vector<Mat_<uchar>> loadDigitTemplates() {
    _wchdir(projectPath);
    _wchdir(L"Digits");
    vector<Mat_<uchar>> templates;
    for (int digit = 1; digit <= 9; digit++) {
        string filename = "nr_" + to_string(digit) + ".png";
        Mat_<uchar> img = imread(filename, IMREAD_GRAYSCALE);
        if (img.empty()) {
            cerr << "Could not load " << filename << endl;
            continue;
        }
        /*imshow("Digit " + to_string(digit)  , img);
        waitKey(100);*/
        templates.push_back(img);
    }
    return templates;
}
Mat_<int> recognizeSudokuGrid(const Mat_<uchar>& sudokuBoard, const vector<Mat_<uchar>>& digitTemplates) {
    Mat_<int> grid(9, 9, 0); // Initialize with zeros (empty cells)

    // Calculate cell dimensions
    int cellHeight = sudokuBoard.rows / 9;
    int cellWidth = sudokuBoard.cols / 9;

    // Process each cell
    for (int row = 0; row < 9; row++) {
        for (int col = 0; col < 9; col++) {
            // Extract the cell
            Rect cellRect(col * cellWidth, row * cellHeight, cellWidth, cellHeight);
            Mat_<uchar> cell = sudokuBoard(cellRect).clone();

            // Preprocess the cell
            Mat_<uchar> processedCell;
            threshold(cell, processedCell, 0, 255, THRESH_BINARY_INV | THRESH_OTSU);

            // Remove noise
            Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(3, 3));
            morphologyEx(processedCell, processedCell, MORPH_OPEN, kernel);

            // Check if cell is empty
            double whitePixelRatio = countNonZero(processedCell) / (double)(cellWidth * cellHeight);
            if (whitePixelRatio < 0.03) {
                grid(row, col) = 0; // Empty cell
                continue;
            }

            // Find contours
            vector<vector<Point>> contours;
            vector<Vec4i> hierarchy;
            findContours(processedCell.clone(), contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

            // If no significant contours, cell is empty
            if (contours.empty()) {
                grid(row, col) = 0;
                continue;
            }

            // Find the largest contour
            int largestContourIdx = 0;
            double largestContourArea = 0;
            for (size_t i = 0; i < contours.size(); i++) {
                double area = contourArea(contours[i]);
                if (area > largestContourArea) {
                    largestContourArea = area;
                    largestContourIdx = i;
                }
            }

            // If largest contour is too small, cell is likely empty
            if (largestContourArea < 20) {
                grid(row, col) = 0;
                continue;
            }

            // Get bounding rectangle of the digit
            Rect boundingRect = cv::boundingRect(contours[largestContourIdx]);

            // Add padding
            int padding = 2;
            boundingRect.x = max(0, boundingRect.x - padding);
            boundingRect.y = max(0, boundingRect.y - padding);
            boundingRect.width = min(processedCell.cols - boundingRect.x, boundingRect.width + 2 * padding);
            boundingRect.height = min(processedCell.rows - boundingRect.y, boundingRect.height + 2 * padding);

            // Extract the digit
            Mat_<uchar> digit = processedCell(boundingRect).clone();

            // Match against each template
            int bestMatch = 0;
            double bestScore = -1;

            for (int i = 0; i < digitTemplates.size(); i++) {
                // Resize digit to match template size
                Mat_<uchar> resizedDigit;
                resize(digit, resizedDigit, digitTemplates[i].size());

                // Calculate match score
                Mat result;
                matchTemplate(resizedDigit, digitTemplates[i], result, TM_CCOEFF_NORMED);
                double minVal, maxVal;
                Point minLoc, maxLoc;
                minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc);

                // Update best match if score is higher
                if (maxVal > bestScore) {
                    bestScore = maxVal;
                    bestMatch = i + 1; // Templates are 1-indexed
                }
            }

            // If match confidence is too low, consider it empty
            if (bestScore < 0.4) {
                grid(row, col) = 0;
            }
            else {
                grid(row, col) = bestMatch;
            }

            // Uncomment for debugging
            /*
            cout << "Cell (" << row << "," << col << "): " << grid(row, col)
                 << " (score: " << bestScore << ")" << endl;
            imshow("Current Cell", cell);
            imshow("Processed Cell", processedCell);
            imshow("Extracted Digit", digit);
            waitKey(100);
            */
        }
    }

    return grid;
}
void printSudokuMatrix(const Mat_<int>& grid) {
    for (int i = 0; i < grid.rows; i++) {
        for (int j = 0; j < grid.cols; j++) {
            cout << grid(i, j) << " ";
            if ((j + 1) % 3 == 0 && j != grid.cols - 1)
                cout << "| ";
        }
        cout << endl;
        if ((i + 1) % 3 == 0 && i != grid.rows - 1)
            cout << "------+-------+------" << endl;
    }
}


//---------------Load image
void loadSudokuImage(Mat& sudokuImage, vector<Mat_<uchar>> digitTemplates)
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
        //cout << "------+-------+------" << endl;
        Mat_<uchar> wrappedSudoku = localizeSudoku(sudokuImage);
        Mat_<int> grid = recognizeSudokuGrid(wrappedSudoku,digitTemplates);
        printSudokuMatrix(grid);
        waitKey(0);
        break;
    }
}

int main()
{
    cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_FATAL);
    projectPath = _wgetcwd(0, 0);
    vector<Mat_<uchar>> digitTemplates = loadDigitTemplates();

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
             
             loadSudokuImage(sudokuImage, digitTemplates);

        }

    } while (op != 0);

    return 0;
}
