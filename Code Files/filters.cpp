// Authors: Danny Chinn, Priyanshu Ranka

#include "filters.h"
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include "faceDetect.h"


using namespace cv;
using namespace std;

// Default greyscale filter using opencv cvtColor
int defaultGreyscale( cv::Mat& src, cv::Mat& dst )
{
    cv::Mat filteredFrame;
    cv::cvtColor(src, dst, COLOR_BGR2GRAY);  // Convert to greyscale
    return 0;
}

// Naive greyscale using custom formula
int greyscale( cv::Mat& src, cv::Mat& dst )
{
    // Iterate through pixels
    for(int i=0;i<src.rows;i++) {
        for(int j=0;j<src.cols;j++) {
            uchar blue_tmp = src.at<cv::Vec3b>(i,j)[0]; // save the blue channel to tmp
            uchar green_tmp = src.at<cv::Vec3b>(i,j)[1]; // save the green channel to tmp
            uchar red_tmp = src.at<cv::Vec3b>(i,j)[2]; // save the blue channel to tmp

            uchar grey = (blue_tmp + green_tmp) / 2 - red_tmp; // random formula to convert to greyscale
            
            dst.at<cv::Vec3b>(i,j)[0] = grey;
            dst.at<cv::Vec3b>(i,j)[1] = grey;
            dst.at<cv::Vec3b>(i, j)[2] = grey;
        }
    }
    return 0;
}

// Sepia filter using openCV transform
int filterSepia( cv::Mat& src, cv::Mat& dst )
{
    // Define the sepia kernel as a 3x3 matrix
    cv::Mat sepiaKernel = (cv::Mat_<float>(3, 3) <<
        0.272, 0.534, 0.131,
        0.349, 0.686, 0.168,
        0.393, 0.769, 0.189);

    // Ensure 3 channels in image
    if (src.channels() != 3) {
        cv::cvtColor(src, src, cv::COLOR_GRAY2BGR); // Convert grayscale to BGR
    }

    // Convert the source image to a floating-point representation
    cv::Mat srcFloat;
    src.convertTo(srcFloat, CV_32F);

    // Apply the sepia transformation
    cv::transform(srcFloat, dst, sepiaKernel);

    // Clip values to the 0-255 range and convert back to 8-bit
    cv::convertScaleAbs(dst, dst);

    return 0;
}

// Naive 5x5 blur filter with single nested for loop
int blur5x5_1( cv::Mat &src, cv::Mat &dst )
{
    // Iterate over each row
    for(int i=2;i<src.rows - 2;i++) {
        // Iterate over each column
        for(int j=2;j<src.cols - 2;j++) {
            // Iterate over each color channel
            for (int c = 0; c < 3; c++) {
                // Apply filter
                dst.at<cv::Vec3b>(i,j)[c] = (src.at<cv::Vec3b>(i-2,j-2)[c] + src.at<cv::Vec3b>(i-2,j-1)[c] * 2 + src.at<cv::Vec3b>(i-2,j)[c] * 4 + src.at<cv::Vec3b>(i-2,j+1)[c] * 2 + src.at<cv::Vec3b>(i-2,j+2)[c] 
                                        + src.at<cv::Vec3b>(i-1,j-2)[c] * 2 + src.at<cv::Vec3b>(i-1,j-1)[c] * 4 + src.at<cv::Vec3b>(i-1,j)[c] * 8 + src.at<cv::Vec3b>(i-1,j+1)[c] * 4 + src.at<cv::Vec3b>(i-1,j+2)[c] * 2 
                                        + src.at<cv::Vec3b>(i,j-2)[c] * 4 + src.at<cv::Vec3b>(i,j-1)[c] * 8 + src.at<cv::Vec3b>(i,j)[c] * 16 + src.at<cv::Vec3b>(i,j+1)[c] * 8 + src.at<cv::Vec3b>(i,j+2)[c] * 4 
                                        + src.at<cv::Vec3b>(i+1,j-2)[c] * 2 + src.at<cv::Vec3b>(i+1,j-1)[c] * 4 + src.at<cv::Vec3b>(i+1,j)[c] * 8 + src.at<cv::Vec3b>(i+1,j+1)[c] * 4 + src.at<cv::Vec3b>(i+1,j+2)[c] * 2 
                                        + src.at<cv::Vec3b>(i+2,j-2)[c] + src.at<cv::Vec3b>(i+2,j-1)[c] * 2 + src.at<cv::Vec3b>(i+2,j)[c] * 4 + src.at<cv::Vec3b>(i+2,j+1)[c] * 2 + src.at<cv::Vec3b>(i+2,j+2)[c]) / 100;
            }
        }
    }
    return 0;
}

int blur5x5_2( cv::Mat& src, cv::Mat& dst )
{

    // Temporary image for partial sums in 16bit
    cv::Mat tmp(src.size(), CV_16SC3);

    // 5x5 binomial filter
    int blurFilter[5] = {1, 2, 4, 2, 1};

    // Vertical pass
    for(int y = 2; y < src.rows - 2; y++)
    {
        for(int x = 0; x < src.cols; x++)
        {
            int sumB = 0, sumG = 0, sumR = 0;

            // sum over neighbors in y-direction
            for(int k = -2; k <= 2; k++)
            {
                cv::Vec3b pixel = src.at<cv::Vec3b>(y + k, x);
                sumB += pixel[0] * blurFilter[k + 2];
                sumG += pixel[1] * blurFilter[k + 2];
                sumR += pixel[2] * blurFilter[k + 2];
            }

            // store partial sums
            tmp.at<cv::Vec3s>(y, x) = cv::Vec3s(sumB, sumG, sumR);
        }
    }

    // Horizontal pass
    for(int y = 2; y < tmp.rows - 2; y++)
    {
        for(int x = 2; x < tmp.cols - 2; x++)
        {
            int sumB = 0, sumG = 0, sumR = 0;

            // sum over neighbors in x-direction
            for(int k = -2; k <= 2; k++)
            {
                // read from tmp
                cv::Vec3s pixel = tmp.at<cv::Vec3s>(y, x + k);
                sumB += pixel[0] * blurFilter[k + 2];
                sumG += pixel[1] * blurFilter[k + 2];
                sumR += pixel[2] * blurFilter[k + 2];
            }

            // Rounding, two passes with 10 filter sum so 10*10
            sumB = (sumB + 50) / 100;
            sumG = (sumG + 50) / 100;
            sumR = (sumR + 50) / 100;

            // clamp to [0..255] for 8-bit
            sumB = std::max(0, std::min(255, sumB));
            sumG = std::max(0, std::min(255, sumG));
            sumR = std::max(0, std::min(255, sumR));

            dst.at<cv::Vec3b>(y, x) = cv::Vec3b(sumB, sumG, sumR);
        }
    }

    return 0;
}

// FIX CANNOT USE OPENCV FILTER2D
int blur5x5_3(cv::Mat& src, cv::Mat& dst) {
    // Define the separable blur kernels
    cv::Mat blurKernelH = (cv::Mat_<float>(1, 5) << 1, 2, 4, 2, 1);
    cv::Mat blurKernelV = (cv::Mat_<float>(5, 1) << 1, 2, 4, 2, 1);

    // Normalize the kernels to ensure the sum equals 1
    blurKernelH /= 10.0;
    blurKernelV /= 10.0;

    // Create tmp frame for intermediate step
    cv::Mat tmp;

    // Horizontal blur
    cv::filter2D(src, tmp, -1, blurKernelH, cv::Point(-1, -1), 0, cv::BORDER_DEFAULT);

    // Vertical blur
    cv::filter2D(tmp, dst, -1, blurKernelV, cv::Point(-1, -1), 0, cv::BORDER_DEFAULT);

    return 0; // Success
}

// Sobel filter for vertical edges
int sobelX3x3( cv::Mat &src, cv::Mat &dst )
{
    // Define the separable sobel kernels
    int sobelHorizontal[3] = {-1, 0, 1};
    int sobelVertical[3] = {1, 2, 1};

    // Create tmp frame for intermediate step
    cv::Mat tmp = cv::Mat::zeros(src.size(), CV_16SC3);
    dst = cv::Mat::zeros(src.size(), CV_16SC3);

    // Horizontal gradient
    for(int y=0; y<src.rows; y++) {
        for(int x=1; x<src.cols - 1; x++) {
            cv::Vec3s gradient = {0, 0, 0};

            // Apply the filter to the pixel
            for (int k = -1; k < 2; k++) {
                cv::Vec3b pixel = src.at<cv::Vec3b>(y, x + k);

                // Gradient of each color channel
                for (int c = 0; c < 3; c++) {
                    gradient[c] += pixel[c] * sobelHorizontal[k + 1];
                }
            }

            tmp.at<cv::Vec3s>(y, x) = gradient;

        }
    }

    // Vertical gradient
    for(int y=1; y<tmp.rows - 1; y++) {
        for(int x=0; x<tmp.cols; x++) {
            cv::Vec3s gradient = {0, 0, 0};

            for (int k = -1; k < 2; k++) {
                cv::Vec3s pixel = tmp.at<cv::Vec3s>(y + k, x);

                for (int c = 0; c < 3; c++) {
                    gradient[c] += pixel[c] * sobelVertical[k + 1];
                }
            }

            dst.at<cv::Vec3s>(y, x) = gradient;

        }
    }

    return 0; // Success
}

// Sobel filter for horizontal edges
int sobelY3x3( cv::Mat &src, cv::Mat &dst )
{
    // Define the sobel kernels
    int sobelVertical[3] = {-1, 0, 1};
    int sobelHorizontal[3] = {1, 2, 1};

    // Create tmp frame for intermediate step
    cv::Mat tmp = cv::Mat::zeros(src.size(), CV_16SC3);
    dst = cv::Mat::zeros(src.size(), CV_16SC3);

    // Vertical gradient
    for(int y=1; y<src.rows - 1; y++) {
        for(int x=0; x<src.cols; x++) {
            cv::Vec3s gradient = {0, 0, 0};

            // Apply the filter to the pixel
            for (int k = -1; k < 2; k++) {
                cv::Vec3b pixel = src.at<cv::Vec3b>(y + k, x);

                // Consider each color channel
                for (int c = 0; c < 3; c++) {
                    gradient[c] += pixel[c] * sobelVertical[k + 1];
                }
            }

            // Set the pixel value to the gradient
            tmp.at<cv::Vec3s>(y, x) = gradient;

        }
    }

    // Horizontal gradient
    for(int y=0; y<tmp.rows; y++) {
        for(int x=1; x<tmp.cols - 1; x++) {
            cv::Vec3s gradient = {0, 0, 0};

            // Apply the filter to the pixel
            for (int k = -1; k < 2; k++) {
                cv::Vec3s pixel = tmp.at<cv::Vec3s>(y, x + k);

                // Consider each color channel
                for (int c = 0; c < 3; c++) {
                    gradient[c] += pixel[c] * sobelHorizontal[k + 1];
                }
            }

            // Set the pixel value to the gradient
            dst.at<cv::Vec3s>(y, x) = gradient;

        }
    }

    return 0; // Success
}

// Magnitude filter from Sobel output
int magnitude( cv::Mat &sx, cv::Mat &sy, cv::Mat &dst )
{
    // Create tmp frame for intermediate step
    cv::Mat tmp = cv::Mat::zeros(sx.size(), CV_16SC3);

    // Iterates over each pixel in the frame
    for(int y=0; y<sx.rows; y++) {
        for(int x=0; x<sx.cols; x++) {
            cv::Vec3s gradient = {0, 0, 0};

            // Iterates over color channels
            for (int c = 0; c < 3; c++) {
                // Calculates the magnitude of the gradient
                gradient[c] = sqrt(pow(sx.at<cv::Vec3s>(y, x)[c], 2) + pow(sy.at<cv::Vec3s>(y, x)[c], 2));
            }

            // Sets the pixel value to the gradient
            tmp.at<cv::Vec3s>(y, x) = gradient;
        }
    }

    // Converts the frame to 8-bit
    cv::convertScaleAbs(tmp, dst);

    return 0; // Success
}

// Blur and quantize filter
int blurQuantize( cv::Mat &src, cv::Mat &dst, int levels )
{
    // Creates temporary frame for blurring step
    cv::Mat tmp = src.clone();
    blur5x5_2(src, tmp);

    // Calculate bucket size
    float bucket = 255 / levels;

    // Iterate over each pixel in the frame
    for(int i=0;i<tmp.rows;i++) {
        for(int j=0;j<tmp.cols;j++) {
            // Iterate over color channels
            for (int c = 0; c < 3; c++) {
                // Quantizes the pixel value by dividing by the bucket size, flooring the result, and multiplying by the bucket size
                dst.at<cv::Vec3b>(i, j)[c] =
                    static_cast<uchar>(std::floor(tmp.at<cv::Vec3b>(i, j)[c] / bucket) * bucket);
            }
        }
    }

    return 0;
}

// Detect faces
int faceFinder(cv::Mat& src, cv::Mat& dst)
{
    std::vector<cv::Rect> faces;
    cv::Mat grey;

    cv::cvtColor(src, grey, cv::COLOR_BGR2GRAY);
    detectFaces(grey, faces);
    drawBoxes(dst, faces);

    return 0;
}

// Redact/blur faces
int faceRedactor(cv::Mat& src, cv::Mat& dst)
{
    std::vector<cv::Rect> faces;
    cv::Mat grey;
    int minWidth = 50;
    float scale = 1.0;

    cv::cvtColor(src, grey, cv::COLOR_BGR2GRAY);
    detectFaces(grey, faces);
    
    // The color to draw the rectangle (B, G, R)
    cv::Scalar wcolor(170, 120, 110);

    for (int i = 0; i < faces.size(); i++) {
        if (faces[i].width > minWidth) {
            // Scale the face rectangle
            cv::Rect face(faces[i]);
            face.x = static_cast<int>(face.x * scale);
            face.y = static_cast<int>(face.y * scale);
            face.width = static_cast<int>(face.width * scale);
            face.height = static_cast<int>(face.height * scale);

            // Ensure the face rectangle stays within image bounds
            face &= cv::Rect(0, 0, src.cols, src.rows);

            // Apply a heavy blur to the face region
            cv::Mat faceROI = dst(face);
            cv::GaussianBlur(faceROI, faceROI, cv::Size(51, 51), 0);

            // Draw a rectangle around the blurred face
            cv::rectangle(dst, face, wcolor, 3);
        }
    }

    return 0;
}

// Custom filter using depth values to blur background
int backgroundBlur(cv::Mat &src, cv::Mat &dst, DA2Network &da_net)
{
    cv::Mat depth_map;
    float reduction = 0.5;

    // Input scaling logic from Bruce Maxwell's example
    float scale_factor = 256.0 / (src.rows > src.cols ? src.cols : src.rows);
    scale_factor = scale_factor > 1.0 ? 1.0 : scale_factor;

    // Resize the input image
    cv::resize(src, src, cv::Size(), reduction, reduction);

    // Run the DA2 network
    da_net.set_input(src, scale_factor);
    da_net.run_network(depth_map, src.size());

    // Normalize depth map to [0, 255]
    cv::normalize(depth_map, depth_map, 0, 255, cv::NORM_MINMAX);
    // Convert depth map to 8-bit
    depth_map.convertTo(depth_map, CV_8U);

    // Apply a blur
    int kernel_size = 5;
    int quantization_levels = 8;  // Number of quantization levels

    // Smooth the depth map with a kernel
    cv::Mat smoothed_depth_map;
    cv::blur(depth_map, smoothed_depth_map, cv::Size(kernel_size, kernel_size));

    // Quantize the depth map
    int step = 256 / quantization_levels; // Step size for quantization
    smoothed_depth_map = (smoothed_depth_map / step) * step;

    // Create a mask for regions below the threshold
    float depth_threshold = 128;
    cv::Mat below_threshold_mask;
    cv::threshold(smoothed_depth_map, below_threshold_mask, depth_threshold, 255, cv::THRESH_BINARY_INV);

    // Create a blurred version of the input image
    cv::Mat blurred_src;
    cv::GaussianBlur(src, blurred_src, cv::Size(15, 15), 0);

    // Use the below-threshold mask to combine blurred and original regions
    cv::Mat blurred_region, sharp_region;
    cv::bitwise_and(blurred_src, blurred_src, blurred_region, below_threshold_mask);
    cv::bitwise_and(src, src, sharp_region, ~below_threshold_mask);

    // Combine the blurred and sharp regions
    cv::add(blurred_region, sharp_region, dst);

    return 0;
}

int colorHighlightFilter(cv::Mat &src, cv::Mat &dst) 
{

    // Convert the input image to HSV color space
    cv::Scalar lowerBound(0, 0, 70);   // Lower bound for red in HSV
    cv::Scalar upperBound(0, 0, 255); // Upper bound for red in HSV

    cv::Mat hsv;
    cv::cvtColor(src, hsv, cv::COLOR_BGR2HSV);

    // Create a mask for the selected color range
    cv::Mat mask;
    cv::inRange(hsv, lowerBound, upperBound, mask);

    // Convert the original image to grayscale
    cv::Mat gray;
    cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);

    // Convert grayscale image back to 3-channel for masking
    cv::Mat grayBGR;
    cv::cvtColor(gray, grayBGR, cv::COLOR_GRAY2BGR);

    // Copy the original color where the mask is non-zero, grayscale elsewhere
    dst = grayBGR.clone();
    src.copyTo(dst, mask);

	return 0;
}

int medianFilter(cv::Mat &src, cv::Mat &dst) 
{
    // Ensure the input is a grayscale image
    CV_Assert(src.channels() == 1);

    // Initialize the output image
    dst = src.clone();

    int kernelSize = 3; // Median filter kernel size
    int halfKernel = kernelSize / 2;

    // Iterate through the image, excluding borders
    for (int y = halfKernel; y < src.rows - halfKernel; y++) {
        for (int x = halfKernel; x < src.cols - halfKernel; x++) {
            // Extract the kernel window
            std::vector<uchar> window;
            for (int ky = -halfKernel; ky <= halfKernel; ky++) {
                for (int kx = -halfKernel; kx <= halfKernel; kx++) {
                    window.push_back(src.at<uchar>(y + ky, x + kx));
                }
            }

            // Find the median of the window
            std::nth_element(window.begin(), window.begin() + window.size() / 2, window.end());
            dst.at<uchar>(y, x) = window[window.size() / 2];
        }
    }
    return 0;
}

int coolToneFilter(cv::Mat &src, cv::Mat &dst) {
    // Split the image into its BGR channels
    std::vector<cv::Mat> channels;
    cv::split(src, channels);

    // Boost the blue channel and reduce the red channel slightly
    channels[0] = cv::min(channels[0] + 60, 255); // Blue
    channels[2] = cv::max(channels[2] - 20  , 0);   // Red

    // Merge back the channels
    cv::merge(channels, dst);

    return 0;
}

int warmToneFilter(cv::Mat &src, cv::Mat &dst) {
    // Split the image into its BGR channels
    std::vector<cv::Mat> channels;
    cv::split(src, channels);

    // Boost red and green channels slightly
    channels[2] = cv::min(channels[2] + 60, 255); // Red
    //channels[1] = cv::min(channels[1] + 15, 255); // Green

    // Merge back the channels
    cv::merge(channels, dst);
    
    return 0;
}

int negativeFilter(cv::Mat &src, cv::Mat &dst) {
    // Ensure the destination has the same size and type as the source
    dst = cv::Mat::zeros(src.size(), src.type());

    // Invert the pixel values
    dst = cv::Scalar::all(255) - src;

	return 0;
}

int embossFilter(cv::Mat &src, cv::Mat &dst) 
{
    for (int i = 2;i < src.rows-1;i++) 
    {
        for (int j = 2;j < src.cols-1;j++) 
        {
            // blue channel
            int c = 0;
            // Apply 3x3 emboss kernel
            dst.at<cv::Vec3b>(i, j)[c] = src.at<cv::Vec3b>(i - 1, j - 1)[c] * (-2) + src.at<cv::Vec3b>(i - 1, j)[c] * (-1)
                + src.at<cv::Vec3b>(i, j - 1)[c] * (-1) + src.at<cv::Vec3b>(i, j)[c] + src.at<cv::Vec3b>(i, j + 1)[c]
                + src.at<cv::Vec3b>(i + 1, j)[c] + src.at<cv::Vec3b>(i + 1, j + 1)[c] * 2;
                
            // green channel
            c = 1;
            // Apply 3x3 emboss kernel
            dst.at<cv::Vec3b>(i, j)[c] = src.at<cv::Vec3b>(i - 1, j - 1)[c] * (-2) + src.at<cv::Vec3b>(i - 1, j)[c] * (-1)
                + src.at<cv::Vec3b>(i, j - 1)[c] * (-1) + src.at<cv::Vec3b>(i, j)[c] + src.at<cv::Vec3b>(i, j + 1)[c]
                + src.at<cv::Vec3b>(i + 1, j)[c] + src.at<cv::Vec3b>(i + 1, j + 1)[c] * 2;

            // red channel  
            c = 2;
            // Apply 3x3 emboss kernel
            dst.at<cv::Vec3b>(i, j)[c] = src.at<cv::Vec3b>(i - 1, j - 1)[c] * (-2) + src.at<cv::Vec3b>(i - 1, j)[c] * (-1)
                + src.at<cv::Vec3b>(i, j - 1)[c] * (-1) + src.at<cv::Vec3b>(i, j)[c] + src.at<cv::Vec3b>(i, j + 1)[c]
                + src.at<cv::Vec3b>(i + 1, j)[c] + src.at<cv::Vec3b>(i + 1, j + 1)[c] * 2;
        }
    }
    return 0;
}





