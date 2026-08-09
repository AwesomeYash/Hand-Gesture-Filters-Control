// Authors: Danny Chinn, Priyanshu Ranka

#ifndef FILTERS_H
#define FILTERS_H

#include <opencv2/opencv.hpp>
#include "DA2Network.hpp"

int defaultGreyscale( cv::Mat& src, cv::Mat& dst );

int greyscale( cv::Mat& src, cv::Mat& dst );

int filterSepia( cv::Mat& src, cv::Mat& dst );

int blur5x5_1( cv::Mat &src, cv::Mat &dst );

int blur5x5_2( cv::Mat& src, cv::Mat& dst );

int blur5x5_3(cv::Mat& src, cv::Mat& dst);

int sobelY3x3( cv::Mat &src, cv::Mat &dst );

int sobelX3x3( cv::Mat &src, cv::Mat &dst );

int magnitude( cv::Mat &sx, cv::Mat &sy, cv::Mat &dst );

int blurQuantize( cv::Mat &src, cv::Mat &dst, int levels );

int faceFinder(cv::Mat& src, cv::Mat& dst);

int faceRedactor(cv::Mat& src, cv::Mat& dst);

int backgroundBlur( cv::Mat &src, cv::Mat &dst, DA2Network &da_net );

int colorHighlightFilter(cv::Mat& src, cv::Mat& dst);

int medianFilter(cv::Mat& src, cv::Mat& dst);

int negativeFilter(cv::Mat& src, cv::Mat& dst);

int warmToneFilter(cv::Mat& src, cv::Mat& dst);

int coolToneFilter(cv::Mat& src, cv::Mat& dst);

int embossFilter(cv::Mat& src, cv::Mat& dst);

#endif
