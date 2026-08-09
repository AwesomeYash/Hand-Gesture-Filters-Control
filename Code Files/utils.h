// Authors: Danny Chinn, Priyanshu Ranka

#ifndef UTILS_H
#define UTILS_H

#include <opencv2/opencv.hpp>
#include <stdio.h>

using namespace std;

// helper function to save a frame/image
std::string saveFrame(cv::Mat& frame);

// helper function to load a frame/image
std::string loadFrame();

// helper function to calculate the magnitude of the gradient
int magnitudeHelper(cv::Mat& src, cv::Mat& dst);

bool recorderHelper(bool isRecording, cv::VideoWriter &videoWriter, cv::Mat &frame);

#endif