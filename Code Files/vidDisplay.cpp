// Authors: Danny Chinn, Priyanshu Ranka

#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <stdio.h>
#include "filters.h"
#include "utils.h"
#include "faceDetect.h"

using namespace cv;
using namespace std;

int main(int, char**)
{
    cv::Mat frame;
    // [From OpenCV] --- INITIALIZE VIDEOCAPTURE
    cv::VideoCapture cap;
    // [From OpenCV] open the default camera using default API
    
    cap.open(0);
    // [From OpenCV] check if we succeeded
    if (!cap.isOpened()) {
        cerr << "ERROR! Unable to open camera\n";
        return -1;
    }

    // [From OpenCV] --- GRAB AND WRITE LOOP
    std::cout << "Start grabbing" << endl;

    // State variables
    int frameNumber = 0;
    int lastKeyPressed = 0;
    bool isRecording = false;

    // Initialize DA2 Network
    DA2Network da_net("model_fp16.onnx");

    // Initialize video writer
    cv::VideoWriter videoWriter;

    for (;;)
    {
        //  [From OpenCV] wait for a new frame from camera and store it into 'frame'
        cap.read(frame);
        //  [From OpenCV] check if we succeeded
        if (frame.empty()) {
            cerr << "ERROR! blank frame grabbed\n";
            break;
        }

        cv::Mat filteredFrame = frame.clone();

        // apply previously selected filters
        switch(lastKeyPressed) {
            case 'g':
                defaultGreyscale(frame, filteredFrame);
                break;
            case 'h':
                greyscale(frame, filteredFrame);
                break;
            case 'o':
                filterSepia(frame, filteredFrame);
                break;
            case 'u':
                blur5x5_1(frame, filteredFrame);
                break;
            case 'b':
                blur5x5_2(frame, filteredFrame);
                break;
            case 'x':
                sobelX3x3(frame, filteredFrame);
                cv::convertScaleAbs(filteredFrame, filteredFrame);
                break;
            case 'y':
                sobelY3x3(frame, filteredFrame);
                cv::convertScaleAbs(filteredFrame, filteredFrame);
                break;
            case 'm': 
                magnitudeHelper(frame, filteredFrame);
                break;
            case 'l':
                blurQuantize(frame, filteredFrame, 10);
                break;
            case 'f':
                faceFinder(frame, filteredFrame);
                break;
            case 'd':
                backgroundBlur(frame, filteredFrame, da_net);
                break;
            case '1':
                negativeFilter(frame, filteredFrame);
                break;
            case '2':
                embossFilter(frame, filteredFrame);
                break;
            case '3':
                faceRedactor(frame, filteredFrame);
                break;
            case '4':
                coolToneFilter(frame, filteredFrame);
                break;
            case '5':
                warmToneFilter(frame, filteredFrame);
                break;
            case '6':
                blur5x5_3(frame, filteredFrame);
            case '7': {
                cv::Mat grey;
                cv::cvtColor(frame, grey, cv::COLOR_BGR2GRAY);
                medianFilter(grey, filteredFrame);
                cv::cvtColor(filteredFrame, filteredFrame, cv::COLOR_GRAY2BGR);
                break;
            }
            default:
                break;
        }

        frame = filteredFrame;

        // [From OpenCV] show live and wait for a key with timeout long enough to show images
        cv::imshow("Live", frame);

        // Check if recording flag is set
        if (isRecording) {
            // Write the frame to the video
            videoWriter.write(frame);
        }

        int keyPress = cv::waitKey(10);
        // quit, save, or set filter
        switch(keyPress) {
            case 'q':
                cap.release();
                destroyAllWindows();
                return 0;
            case 's':
                saveFrame(frame);
                break;
            case 'r':
                // Flip boolean
                isRecording = !isRecording;
                // Call helper function to facilitate recording
                recorderHelper(isRecording, videoWriter, frame);
                break;
            default:
                // checks for repeated key press to turn off filter
                if (lastKeyPressed == keyPress) {
                    lastKeyPressed = 0;
                }
                else {
                    // checks for valid keypress to store as lastKeyPressed
                    lastKeyPressed = (keyPress != -1) ? keyPress : lastKeyPressed;
                }
                break;
        }

        // increment frame number
        frameNumber++;
    }
    // [From OpenCV] the camera will be deinitialized automatically in VideoCapture destructor
    return 0;
}