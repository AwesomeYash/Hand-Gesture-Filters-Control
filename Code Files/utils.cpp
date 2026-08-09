// Authors: Danny Chinn, Priyanshu Ranka

#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include "filters.h"
#include "tinyfiledialogs.h"
#include <cstdio>

using namespace std;

// Load a frame/image using tinyfiledialogs explorer
std::string loadFrame()
{
    const char* filter_patterns[] = { "*.jpg", "*.png", "*.jpeg" };
    const char* load_path = tinyfd_openFileDialog(
        "Select an Image", "", 3, filter_patterns, "Image Files (jpg, png)", 0);
    if (!load_path) {
        std::cerr << "Error: No file selected!" << std::endl;
        return "";
    }
    return load_path;
}


// Save a frame/image using tinyfiledialogs explorer
std::string saveFrame(cv::Mat& frame)
{
    // Define filter patterns
    const char* filter_patterns[] = { "*.jpg", "*.png", "*.avi" };

    const char* save_path = tinyfd_saveFileDialog(
                "Save media", "file_name.jpg", 3, filter_patterns, "Image Files (jpg, png), Video Files (avi)");

    // Check if the user canceled the dialog
    if (!save_path) {
        std::cerr << "Save operation canceled by the user." << std::endl;
        return ""; // Return an empty string to indicate no file was saved
    }

    // Save the image to the selected path
    if (cv::imwrite(save_path, frame)) {
        std::cout << "Image saved as: " << save_path << std::endl;
    } else {
        std::cerr << "Error: Could not save the image!" << std::endl;
    }

    return save_path; // Return the save path or empty string
}


// Calculate the magnitude of the gradient
int magnitudeHelper(cv::Mat& src, cv::Mat& dst)
{
    cv::Mat sx, sy;
    sobelX3x3(src, sx);
    sobelY3x3(src, sy);
    magnitude(sx, sy, dst);
    return 0;
}

// Uses openCV video writer to record video, tinyfile to save
bool recorderHelper(bool isRecording, cv::VideoWriter &videoWriter, cv::Mat &frame)
{
    string temp_filename = "temp.avi";

    int fps = 10;
    cv::Size frame_size = cv::Size(frame.cols, frame.rows);

    // Ensure frame has 3 channels for VideoWriter
    cv::Mat outputFrame;
    if (frame.channels() == 1) {
        // Convert grayscale to BGR
        cv::cvtColor(frame, outputFrame, cv::COLOR_GRAY2BGR);
    } else {
        // Use the original frame if already in BGR
        outputFrame = frame;
    }

    // Check if the user wants to start/stop recording
    if (isRecording && videoWriter.isOpened()) {
        // Write the frame to the video
        videoWriter.write(frame);
    }
    
    else if (isRecording && !videoWriter.isOpened()) {
        videoWriter.open(temp_filename, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), fps, frame_size, true);
        if (!videoWriter.isOpened()) {
            std::cerr << "Error: Could not open the video writer!" << std::endl;
            return false;
        }
        videoWriter.write(frame);
    }

    else if (!isRecording && videoWriter.isOpened()) {
        // Release the video writer
        videoWriter.release();
        std::cout << "Recording stopped." << std::endl;
        // Prompt user to save the file
        const char* filter_patterns[] = { "*.avi" };
        const char* save_path = tinyfd_saveFileDialog(
            "Save Video", "recorded_video.avi", 1, filter_patterns, "AVI Video Files (*.avi)");
        
        if (save_path) {
            // Rename temporary file to user-specified path
            if (std::rename(temp_filename.c_str(), save_path) == 0) {
                cout << "Video saved as: " << save_path << endl;
            } else {
                cerr << "Error: Could not save the video to the selected location!" << endl;
            }
        } else {
            // User canceled save dialog
            cout << "Save operation canceled. Temporary file retained: " << temp_filename << endl;
        }
    }

    return isRecording;
}