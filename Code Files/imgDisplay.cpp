// Authors: Danny Chinn, Priyanshu Ranka

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include "filters.h"
#include "utils.h"

#include <iostream>

using namespace cv;

int main()
{
    std::string image_path = loadFrame();
    Mat img = imread(image_path, IMREAD_COLOR);

    if(img.empty())
    {
        std::cout << "Could not read the image: " << image_path << std::endl;
        return 1;
    }

    // store last key press for filter determination
    int lastKeyPressed = 0;

    // initialize DA2 Network
    DA2Network da_net("model_fp16.onnx");

    // Create clones for filter/undo functionality
    cv::Mat tmp = img.clone();
    cv::Mat filteredFrame = img.clone();

    // Loop to wait for input
    for (;;)
    {
        // Apply previously selected filters
        switch(lastKeyPressed) {
            case 'g':
                defaultGreyscale(img, filteredFrame);
                break;
            case 'h':
                greyscale(img, filteredFrame);
                break;
            case 'o':
                filterSepia(img, filteredFrame);
                break;
            case 'u':
                blur5x5_1(img, filteredFrame);
                break;
            case 'b':
                blur5x5_2(img, filteredFrame);
                break;
            case 'x':
                sobelX3x3(img, filteredFrame);
                cv::convertScaleAbs(filteredFrame, filteredFrame);
                break;
            case 'y':
                sobelY3x3(img, filteredFrame);
                cv::convertScaleAbs(filteredFrame, filteredFrame);
                break;
            case 'm': 
                magnitudeHelper(img, filteredFrame);
                break;
            case 'l':
                blurQuantize(img, filteredFrame, 10);
                break;
            case 'f':
                faceFinder(img, filteredFrame);
            case 'd':
                backgroundBlur(img, filteredFrame, da_net);
                break;
            case '1':
                negativeFilter(img, filteredFrame);
                break;
            case '2':
                embossFilter(img, filteredFrame);
                break;
            case '3':
                faceRedactor(img, filteredFrame);
                break;
            case '4':
                coolToneFilter(img, filteredFrame);
                break;
            case '5':
                warmToneFilter(img, filteredFrame);
                break;
            case '6':
                blur5x5_3(img, filteredFrame);
            case '7': {
                cv::Mat grey;
                cv::cvtColor(img, grey, cv::COLOR_BGR2GRAY);
                medianFilter(grey, filteredFrame);
                cv::cvtColor(filteredFrame, filteredFrame, cv::COLOR_GRAY2BGR);
                break;
            }
            default:
                break;
        }

        // Set image to filtered image
        img = filteredFrame;

        imshow("Display window", img);
        int keyPress = waitKey(0); // Wait for a keystroke in the window

        // Handle key presses
        switch(keyPress) {
            // Quit and close
            case 'q':
                destroyAllWindows();
                return 0;
            // Save image
            case 's':
                saveFrame(img);
                lastKeyPressed = 0;
                break;
            // Load a different image
            case 'n': {
                // Call tinyfiledialog to load new image
                std::string new_image_path = loadFrame();
                
                // Check for cancelled dialog/empty path
                if (new_image_path.empty()) {
                    std::cerr << "No file selected. Canceling load operation." << std::endl;
                    break;
                }
                
                // Load new image
                Mat new_img = imread(new_image_path, IMREAD_COLOR);

                // Ensure new image not empty
                if (new_img.empty()) {
                    std::cerr << "Could not read the image: " << new_image_path << std::endl;
                } else {
                    // Set new image, clones, and reset filters
                    img = new_img.clone();
                    tmp = img.clone();
                    filteredFrame = img.clone();
                }
                break;
            }
            // Set filter
            default:
                // Sets last valid key pressed for filter selection
                if (lastKeyPressed == keyPress) {
                    lastKeyPressed = 0;
                    filteredFrame = tmp.clone();
                } else if (keyPress != -1) {
                    lastKeyPressed = keyPress;
                }
                break;
        }
    }

    return 0;
}