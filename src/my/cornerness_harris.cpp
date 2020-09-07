#include <iostream>
#include <numeric>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>

using namespace std;

void cornernessHarris()
{
    // load image from file
    cv::Mat img;
    img = cv::imread("../images/img1.png");
    cv::cvtColor(img, img, cv::COLOR_BGR2GRAY); // convert to grayscale

    // Detector parameters
    int blockSize = 2;     // for every pixel, a blockSize × blockSize neighborhood is considered
    int apertureSize = 3;  // aperture parameter for Sobel operator (must be odd)
    int minResponse = 100; // minimum value for a corner in the 8bit scaled response matrix
    double k = 0.04;       // Harris parameter (see equation for details)

    // Detect Harris corners and normalize output
    cv::Mat dst, dst_norm, dst_norm_scaled;
    dst = cv::Mat::zeros(img.size(), CV_32FC1);
    cv::cornerHarris(img, dst, blockSize, apertureSize, k, cv::BORDER_DEFAULT);
    cv::normalize(dst, dst_norm, 0, 255, cv::NORM_MINMAX, CV_32FC1, cv::Mat());
    cv::convertScaleAbs(dst_norm, dst_norm_scaled);

    // visualize results
    string windowName = "Harris Corner Detector Response Matrix";
    cv::namedWindow(windowName, 4);
    cv::imshow(windowName, dst_norm_scaled);
    cv::waitKey(0);

    // TODO: Your task is to locate local maxima in the Harris response matrix 
    // and perform a non-maximum suppression (NMS) in a local neighborhood around 
    // each maximum. The resulting coordinates shall be stored in a list of keypoints 
    // of the type `vector<cv::KeyPoint>`

  //  ---- CODE WITH ASSIGNMENT---
    vector<cv::KeyPoint> keypts;
    double maxov = 0.0;
    for(size_t i = 0; i<dst_norm.rows;i++)
    {
        for(size_t j=0; j<dst_norm.cols;j++)
        {
            int value_response = (int)dst_norm.at<float>(i,j);
            if(value_response > minResponse)
            {
                cv::KeyPoint newkeypts;
                newkeypts.pt = cv::Point2f(j,i); //cordinate
                newkeypts.response = value_response; //the magnitude at that coordinate
                newkeypts.size = 2*apertureSize;
                bool flag = false;
                for(auto itr=keypts.begin();itr!=keypts.end(); ++itr)
                {
                    
                    double overlapkey = cv::KeyPoint::overlap(newkeypts,*itr);
                    if(overlapkey>maxov)
                    { 
                        flag = true;
                        if(newkeypts.response > (*itr).response)
                        {
                            *itr= newkeypts;
                            break;
                        }
                    }
                }
                if(!flag)
                {
                    keypts.push_back(newkeypts);                   
                }
                
            }
        }
    }

    windowName = "Harris Corner Detection Results";
    cv::namedWindow(windowName, 5);
    cv::Mat visImage = dst_norm_scaled.clone();
    cv::drawKeypoints(dst_norm_scaled, keypts, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    cv::imshow(windowName, visImage);
    cv::waitKey(0);


// ------ WRONG APPROACH --
//     vector<cv::KeyPoint> final_points;
//     for(int i = 0; i<dst_norm_scaled.rows;i++)
//     {
//         for(int j=0; j<dst_norm_scaled.column;i++)
//         {
//             if dst_norm_scaled(i,j) >= 100;
//                 if(norm_scaled(i+1,j)<100 && norm_scaled(i,j+1)<100 && norm_scaled(i-1,j)<100 && norm_scaled(i,j-1)<100)
//                     final_points.push_back(dst_norm_scaled(i,j));
//         }
//     }


//-=-=--------- SOLUTION----
//  vector<cv::KeyPoint> keypoints;
//     double maxOverlap = 0.0; // max. permissible overlap between two features in %, used during non-maxima suppression
//     for (size_t j = 0; j < dst_norm.rows; j++)
//     {
//         for (size_t i = 0; i < dst_norm.cols; i++)
//         {
//             int response = (int)dst_norm.at<float>(j, i);
//             if (response > minResponse)
//             { // only store points above a threshold

//                 cv::KeyPoint newKeyPoint;
//                 newKeyPoint.pt = cv::Point2f(i, j);
//                 newKeyPoint.size = 2 * apertureSize;
//                 newKeyPoint.response = response;

//                 // perform non-maximum suppression (NMS) in local neighbourhood around new key point
//                 bool bOverlap = false;
//                 for (auto it = keypoints.begin(); it != keypoints.end(); ++it)
//                 {
//                     double kptOverlap = cv::KeyPoint::overlap(newKeyPoint, *it);
//                     if (kptOverlap > maxOverlap)
//                     {
//                         bOverlap = true;
//                         if (newKeyPoint.response > (*it).response)
//                         {                      // if overlap is >t AND response is higher for new kpt
//                             *it = newKeyPoint; // replace old key point with new one
//                             break;             // quit loop over keypoints
//                         }
//                     }
//                 }
//                 if (!bOverlap)
//                 {                                     // only add new key point if no overlap has been found in previous NMS
//                     keypoints.push_back(newKeyPoint); // store new keypoint in dynamic list
//                 }
//             }
//         } // eof loop over cols
//     }     // eof loop over rows

//     // visualize keypoints
//     windowName = "Harris Corner Detection Results";
//     cv::namedWindow(windowName, 5);
//     cv::Mat visImage = dst_norm_scaled.clone();
//     cv::drawKeypoints(dst_norm_scaled, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
//     cv::imshow(windowName, visImage);
//     cv::waitKey(0);



}

int main()
{
    cornernessHarris();
}