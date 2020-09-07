#include <numeric>
#include "matching2D.hpp"
/* INCLUDES FOR THIS PROJECT */
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <limits>
#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>

#include "dataStructures.h"
//#include "matching2D.hpp"


using namespace std;

// Find best matches for keypoints in two camera images based on several matching methods
void matchDescriptors(std::vector<cv::KeyPoint> &kPtsSource, std::vector<cv::KeyPoint> &kPtsRef, cv::Mat &descSource, cv::Mat &descRef,
                      std::vector<cv::DMatch> &matches, std::string descriptorType, std::string matcherType, std::string selectorType)
{
    // configure matcher
    bool crossCheck = false;
    cv::Ptr<cv::DescriptorMatcher> matcher;
    //BY ME--new vector definitions-----
    vector<vector<cv::DMatch>> matches_knn;
    vector<cv::DMatch> good_matches;
    int k=2;
    //----------

    if (matcherType.compare("MAT_BF") == 0)
    {
        //int normType = cv::NORM_HAMMING;
        int normType = descriptorType.compare("DES_BINARY") == 0 ? cv::NORM_HAMMING : cv::NORM_L2;
        matcher = cv::BFMatcher::create(normType, crossCheck);
        cout<<"BFF matching";
    }
    else if (matcherType.compare("MAT_FLANN") == 0)
    {
        if(descSource.type()!=CV_32F)
        {
            descSource.convertTo(descSource, CV_32F);
            descRef.convertTo(descRef, CV_32F);
        }
        int normType = descriptorType.compare("DES_BINARY") == 0 ? cv::NORM_HAMMING : cv::NORM_L2;
        matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
        cout<<"FLANN matching"<<endl;

    }

    // perform matching task
    if (selectorType.compare("SEL_NN") == 0)
    { // nearest neighbor (best match)
        matcher->match(descSource, descRef, matches); // Finds the best match for each descriptor in desc1
    }
    else if (selectorType.compare("SEL_KNN") == 0)
    { // k nearest neighbors (k=2)
        matcher->knnMatch(descSource,descRef,matches_knn,k);
    }

    if (selectorType.compare("SEL_KNN") == 0)
    {
        for(int i= 0;i<matches_knn[0].size();i++)
        {
            if(matches_knn[0][i].distance/matches_knn[1][i].distance < 0.8)
            {
                good_matches.push_back(matches_knn[0][1]);
            }
        }
    matches = good_matches;
    }
    // else if (selectorType.compare("SEL_NN") == 0)
    // {
    //     for(int i= 0;i<matches[0].size();i++)
    //     {
    //         if(matches[0][i].distance/matches[1][i].distance < 0.8)
    //         {
    //             good_matches.push_back(matches[0][1]);
    //         }
    //     }

    // }
    
}

// Use one of several types of state-of-art descriptors to uniquely identify keypoints
void descKeypoints(vector<cv::KeyPoint> &keypoints, cv::Mat &img, cv::Mat &descriptors, string descriptorType)
{
    // select appropriate descriptor
    cv::Ptr<cv::DescriptorExtractor> extractor;
    cv::Ptr<cv::AKAZE> akaze_extractor = cv::AKAZE::create();
    if (descriptorType.compare("BRISK") == 0)
    {

        int threshold = 30;        // FAST/AGAST detection threshold score.
        int octaves = 3;           // detection octaves (use 0 to do single scale)
        float patternScale = 1.0f; // apply this scale to the pattern used for sampling the neighbourhood of a keypoint.

        extractor = cv::BRISK::create(threshold, octaves, patternScale);
    }
     else if((descriptorType.compare("AKAZE") == 0))
    {
        akaze_extractor = cv::AKAZE::create();
        akaze_extractor -> detectAndCompute(img, cv::noArray(), keypoints, descriptors);
   
    }
    else if((descriptorType.compare("ORB") == 0))
    {
        extractor = cv::ORB::create();  
    }
    else if((descriptorType.compare("FREAK") == 0))
    {
        extractor = cv::xfeatures2d::FREAK::create();  
    }
    else if((descriptorType.compare("SIFT") == 0))
    {
        extractor = cv::xfeatures2d::SIFT::create();  
    }
      else if((descriptorType.compare("BRIEF") == 0))
    {
        extractor = cv::xfeatures2d::BriefDescriptorExtractor::create();  
    }
    if((descriptorType.compare("AKAZE") != 0))
    {
        double t = (double)cv::getTickCount();
       
        extractor->compute(img, keypoints, descriptors);
       
        t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
        cout<<descriptorType << " descriptor extraction in " << 1000 * t / 1.0 << " ms" << endl;
    }
}

// Detect keypoints in image using the traditional Shi-Thomasi detector
void detKeypointsShiTomasi(vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis)
{
    // compute detector parameters based on image size
    int blockSize = 4;       //  size of an average block for computing a derivative covariation matrix over each pixel neighborhood
    double maxOverlap = 0.0; // max. permissible overlap between two features in %
    double minDistance = (1.0 - maxOverlap) * blockSize;
    int maxCorners = img.rows * img.cols / max(1.0, minDistance); // max. num. of keypoints

    double qualityLevel = 0.01; // minimal accepted quality of image corners
    double k = 0.04;

    // Apply corner detection
    double t = (double)cv::getTickCount();
    vector<cv::Point2f> corners;
    cv::goodFeaturesToTrack(img, corners, maxCorners, qualityLevel, minDistance, cv::Mat(), blockSize, false, k);

    // add corners to result vector
    for (auto it = corners.begin(); it != corners.end(); ++it)
    {

        cv::KeyPoint newKeyPoint;
        newKeyPoint.pt = cv::Point2f((*it).x, (*it).y);
        newKeyPoint.size = blockSize;
        keypoints.push_back(newKeyPoint);
    }
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << "Shi-Tomasi detection with n=" << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;

    // visualize results
    if (bVis)
    {
        cv::Mat visImage = img.clone();
        cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        string windowName = "Shi-Tomasi Corner Detector Results";
        cv::namedWindow(windowName, 6);
        imshow(windowName, visImage);
        cv::waitKey(0);
    }
}

void detKeypointsHarris(vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis)
{
    int blockSize = 2;     // for every pixel, a blockSize × blockSize neighborhood is considered
    int apertureSize = 3;  // aperture parameter for Sobel operator (must be odd)
    int minResponse = 100; // minimum value for a corner in the 8bit scaled response matrix
    double k = 0.04;       // Harris parameter (see equation for details)
    
    cv::Mat dst, dst_norm, dst_norm_scaled;
    dst = cv::Mat::zeros(img.size(), CV_32FC1);
    cv::cornerHarris(img, dst, blockSize, apertureSize, k, cv::BORDER_DEFAULT);
    cv::normalize(dst, dst_norm, 0, 255, cv::NORM_MINMAX, CV_32FC1, cv::Mat());
    cv::convertScaleAbs(dst_norm, dst_norm_scaled);

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
                for(auto itr=keypoints.begin();itr!=keypoints.end(); ++itr)
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
                    keypoints.push_back(newkeypts);                   
                }
                
            }
        }
    }

      // visualize results
    if (bVis)
    {
        cv::Mat visImage = img.clone();
        cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        string windowName = "Harris Corner Detector Results";
        cv::namedWindow(windowName, 6);
        imshow(windowName, visImage);
        cv::waitKey(0);
    }
}

void detKeypointsFAST(vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis)
{
    int threshold = 30;
    bool NMS = true;
    cv::FastFeatureDetector::DetectorType type =  cv::FastFeatureDetector::TYPE_9_16;
    //cv::Ptr<cv::FeatureDetector> detector = cv::FastFeatureDetector::create(); even works without defining the type
    cv::Ptr<cv::FeatureDetector> detector = cv::FastFeatureDetector::create(threshold,NMS,type); 
    
    double t = (double)cv::getTickCount();
    detector->detect(img,keypoints);
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << "FAST detection with n=" << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;
    
        if (bVis)
    {
        cv::Mat visImage = img.clone();
        cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        string windowName = "FAST Corner Detector Results";
        cv::namedWindow(windowName, 6);
        imshow(windowName, visImage);
        cv::waitKey(0);
    }

    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << "FAST n=" << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;
    // cv::Mat FASTimage = img.clone();
    // cv::FAST(img, keypoints, 30, true);
    // cv::drawKeypoints(img,keypoints,FASTimage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

    // string windowName2 = "FAST corner detection 2006";
    // cv::namedWindow(windowName2,2);
    // imshow(windowName2, FASTimage);
    // cv::waitKey(0);
}

void detKeypointsBRISK(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis)
{

    cv::Ptr<cv::FeatureDetector> detector = cv::BRISK::create();    
    double t = (double)cv::getTickCount();
    detector->detect(img,keypoints);
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();

    if (bVis)
    {
        cv::Mat visImage = img.clone();
        cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        string windowName = "BRISK Detector Results";
        cv::namedWindow(windowName, 6);
        imshow(windowName, visImage);
        cv::waitKey(0);
    }

}

void detKeypointsSIFT(vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis)
{

    cv::Ptr<cv::FeatureDetector> detector = cv::xfeatures2d::SIFT::create();
    double t = (double)cv::getTickCount();
    detector->detect(img,keypoints);
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();

    if (bVis)
    {
        cv::Mat visImage = img.clone();
        cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        string windowName = "SIFT Detector Results";
        cv::namedWindow(windowName, 6);
        imshow(windowName, visImage);
        cv::waitKey(0);
    }

}

void detKeypointsAkaze(vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis)
{
    cv::Mat desc_akaze;
    cv::Ptr<cv::AKAZE> akaze = cv::AKAZE::create();
    double t = (double)cv::getTickCount();
    akaze -> detectAndCompute(img, cv::noArray(), keypoints, desc_akaze);
     t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
     if (bVis)
    {
        cv::Mat visImage = img.clone();
        cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        string windowName = "AKAZE Detector Results";
        cv::namedWindow(windowName, 6);
        imshow(windowName, visImage);
        cv::waitKey(0);
    }


}

void detKeypointsORB(vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis)
{
        cv::Mat desc_ORB;
        cv::Ptr<cv::ORB> orb = cv::ORB::create();
        orb->detectAndCompute(img, cv::noArray(),keypoints,desc_ORB);


     if (bVis)
    {
        cv::Mat visImage = img.clone();
        cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        string windowName = "ORB Detector Results";
        cv::namedWindow(windowName, 6);
        imshow(windowName, visImage);
        cv::waitKey(0);
    }
}


