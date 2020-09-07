#include <iostream>
#include <numeric>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>

using namespace std;

void descKeypoints1()
{
    // load image from file and convert to grayscale
    cv::Mat imgGray;
    cv::Mat img = cv::imread("../images/img1.png");
    cv::cvtColor(img, imgGray, cv::COLOR_BGR2GRAY);

    // BRISK detector / descriptor
    cv::Ptr<cv::FeatureDetector> detector = cv::BRISK::create();
    vector<cv::KeyPoint> kptsBRISK;

    double t = (double)cv::getTickCount();
    detector->detect(imgGray, kptsBRISK);
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << "BRISK detector with n= " << kptsBRISK.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;

    cv::Ptr<cv::DescriptorExtractor> descriptor = cv::BRISK::create();
    cv::Mat descBRISK;
    t = (double)cv::getTickCount();
    descriptor->compute(imgGray, kptsBRISK, descBRISK);
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << "BRISK descriptor in " << 1000 * t / 1.0 << " ms" << endl;

    // visualize results
    cv::Mat visImage = img.clone();
    cv::drawKeypoints(img, kptsBRISK, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    string windowName = "BRISK Results";
    cv::namedWindow(windowName, 1);
    imshow(windowName, visImage);
    cv::waitKey(0);

    // TODO: Add the SIFT detector / descriptor, compute the 
    // time for both steps and compare both BRISK and SIFT
    // with regard to processing speed and the number and 
    // visual appearance of keypoints.

    //----answer

    cv::Ptr<cv::FeatureDetector> detector_sift = cv::SIFT::create();
    vector<cv::KeyPoint> SIFTKy_list;

    double t_2 = (double)cv::getTickCount();
    detector_sift -> detect(imgGray, SIFTKy_list);
    t_2 = ((double)cv::getTickCount() - t_2)/ cv::getTickFrequency();
    cout << "SIFT detector in " << 1000 * t_2 / 1.0 << " ms" << endl;

    cv::Ptr<cv::DescriptorExtractor> descriptor_sift = cv::SIFT::create();
    cv::Mat descSIFT;
    t_2 = ((double)cv::getTickCount() - t_2) cv::getTickFrequency();
    descriptor_sift -> compute(imgGray, SIFTKy_list, descSIFT);
    t_2 = ((double)cv::getTickCount() - t_2)/ cv::getTickFrequency();
    cout << "SIFT descriptor in " << 1000 * t_2 / 1.0 << " ms" << endl;
    cv::Mat visImage2 = img.clone();
    cv::drawkeyPoints(img, SIFTKy_list,visImage2,cv::Scalar::all(-1),cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    string windowName2 = "SIFT Results";
    cv::namedWindow(windowName2, 1);
    imshow(windowName2, visImage2);
    cv::waitKey(0);
   
}

int main()
{
    descKeypoints1();
    return 0;
}