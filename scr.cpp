#include<iostream>
#include<opencv2/opencv.hpp>
#include<vector>
#include<cmath>

bool checkvideo(cv::VideoCapture& cap,const std::string& videopath);
void conterdetection(cv::Mat& blueMask,std::vector<cv::RotatedRect>& lightBars,cv::Mat& outputFrame);
std::vector<cv::Point2f> extractArmorCorners(const cv::RotatedRect& leftBar, const cv::RotatedRect& rightBar);
void drawArmor(cv::Mat& frame, const std::vector<cv::Point2f>& corners, int armorIndex);

std::vector<cv::Point3f> POINT_3D_OF_ARMOR_SMALL = {
    cv::Point3f(0, 67.5, -27.5),  
    cv::Point3f(0, 67.5, 27.5),   
    cv::Point3f(0, -67.5, 27.5),  
    cv::Point3f(0, -67.5, -27.5)  
};


int main(){
  cv::Scalar blueLower(90, 80, 140);   
  cv::Scalar blueUpper(105, 255, 255);
  cv::VideoCapture cap;
  const std::string videopath="/home/dqx/c11/armor_detector/video/lv_0_20251121230350.mp4";

  if (!checkvideo(cap,videopath))  return -1;
    

  cv::Mat frame;
  while (cap.read(frame)) {  
    cv::Mat display = frame.clone();
    
    double alpha = 0.8; // 对比度系数
    double beta = -80;  // 亮度调整值（负数降亮，正数增亮，范围-255到255）
    frame.convertTo(frame, -1, 1.1, 0);
    display.convertTo(display, -1, alpha, beta);

    cv::GaussianBlur(frame, frame, cv::Size(3,3), 0);
    cv::Mat hsvfram;
    cv::cvtColor(frame, hsvfram, cv::COLOR_BGR2HSV);
    
    cv::Mat blueMask;  
    cv::inRange(hsvfram, blueLower, blueUpper, blueMask);
    
    
    cv::Mat kernel_1 = cv::getStructuringElement(cv::MORPH_RECT,cv::Size(3,3));
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(7,7));
    cv::morphologyEx(blueMask, blueMask, cv::MORPH_CLOSE, kernel);
    cv::morphologyEx(blueMask, blueMask, cv::MORPH_OPEN, kernel_1);
    

    std::vector<cv::RotatedRect> lightBars;
    conterdetection(blueMask,lightBars,display);

    cv::imshow("Armor Detection", display);
    if (cv::waitKey(1) == 27) {
     break;
    }    
            
  }

  cap.release();
  cv::destroyAllWindows();
  return 0;

}


bool checkvideo(cv::VideoCapture& cap,const std::string& videopath)
{
  cap.open(videopath);
  if(!cap.isOpened()){
    std::cout<<"false"<<std::endl;
    return false;
  }
  std::cout<<"true"<<std::endl;
  return true;
}


void conterdetection(cv::Mat& blueMask,std::vector<cv::RotatedRect>& lightBars,cv::Mat& outputFrame)
{
  std::vector<std::vector<cv::Point>> contours;
  std::vector<cv::Vec4i> hierarchy;
  cv::findContours(blueMask, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
  
  lightBars.clear();

  for (const auto& contour : contours) {
    
    double area = cv::contourArea(contour);
    if (area < 80 || area > 6000) continue;
    
    
    cv::RotatedRect rect = cv::minAreaRect(contour);
    cv::Size2f size = rect.size;
    
    
    //float aspectRatio = std::max(size.width, size.height) / std::min(size.width, size.height);
    //if (aspectRatio < 1.5 || aspectRatio > 10.0) continue;

    double perimeter = cv::arcLength(contour, true);
    if (perimeter < 100) continue;

    int width = std::min(size.width,size.height);
    int height=std::max(size.width,size.height);
    if (width < 3 || height < 7 ) continue;

    
    lightBars.push_back(rect);

    cv::Point2f vertices[4];
    rect.points(vertices);
    for (int j = 0; j < 4; j++) {
     line(outputFrame, vertices[j], vertices[(j+1)%4], cv::Scalar(0, 255, 255), 2);
    }

  }

  std::vector<std::vector<cv::Point2f>> armorCandidates;
   
  std::vector<bool> used(lightBars.size(), false);

  int armorCount = 0;
  for (size_t i = 0; i < lightBars.size(); i++) {
    if (used[i]) continue; 
    for (size_t j = i + 1; j < lightBars.size(); j++) {
      if (used[j]) continue; 
      const cv::RotatedRect& bar1 = lightBars[i];
      const cv::RotatedRect& bar2 = lightBars[j];
        
      float centerDistance = cv::norm(bar1.center - bar2.center);
      float angleDiff = std::abs(bar1.angle - bar2.angle);
      float heightDiff = std::abs(bar1.size.height - bar2.size.height);
        
        
      if (centerDistance > 80 && centerDistance < 250 &&angleDiff < 10.0 && heightDiff < 20)  {
            
        std::vector<cv::Point2f> corners = extractArmorCorners(bar1, bar2);
        
        float width = cv::norm(corners[0] - corners[1]);
        float height = cv::norm(corners[1] - corners[2]);
        float armorRatio = width / height;

        if(armorRatio>1.0&&armorRatio<2.5){
         drawArmor(outputFrame, corners, armorCount);
        }

        armorCount++;
        used[i] = true;
        used[j] = true;
        break;
      }
    }
  }
}



std::vector<cv::Point2f> extractArmorCorners(const cv::RotatedRect& leftBar, const cv::RotatedRect& rightBar){
  
  std::vector<cv::Point2f> corners(4);

  cv::Point2f leftPoints[4], rightPoints[4];
  leftBar.points(leftPoints);
  rightBar.points(rightPoints);
    
    
  const cv::RotatedRect* pLeft = &leftBar;
  const cv::RotatedRect* pRight = &rightBar;
  if (leftBar.center.x > rightBar.center.x) {
    std::swap(pLeft, pRight);
  }

  cv::Point2f leftPts[4], rightPts[4];
  pLeft->points(leftPts);
  pRight->points(rightPts);

    
  auto sortByY = [](const cv::Point2f& a, const cv::Point2f& b) { return a.y < b.y; };
    
  // 左
  std::vector<cv::Point2f> leftVec(leftPts, leftPts+4);
  std::sort(leftVec.begin(), leftVec.end(), sortByY);
  cv::Point2f leftTop1 = leftVec[0], leftTop2 = leftVec[1];
  cv::Point2f leftBottom1 = leftVec[2], leftBottom2 = leftVec[3];
    
  // 右
  std::vector<cv::Point2f> rightVec(rightPts, rightPts+4);
  std::sort(rightVec.begin(), rightVec.end(), sortByY);
  cv::Point2f rightTop1 = rightVec[0], rightTop2 = rightVec[1];
  cv::Point2f rightBottom1 = rightVec[2], rightBottom2 = rightVec[3];

  
  cv::Point2f leftTop = (leftTop1 + leftTop2) / 2.0f;
  cv::Point2f leftBottom = (leftBottom1 + leftBottom2) / 2.0f;
  cv::Point2f rightTop = (rightTop1 + rightTop2) / 2.0f;
  cv::Point2f rightBottom = (rightBottom1 + rightBottom2) / 2.0f;
   
  
  corners[0] = leftTop;
  corners[1] = rightTop;
  corners[2] = rightBottom;
  corners[3] = leftBottom;
  return corners;
}

void drawArmor(cv::Mat& frame, const std::vector<cv::Point2f>& corners, int armorIndex) {
    
  for (int i = 0; i < 4; i++) {
    line(frame, corners[i], corners[(i+1)%4], cv::Scalar(0, 255, 0), 3);
  }
}