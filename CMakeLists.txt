# 最低 CMake 版本要求
cmake_minimum_required(VERSION 3.10)

# 项目名称
project(ARMOR_DETECTOR)

# 设置 C++ 标准（建议 C++11 及以上，OpenCV 常用支持版本）
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 查找 OpenCV 库（需提前安装 OpenCV）
find_package(OpenCV REQUIRED)

# 包含 OpenCV 头文件目录
include_directories(${OpenCV_INCLUDE_DIRS})

# 添加可执行文件，关联 src.cpp 源代码
add_executable(armor_detector scr.cpp)

# 链接 OpenCV 库到可执行文件
target_link_libraries(armor_detector ${OpenCV_LIBS})
