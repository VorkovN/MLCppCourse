cmake_minimum_required(VERSION 4.1)
project(OpenCV)

set(CMAKE_CXX_STANDARD 20)

include(${CMAKE_CURRENT_SOURCE_DIR}/../../cmake/ConanSetup.cmake)
setup_conan()

find_package(OpenCV REQUIRED COMPONENTS core calib3d)
message(STATUS "OpenCV path: ${OpenCV_INCLUDE_DIRS}")

add_executable(${PROJECT_NAME} main.cpp)

target_link_libraries(${PROJECT_NAME}
        PUBLIC
        ${OpenCV_LIBS}
)