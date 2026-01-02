cmake_minimum_required(VERSION 3.22)
project(Mediapipe)

set(CMAKE_CXX_STANDARD 20)

### Conan START
if(APPLE)
    set(CONAN_PROFILE "macos_debug")
else()
    set(CONAN_PROFILE "linux_debug")
endif()

find_program(CONAN_EXECUTABLE NAMES conan REQUIRED)

execute_process(
        COMMAND ${CONAN_EXECUTABLE} install ${CMAKE_SOURCE_DIR}/conan/conanfile.txt
        --output-folder=${CMAKE_BINARY_DIR}
        --build=missing
        --profile:build ${CMAKE_SOURCE_DIR}/conan/profile_${CONAN_PROFILE}
        --profile:host ${CMAKE_SOURCE_DIR}/conan/profile_${CONAN_PROFILE}
        --lockfile ${CMAKE_SOURCE_DIR}/conan/conan.lock
        RESULT_VARIABLE CONAN_RESULT
)

if(NOT CONAN_RESULT EQUAL 0)
    message(FATAL_ERROR "Conan install failed with code ${CONAN_RESULT}")
endif()

include(${CMAKE_BINARY_DIR}/conan_toolchain.cmake)
### Conan END

find_package(mediapipe CONFIG REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core imgproc highgui)

add_executable(${PROJECT_NAME} main.cpp)

target_link_libraries(${PROJECT_NAME}
        PUBLIC
        mediapipe::mediapipe
        ${OpenCV_LIBS}
)