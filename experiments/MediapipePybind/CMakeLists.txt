cmake_minimum_required(VERSION 4.1)
project(MediapipePybind)

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

set(Python_EXECUTABLE "${CMAKE_SOURCE_DIR}/venv/bin/python")
set(Python_FIND_VIRTUALENV ONLY)

if(NOT EXISTS "${Python_EXECUTABLE}")
    message(FATAL_ERROR "Python из venv не найден: ${Python_EXECUTABLE}")
endif()

find_package(OpenCV REQUIRED COMPONENTS core imgproc highgui)
find_package(Python REQUIRED COMPONENTS Interpreter Development)
find_package(pybind11 REQUIRED)

add_executable(${PROJECT_NAME} main.cpp)

target_link_libraries(${PROJECT_NAME}
        PRIVATE
        pybind11::embed
        ${OpenCV_LIBS}
        Python::Python
)

target_include_directories(${PROJECT_NAME}
        PRIVATE
        ${Python_INCLUDE_DIRS}
)

message(STATUS "Python executable (CMake): ${Python_EXECUTABLE}")
message(STATUS "Python version (CMake): ${Python_VERSION}")
message(STATUS "Python include dirs (CMake): ${Python_INCLUDE_DIRS}")
message(STATUS "Python library (CMake): ${Python_LIBRARIES}")

execute_process(
        COMMAND "${Python_EXECUTABLE}" -c "import site; print(site.getsitepackages()[0])"
        OUTPUT_VARIABLE MEDIAPIPEPYBIND_VENV_SITE_PACKAGES
        OUTPUT_STRIP_TRAILING_WHITESPACE
)
message(STATUS "Venv site-packages: ${MEDIAPIPEPYBIND_VENV_SITE_PACKAGES}")

target_compile_definitions(${PROJECT_NAME}
        PRIVATE
        MEDIAPIPEPYBIND_VENV_SITE_PACKAGES="${MEDIAPIPEPYBIND_VENV_SITE_PACKAGES}"
)