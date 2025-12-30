#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <optional>

int main() {
    cv::VideoCapture camera(0);

    if (!camera.isOpened()) {
        std::cerr << "Ошибка: не удалось открыть камеру" << std::endl;
        return -1;
    }

    camera.set(cv::CAP_PROP_FRAME_WIDTH, 1280);
    camera.set(cv::CAP_PROP_FRAME_HEIGHT, 720);
    camera.set(cv::CAP_PROP_FPS, 30);

    cv::Mat frame, hsv, mask;
    std::optional<std::pair<cv::Point, int> > detectedCircle;

    while (true) {
        camera >> frame;
        if (frame.empty()) {
            break;
        }

        detectedCircle.reset();

        cv::cvtColor(frame, hsv, cv::COLOR_BGR2HSV);

        cv::Scalar lowerGreen(40, 50, 50);
        cv::Scalar upperGreen(80, 255, 255);

        cv::inRange(hsv, lowerGreen, upperGreen, mask);

        std::vector<cv::Vec3f> circles;
        cv::HoughCircles(mask, circles, cv::HOUGH_GRADIENT_ALT, 20.0, 10, 10, 0.70, 50, 2000);

        if (!circles.empty()) {
            auto maxCircle = *std::max_element(circles.begin(), circles.end(),
                                               [](const cv::Vec3f &a, const cv::Vec3f &b) { return a[2] < b[2]; });
            detectedCircle = std::make_pair(cv::Point(maxCircle[0], maxCircle[1]), maxCircle[2]);
        } else {
            std::cout << "Круги не найдены" << std::endl;
        }

        if (detectedCircle.has_value()) {
            const auto &[center, radius] = detectedCircle.value();
            cv::circle(frame, center, radius, cv::Scalar(255, 0, 0), 2);
            cv::circle(frame, center, 3, cv::Scalar(0, 0, 255), -1);

            std::string coordText = "(" + std::to_string(center.x) + ", " + std::to_string(center.y) + ")";
            cv::putText(frame, coordText, cv::Point(center.x + 10, center.y),
                        cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 2);
        }

        cv::imshow("Зеленый круг", frame);

        if (cv::waitKey(1) == 27) { // esc
            break;
        }
    }

    camera.release();
    cv::destroyAllWindows();

    return 0;
}
