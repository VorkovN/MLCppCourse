#include <opencv2/opencv.hpp>
#include <mediapipe/tasks/cc/vision/face_landmarker/face_landmarker.h>
#include <mediapipe/framework/formats/image.h>
#include <mediapipe/framework/formats/image_frame.h>
#include <memory>
#include <filesystem>
#include <iostream>

using namespace mediapipe::tasks::vision;
using namespace mediapipe::tasks::vision::core;
using namespace mediapipe::tasks::vision::face_landmarker;

int main() {
    cv::VideoCapture camera(0);
    if (!camera.isOpened()) {
        return -1;
    }

    camera.set(cv::CAP_PROP_FRAME_WIDTH, 1280);
    camera.set(cv::CAP_PROP_FRAME_HEIGHT, 720);
    camera.set(cv::CAP_PROP_FPS, 30);

    const std::string model_path = "face_landmarker.task";
    if (!std::filesystem::exists(model_path)) {
        std::cerr << "Ошибка: файл модели не найден: " << model_path << "Убедитесь что корневая директория: Mediapipe" << std::endl;
        return -1;
    }

    auto options = std::make_unique<FaceLandmarkerOptions>();
    options->base_options.model_asset_path = model_path;
    options->running_mode = VIDEO;
    options->num_faces = 1;

    auto landmarker_or = FaceLandmarker::Create(std::move(options));
    if (!landmarker_or.ok()) {
        std::cerr << "Ошибка создания FaceLandmarker: " << landmarker_or.status().ToString() << std::endl;
        return -1;
    }
    auto landmarker = std::move(landmarker_or.value());

    cv::Mat frame, rgb_frame;
    int64_t timestamp_ms = 0;
    const int delay_ms = 33;

    while (true) {
        bool success = camera.read(frame);
        if (!success || frame.empty()) {
            break;
        }

        cv::cvtColor(frame, rgb_frame, cv::COLOR_BGR2RGB);

        auto image_frame = std::make_shared<mediapipe::ImageFrame>(
            mediapipe::ImageFormat::SRGB,
            rgb_frame.cols,
            rgb_frame.rows,
            mediapipe::ImageFrame::kDefaultAlignmentBoundary
        );

        std::memcpy(image_frame->MutablePixelData(), rgb_frame.data,
                   rgb_frame.rows * rgb_frame.cols * 3);

        mediapipe::Image image(image_frame);
        auto detection_result_or = landmarker->DetectForVideo(image, timestamp_ms);

        if (detection_result_or.ok() && !detection_result_or.value().face_landmarks.empty()) {
            const auto& face_landmarks = detection_result_or.value().face_landmarks[0].landmarks;
            for (const auto& landmark : face_landmarks) {
                int x = static_cast<int>(landmark.x * frame.cols);
                int y = static_cast<int>(landmark.y * frame.rows);
                cv::circle(frame, cv::Point(x, y), 2, cv::Scalar(0, 255, 0), -1);
            }
        }

        try {
            cv::imshow("Face Landmarks", frame);
        } catch (const cv::Exception& e) {
            std::cerr << e.what() << std::endl;
            break;
        }

        timestamp_ms += delay_ms;
        int key = cv::waitKey(delay_ms);
        if (key == 27 || key == 'q') {
            break;
        }
    }

    camera.release();
    cv::destroyAllWindows();
    return 0;
}
