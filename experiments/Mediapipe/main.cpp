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
    // Открываем камеру с индексом 0 (обычно это встроенная или первая подключённая камера). На вашем устройстве индекс камеры может оказаться другим
    cv::VideoCapture camera(0);
    if (!camera.isOpened()) {
        return -1;
    }

    // Просим камеру отдавать кадры 1280x720 @ 30 FPS (не всегда гарантируется драйвером).
    camera.set(cv::CAP_PROP_FRAME_WIDTH, 1280);
    camera.set(cv::CAP_PROP_FRAME_HEIGHT, 720);
    camera.set(cv::CAP_PROP_FPS, 30);

    // Файл модели Task API. Ожидаем, что рабочая директория — experiments/Mediapipe.
    const std::string model_path = "face_landmarker.task";
    if (!std::filesystem::exists(model_path)) {
        std::cerr << "Ошибка: файл модели не найден: " << model_path << "Убедитесь что корневая директория: Mediapipe" << std::endl;
        return -1;
    }

    // Настройка FaceLandmarker:
    // - путь к модели,
    // - режим VIDEO (нужны таймстемпы кадров),
    // - максимум 1 лицо (проще и быстрее).
    auto options = std::make_unique<FaceLandmarkerOptions>();
    options->base_options.model_asset_path = model_path;
    options->running_mode = VIDEO;
    options->num_faces = 1;

    // Create(...) возвращает StatusOr: либо готовый объект, либо описание ошибки.
    auto landmarker_or = FaceLandmarker::Create(std::move(options));
    if (!landmarker_or.ok()) {
        std::cerr << "Ошибка создания FaceLandmarker: " << landmarker_or.status().ToString() << std::endl;
        return -1;
    }
    auto landmarker = std::move(landmarker_or.value());

    cv::Mat frame, rgb_frame;

    // VIDEO-режим требует monotonically increasing таймстемпы (мс).
    // Здесь делаем простую модель: прибавляем 33 мс на каждый кадр (≈30 FPS).
    int64_t timestamp_ms = 0;
    const int delay_ms = 33;

    while (true) {
        // Считываем кадр в BGR.
        bool success = camera.read(frame);
        if (!success || frame.empty()) {
            break;
        }

        // Mediapipe ожидает SRGB/RGB, а OpenCV отдаёт BGR -> конвертируем.
        cv::cvtColor(frame, rgb_frame, cv::COLOR_BGR2RGB);

        // Task API работает с mediapipe::Image.
        // Чтобы создать Image из OpenCV-кадра, упаковываем пиксели в ImageFrame.
        auto image_frame = std::make_shared<mediapipe::ImageFrame>(
            mediapipe::ImageFormat::SRGB,
            rgb_frame.cols,
            rgb_frame.rows,
            mediapipe::ImageFrame::kDefaultAlignmentBoundary
        );

        // Копируем bytes из cv::Mat в буфер ImageFrame.
        // Важно: это копия. Она упрощает жизнь (владелец памяти — ImageFrame),
        // но добавляет накладные расходы.
        std::memcpy(image_frame->MutablePixelData(), rgb_frame.data,
                   rgb_frame.rows * rgb_frame.cols * 3);

        mediapipe::Image image(image_frame);

        // Инференс для видео-режима: на вход кадр и timestamp, на выход — landmarks.
        auto detection_result_or = landmarker->DetectForVideo(image, timestamp_ms);

        if (detection_result_or.ok() && !detection_result_or.value().face_landmarks.empty()) {
            // Берём первое лицо (num_faces = 1).
            const auto& face_landmarks = detection_result_or.value().face_landmarks[0].landmarks;
            for (const auto& landmark : face_landmarks) {
                // landmark.x/y — нормализованные координаты [0..1].
                // Переводим в пиксели исходного кадра.
                int x = static_cast<int>(landmark.x * frame.cols);
                int y = static_cast<int>(landmark.y * frame.rows);
                cv::circle(frame, cv::Point(x, y), 2, cv::Scalar(0, 255, 0), -1);
            }
        }
        // Сдвигаем таймстемп и обрабатываем события окна.
        timestamp_ms += delay_ms;

        // Показываем кадр в окне.
        cv::imshow("Face Landmarks", frame);

        // waitKey:
        // - даёт OpenCV обработать события окна,
        // - возвращает код нажатой клавиши (если была).
        // Здесь выходим по ESC (27).
        if (cv::waitKey(1) == 27) {
            break;
        }
    }

    // Освобождаем ресурсы.
    camera.release();
    cv::destroyAllWindows();
    return 0;
}
