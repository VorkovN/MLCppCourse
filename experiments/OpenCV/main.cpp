#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <optional>

int main() {
    // Открываем камеру с индексом 0 (обычно это встроенная или первая подключённая камера). На вашем устройстве индекс камеры может оказаться другим
    cv::VideoCapture camera(0);

    // Перед обращением к камере обязательно нужно проверить, что камера открылась
    if (!camera.isOpened()) {
        std::cerr << "Ошибка: не удалось открыть камеру" << std::endl;
        return -1;
    }

    // Просим у драйвера камеры желаемые параметры видеопотока.
    // Важно: это "request", а не гарантия — некоторые камеры игнорируют часть настроек.
    camera.set(cv::CAP_PROP_FRAME_WIDTH, 1280);
    camera.set(cv::CAP_PROP_FRAME_HEIGHT, 720);
    camera.set(cv::CAP_PROP_FPS, 30);

    // frame — исходный кадр в BGR,
    // gray — тот же кадр в градациях серого (для поиска окружностей без привязки к цвету).
    cv::Mat frame, gray;


    while (true) {
        // Читаем следующий кадр из камеры.
        // Оператор >> для VideoCapture эквивалентен camera.read(frame).
        camera >> frame;
        if (frame.empty()) {
            // Кадр пустой — источник закончился или произошла ошибка.
            break;
        }

        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
        cv::GaussianBlur(gray, gray, cv::Size(9, 9), 2.0, 2.0);

        // Вектор найденных окружностей. cv::Vec3f хранит (center_x, center_y, radius).
        std::vector<cv::Vec3f> circles;

        // HoughCircles ищет окружности по изображению.
        cv::HoughCircles(gray, circles, cv::HOUGH_GRADIENT_ALT, 20.0, 10, 10, 0.9, 50, 2000);


        if (!circles.empty()) {
            // Если окружности найдены — берём самую большую по радиусу (часто это "главный" объект).
            auto maxCircle = *std::max_element(circles.begin(), circles.end(),
                                               [](const cv::Vec3f &a, const cv::Vec3f &b) { return a[2] < b[2]; });

            // maxCircle[0], maxCircle[1] — координаты центра, maxCircle[2] — радиус.
            const auto center = cv::Point(maxCircle[0], maxCircle[1]);
            const auto radius = maxCircle[2];

            // Рисуем окружность (синий контур).
            cv::circle(frame, center, radius, cv::Scalar(255, 0, 0), 2);
            // Рисуем точку в центре (красная заливка).
            cv::circle(frame, center, 3, cv::Scalar(0, 0, 255), -1);

            // Подписываем координаты центра рядом с окружностью.
            std::string coordText = "(" + std::to_string(center.x) + ", " + std::to_string(center.y) + ")";
            cv::putText(frame, coordText, cv::Point(center.x + 10, center.y),
                        cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 2);
        }

        // Показываем кадр в окне.
        cv::imshow("Круг", frame);

        // waitKey:
        // - даёт OpenCV обработать события окна,
        // - возвращает код нажатой клавиши (если была).
        // Здесь выходим по ESC (27).
        if (cv::waitKey(1) == 27) {
            break;
        }
    }

    // Освобождаем ресурс камеры и закрываем окна.
    camera.release();
    cv::destroyAllWindows();

    return 0;
}
