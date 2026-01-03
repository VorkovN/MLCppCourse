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

    // frame — исходный кадр в цветовом пространстве BGR,
    // hsv — тот же кадр в HSV (удобно выделять цвет по диапазону hue),
    // mask — бинарная маска (0/255), где 255 означает "пиксель подходит под условие".
    cv::Mat frame, hsv, mask;


    while (true) {
        // Читаем следующий кадр из камеры.
        // Оператор >> для VideoCapture эквивалентен camera.read(frame).
        camera >> frame;
        if (frame.empty()) {
            // Кадр пустой — источник закончился или произошла ошибка.
            break;
        }

        // OpenCV по умолчанию отдаёт кадры в BGR. Для выделения зелёного цвета удобнее HSV.
        cv::cvtColor(frame, hsv, cv::COLOR_BGR2HSV);

        // Диапазон зелёного в HSV:
        // Hue примерно 40..80 (зависит от освещения/камеры),
        // Saturation/Value отсекают слишком серые/тёмные пиксели.
        cv::Scalar lowerGreen(40, 50, 50);
        cv::Scalar upperGreen(80, 255, 255);

        // Строим бинарную маску: пиксели внутри диапазона -> 255, остальные -> 0.
        cv::inRange(hsv, lowerGreen, upperGreen, mask);

        // Вектор найденных окружностей. cv::Vec3f хранит (center_x, center_y, radius).
        std::vector<cv::Vec3f> circles;

        // HoughCircles ищет окружности по изображению.
        // Мы передаём mask, т.е. ищем окружность среди "зелёных" пикселей.
        cv::HoughCircles(mask, circles, cv::HOUGH_GRADIENT_ALT, 20.0, 10, 10, 0.70, 50, 2000);


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
        cv::imshow("Зеленый круг", frame);

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
