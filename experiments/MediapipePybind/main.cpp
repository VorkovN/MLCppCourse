#include <opencv2/opencv.hpp>
#include <pybind11/embed.h>
#include <pybind11/numpy.h>
#include <filesystem>
#include <iostream>
#include <vector>
#include <cstring>

namespace py = pybind11;

class FaceLandmarkerWrapper {
public:
    FaceLandmarkerWrapper(const std::string& model_path) : guard_{} {
        // Поднимаем embedded Python и при необходимости добавляем путь до venv site-packages.
        // Это нужно, чтобы `import mediapipe` находил пакет в окружении, которое вы подготовили.
        py::module_ sys = py::module_::import("sys");
        sys.attr("path").attr("insert")(0, MEDIAPIPEPYBIND_VENV_SITE_PACKAGES);

        // Импортируем модули Mediapipe Task API из Python.
        py::module_ mediapipe = py::module_::import("mediapipe");
        py::module_ python_tasks = py::module_::import("mediapipe.tasks.python");
        py::module_ vision = py::module_::import("mediapipe.tasks.python.vision");
        py::module_ base_options_module = py::module_::import("mediapipe.tasks.python.core.base_options");

        py::object FaceLandmarker = vision.attr("FaceLandmarker");
        py::object FaceLandmarkerOptions = vision.attr("FaceLandmarkerOptions");

        py::object BaseOptions = base_options_module.attr("BaseOptions");
        py::object RunningMode = vision.attr("RunningMode");

        // base_options = BaseOptions(model_asset_path=model_path)
        py::kwargs base_options_kwargs;
        base_options_kwargs["model_asset_path"] = model_path;
        py::object base_options = BaseOptions(**base_options_kwargs);

        // options = FaceLandmarkerOptions(base_options=..., running_mode=VIDEO, num_faces=1)
        py::kwargs options_kwargs;
        options_kwargs["base_options"] = base_options;
        options_kwargs["running_mode"] = RunningMode.attr("VIDEO");
        options_kwargs["num_faces"] = 1;
        py::object options = FaceLandmarkerOptions(**options_kwargs);

        // landmarker_ — Python-объект детектора, который будем вызывать на каждом кадре.
        landmarker_ = FaceLandmarker.attr("create_from_options")(options);
    }
    
    std::vector<std::pair<int, int>> detect(const cv::Mat& frame) {
        std::vector<std::pair<int, int>> landmarks;
        
        // Гарантируем непрерывный буфер: numpy ожидает линейные данные.
        if (!frame.isContinuous()) {
            return {};
        }
        
        // Создаём numpy массив HWC uint8 и копируем туда пиксели.
        py::array_t<uint8_t> img_array({frame.rows, frame.cols, 3});
        std::memcpy(img_array.mutable_data(), frame.data, frame.rows * frame.cols * 3);
        
        // Строим mediapipe.Image(image_format=SRGB, data=np_array)
        py::module_ mediapipe = py::module_::import("mediapipe");
        py::object Image = mediapipe.attr("Image");
        py::object ImageFormat = mediapipe.attr("ImageFormat");
        py::kwargs image_kwargs;
        image_kwargs["image_format"] = ImageFormat.attr("SRGB");
        image_kwargs["data"] = img_array;
        py::object image = Image(**image_kwargs);
        
        // VIDEO режим требует timestamp (в миллисекундах).
        py::object result = landmarker_.attr("detect_for_video")(image, timestamp_ms_);
        
        if (!result.is_none()) {
            py::object face_landmarks_list = result.attr("face_landmarks");
            if (!face_landmarks_list.is_none() && py::len(face_landmarks_list) > 0) {
                py::object first_face = face_landmarks_list.attr("__getitem__")(0);
                for (size_t i = 0; i < py::len(first_face); ++i) {
                    py::object landmark = first_face.attr("__getitem__")(py::cast(i));
                    const auto x = landmark.attr("x").cast<double>();
                    const auto y = landmark.attr("y").cast<double>();

                    // x,y в [0..1] -> переводим в пиксели.
                    landmarks.emplace_back(
                        static_cast<int>(x * frame.cols),
                        static_cast<int>(y * frame.rows)
                    );
                }
            }
        }
        
        // Сдвигаем таймстемп на "примерно 30 FPS".
        timestamp_ms_ += 33;
        return landmarks;
    }
    
private:
    // guard_ должен жить столько же, сколько мы используем Python API.
    py::scoped_interpreter guard_;
    py::object landmarker_;
    int64_t timestamp_ms_ = 0;
};

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
        std::cerr << "Ошибка: файл модели не найден: " << model_path << std::endl;
        return -1;
    }

    // Создаём Python-детектор один раз, дальше используем его на каждом кадре.
    FaceLandmarkerWrapper landmarker(model_path);

    while (true) {
        cv::Mat frame;

        bool success = camera.read(frame);
        if (!success || frame.empty()) {
            break;
        }

        // OpenCV-кадр приходит в BGR, mediapipe ожидает RGB/SRGB.
        cv::Mat rgb_frame;
        cv::cvtColor(frame, rgb_frame, cv::COLOR_BGR2RGB);

        // Детектируем точки лица и рисуем их.
        auto landmarks = landmarker.detect(rgb_frame);

        for (const auto& [x, y] : landmarks) {
            cv::circle(frame, cv::Point(x, y), 2, cv::Scalar(0, 255, 0), -1);
        }

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

    camera.release();
    cv::destroyAllWindows();
    return 0;
}
