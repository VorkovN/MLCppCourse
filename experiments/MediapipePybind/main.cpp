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
        py::module_ sys = py::module_::import("sys");

#ifdef MEDIAPIPEPYBIND_VENV_SITE_PACKAGES
        sys.attr("path").attr("insert")(0, MEDIAPIPEPYBIND_VENV_SITE_PACKAGES);
#endif
        
        try {
            py::module_ mediapipe = py::module_::import("mediapipe");
            py::module_ python_tasks = py::module_::import("mediapipe.tasks.python");
            py::module_ vision = py::module_::import("mediapipe.tasks.python.vision");
            py::module_ base_options_module = py::module_::import("mediapipe.tasks.python.core.base_options");
        
            py::object FaceLandmarker = vision.attr("FaceLandmarker");
            py::object FaceLandmarkerOptions = vision.attr("FaceLandmarkerOptions");

            py::object BaseOptions = base_options_module.attr("BaseOptions");

            py::object RunningMode;
            if (py::hasattr(vision, "RunningMode")) {
                RunningMode = vision.attr("RunningMode");
            } else if (py::hasattr(vision, "VisionTaskRunningMode")) {
                RunningMode = vision.attr("VisionTaskRunningMode");
            } else {
                RunningMode = vision.attr("RunningMode");
            }
        
            py::kwargs base_options_kwargs;
            base_options_kwargs["model_asset_path"] = model_path;
            py::object base_options = BaseOptions(**base_options_kwargs);
        
            py::kwargs options_kwargs;
            options_kwargs["base_options"] = base_options;
            options_kwargs["running_mode"] = RunningMode.attr("VIDEO");
            options_kwargs["num_faces"] = 1;
            py::object options = FaceLandmarkerOptions(**options_kwargs);
        
            landmarker_ = FaceLandmarker.attr("create_from_options")(options);
        } catch (const py::error_already_set& e) {
            std::cerr << "Ошибка импорта mediapipe: " << e.what() << std::endl;
            py::object sys_path = sys.attr("path");
            std::cerr << "sys.path:" << std::endl;
            for (size_t i = 0; i < py::len(sys_path); ++i) {
                std::cerr << "  " << sys_path.attr("__getitem__")(py::cast(i)).cast<std::string>() << std::endl;
            }
            throw;
        }
    }
    
    std::vector<std::pair<int, int>> detect(const cv::Mat& frame) {
        std::vector<std::pair<int, int>> landmarks;
        
        cv::Mat rgb_frame;
        cv::cvtColor(frame, rgb_frame, cv::COLOR_BGR2RGB);
        
        cv::Mat rgb_frame_contiguous;
        rgb_frame.copyTo(rgb_frame_contiguous);
        
        size_t size = rgb_frame_contiguous.rows * rgb_frame_contiguous.cols * 3;
        py::array_t<uint8_t> img_array({rgb_frame_contiguous.rows, rgb_frame_contiguous.cols, 3});
        std::memcpy(img_array.mutable_data(), rgb_frame_contiguous.data, size);
        
        py::module_ mediapipe = py::module_::import("mediapipe");
        py::object Image = mediapipe.attr("Image");
        py::object ImageFormat = mediapipe.attr("ImageFormat");
        py::kwargs image_kwargs;
        image_kwargs["image_format"] = ImageFormat.attr("SRGB");
        image_kwargs["data"] = img_array;
        py::object image = Image(**image_kwargs);
        
        py::object result = landmarker_.attr("detect_for_video")(image, timestamp_ms_);
        
        if (!result.is_none()) {
            py::object face_landmarks_list = result.attr("face_landmarks");
            if (!face_landmarks_list.is_none() && py::len(face_landmarks_list) > 0) {
                py::object first_face = face_landmarks_list.attr("__getitem__")(0);
                for (size_t i = 0; i < py::len(first_face); ++i) {
                    py::object landmark = first_face.attr("__getitem__")(py::cast(i));
                    auto x = landmark.attr("x").cast<double>();
                    auto y = landmark.attr("y").cast<double>();
                    landmarks.emplace_back(
                        static_cast<int>(x * frame.cols),
                        static_cast<int>(y * frame.rows)
                    );
                }
            }
        }
        
        timestamp_ms_ += 33;
        return landmarks;
    }
    
private:
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

    try {
        FaceLandmarkerWrapper landmarker(model_path);

        cv::Mat frame;
        const int delay_ms = 33;

        while (true) {
            bool success = camera.read(frame);
            if (!success || frame.empty()) {
                break;
            }

            auto landmarks = landmarker.detect(frame);

            for (const auto& [x, y] : landmarks) {
                cv::circle(frame, cv::Point(x, y), 2, cv::Scalar(0, 255, 0), -1);
            }

            try {
                cv::imshow("Face Landmarks", frame);
            } catch (const cv::Exception& e) {
                std::cerr << e.what() << std::endl;
                break;
            }

            int key = cv::waitKey(delay_ms);
            if (key == 27 || key == 'q') {
                break;
            }
        }
    } catch (const py::error_already_set& e) {
        std::cerr << "Ошибка Python: " << e.what() << std::endl;
        return -1;
    } catch (const std::exception& e) {
        std::cerr << "Ошибка: " << e.what() << std::endl;
        return -1;
    }

    camera.release();
    cv::destroyAllWindows();
    return 0;
}
