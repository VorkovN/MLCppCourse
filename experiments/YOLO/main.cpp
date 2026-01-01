#include <onnxruntime_cxx_api.h>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct LetterboxParams {
    float scale = 1.0F;
    float pad_x = 0.0F;
    float pad_y = 0.0F;
};

struct Detection {
    cv::Rect2f box_xyxy; // В координатах исходного изображения
    float confidence = 0.0F;
};

float IoU(const cv::Rect2f& a, const cv::Rect2f& b) {
    const float inter_x1 = std::max(a.x, b.x);
    const float inter_y1 = std::max(a.y, b.y);
    const float inter_x2 = std::min(a.x + a.width, b.x + b.width);
    const float inter_y2 = std::min(a.y + a.height, b.y + b.height);

    const float inter_w = std::max(0.0F, inter_x2 - inter_x1);
    const float inter_h = std::max(0.0F, inter_y2 - inter_y1);
    const float inter_area = inter_w * inter_h;
    const float union_area = a.area() + b.area() - inter_area;
    if (union_area <= 0.0F) {
        return 0.0F;
    }
    return inter_area / union_area;
}

std::vector<int> NmsIndices(const std::vector<Detection>& detections, float iou_threshold) {
    std::vector<int> order(detections.size());
    std::iota(order.begin(), order.end(), 0);

    std::sort(order.begin(), order.end(), [&](int lhs, int rhs) {
        return detections[lhs].confidence > detections[rhs].confidence;
    });

    std::vector<int> keep;
    keep.reserve(order.size());

    std::vector<bool> suppressed(order.size(), false);
    for (size_t i = 0; i < order.size(); ++i) {
        if (suppressed[i]) {
            continue;
        }
        const int idx = order[i];
        keep.push_back(idx);

        for (size_t j = i + 1; j < order.size(); ++j) {
            if (suppressed[j]) {
                continue;
            }
            const int idx2 = order[j];
            if (IoU(detections[idx].box_xyxy, detections[idx2].box_xyxy) > iou_threshold) {
                suppressed[j] = true;
            }
        }
    }

    return keep;
}

cv::Mat Letterbox(const cv::Mat& bgr, int target_w, int target_h, LetterboxParams& params) {
    if (bgr.empty()) {
        throw std::runtime_error("Пустое изображение на входе letterbox");
    }
    const int src_w = bgr.cols;
    const int src_h = bgr.rows;

    const float r_w = static_cast<float>(target_w) / static_cast<float>(src_w);
    const float r_h = static_cast<float>(target_h) / static_cast<float>(src_h);
    const float r = std::min(r_w, r_h);
    params.scale = r;

    const int new_w = static_cast<int>(std::round(static_cast<float>(src_w) * r));
    const int new_h = static_cast<int>(std::round(static_cast<float>(src_h) * r));

    cv::Mat resized;
    cv::resize(bgr, resized, cv::Size(new_w, new_h), 0, 0, cv::INTER_LINEAR);

    const int pad_w = target_w - new_w;
    const int pad_h = target_h - new_h;

    const int pad_left = pad_w / 2;
    const int pad_right = pad_w - pad_left;
    const int pad_top = pad_h / 2;
    const int pad_bottom = pad_h - pad_top;

    params.pad_x = static_cast<float>(pad_left);
    params.pad_y = static_cast<float>(pad_top);

    cv::Mat out;
    cv::copyMakeBorder(resized, out, pad_top, pad_bottom, pad_left, pad_right, cv::BORDER_CONSTANT,
                       cv::Scalar(114, 114, 114));
    return out;
}

} // namespace

int main() {
    // Максимально простой “хардкод” под задачу курса:
    // - модель лежит рядом как ./yolo11n.onnx
    // - используем камеру 0
    // - интересует только COCO-класс person (class_id = 0)
    const std::string model_path = "./yolo11n.onnx";
    constexpr int camera_id = 0;
    constexpr int target_class_id = 0; // COCO: person
    constexpr float conf_threshold = 0.35F;
    constexpr float iou_threshold = 0.45F;

    if (!std::filesystem::exists(model_path)) {
        std::cerr << "Не найден файл модели: " << model_path << "\n";
        return 1;
    }

    // Инициализация ONNXRuntime
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "yolo_person");
    Ort::SessionOptions session_options;
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    Ort::Session session(env, model_path.c_str(), session_options);
    Ort::AllocatorWithDefaultOptions allocator;

    if (session.GetInputCount() != 1) {
        std::cerr << "Ожидался 1 вход у модели, получено: " << session.GetInputCount() << "\n";
        return 1;
    }
    if (session.GetOutputCount() < 1) {
        std::cerr << "У модели нет выходов\n";
        return 1;
    }

    const auto input_name = session.GetInputNameAllocated(0, allocator);
    std::vector<const char*> input_names = {input_name.get()};

    auto input_shape = session.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
    if (input_shape.size() != 4) {
        std::cerr << "Ожидалась форма входа NCHW, получили размерность: " << input_shape.size()
                  << "\n";
        return 1;
    }

    const int64_t input_n = input_shape[0];
    const int64_t input_c = input_shape[1];
    int64_t input_h = input_shape[2];
    int64_t input_w = input_shape[3];

    if (input_n != 1 && input_n != -1) {
        std::cerr << "Поддерживается только batch=1 (или динамический), получили: " << input_n
                  << "\n";
        return 1;
    }
    if (input_c != 3 && input_c != -1) {
        std::cerr << "Поддерживаются только 3 канала (или динамические), получили: " << input_c
                  << "\n";
        return 1;
    }
    // Часто в моделях стоят динамические размеры (-1)
    if (input_h <= 0) {
        input_h = 640;
    }
    if (input_w <= 0) {
        input_w = 640;
    }

    std::vector<int64_t> concrete_input_shape = {1, 3, input_h, input_w};

    // Имена выходов
    std::vector<Ort::AllocatedStringPtr> output_name_ptrs;
    std::vector<const char*> output_names;
    output_name_ptrs.reserve(session.GetOutputCount());
    output_names.reserve(session.GetOutputCount());
    for (size_t i = 0; i < session.GetOutputCount(); ++i) {
        output_name_ptrs.emplace_back(session.GetOutputNameAllocated(i, allocator));
        output_names.push_back(output_name_ptrs.back().get());
    }

    // Источник кадров: камера или видео
    cv::VideoCapture cap;
    cap.open(camera_id);
    if (!cap.isOpened()) {
        std::cerr << "Не удалось открыть источник видео\n";
        return 1;
    }

    cv::namedWindow("YOLO person", cv::WINDOW_NORMAL);

    std::vector<float> input_tensor_data(static_cast<size_t>(1 * 3 * input_h * input_w));
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    while (true) {
        cv::Mat frame_bgr;
        if (!cap.read(frame_bgr) || frame_bgr.empty()) {
            break;
        }

        LetterboxParams lb;
        cv::Mat lb_bgr = Letterbox(frame_bgr, static_cast<int>(input_w), static_cast<int>(input_h),
                                   lb);

        cv::Mat lb_rgb;
        cv::cvtColor(lb_bgr, lb_rgb, cv::COLOR_BGR2RGB);
        lb_rgb.convertTo(lb_rgb, CV_32F, 1.0 / 255.0);

        std::vector<cv::Mat> chw(3);
        cv::split(lb_rgb, chw);
        const size_t plane_size = static_cast<size_t>(input_h * input_w);
        for (size_t c = 0; c < 3; ++c) {
            std::memcpy(input_tensor_data.data() + c * plane_size, chw[c].ptr<float>(),
                        plane_size * sizeof(float));
        }

        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            memory_info, input_tensor_data.data(), input_tensor_data.size(),
            concrete_input_shape.data(), concrete_input_shape.size());

        auto output_tensors = session.Run(Ort::RunOptions{nullptr}, input_names.data(),
                                          &input_tensor, 1, output_names.data(),
                                          output_names.size());
        if (output_tensors.empty()) {
            std::cerr << "Пустой результат инференса\n";
            break;
        }

        // Берем первый выход как основной детектор.
        const auto& out = output_tensors.front();
        auto out_info = out.GetTensorTypeAndShapeInfo();
        const auto out_shape = out_info.GetShape();
        const float* out_data = out.GetTensorData<float>();

        std::vector<Detection> detections;
        detections.reserve(256);

        auto push_det = [&](float cx, float cy, float w, float h, float conf) {
            if (conf < conf_threshold) {
                return;
            }

            // cx,cy,w,h в letterbox-координатах -> xyxy
            float x1 = cx - w * 0.5F;
            float y1 = cy - h * 0.5F;
            float x2 = cx + w * 0.5F;
            float y2 = cy + h * 0.5F;

            // Обратно в координаты исходного кадра
            x1 = (x1 - lb.pad_x) / lb.scale;
            y1 = (y1 - lb.pad_y) / lb.scale;
            x2 = (x2 - lb.pad_x) / lb.scale;
            y2 = (y2 - lb.pad_y) / lb.scale;

            x1 = std::clamp(x1, 0.0F, static_cast<float>(frame_bgr.cols - 1));
            y1 = std::clamp(y1, 0.0F, static_cast<float>(frame_bgr.rows - 1));
            x2 = std::clamp(x2, 0.0F, static_cast<float>(frame_bgr.cols - 1));
            y2 = std::clamp(y2, 0.0F, static_cast<float>(frame_bgr.rows - 1));

            const float bw = std::max(0.0F, x2 - x1);
            const float bh = std::max(0.0F, y2 - y1);
            if (bw <= 1.0F || bh <= 1.0F) {
                return;
            }

            detections.push_back(Detection{
                .box_xyxy = cv::Rect2f(x1, y1, bw, bh),
                .confidence = conf,
            });
        };

        // Поддерживаем только YOLO v11 (ultralytics): выход [1, 4+nc, num]
        // (значения классов — это confidence по классам, без отдельного objectness).
        if (out_shape.size() != 3 || out_shape[0] != 1) {
            std::cerr << "Неподдерживаемая форма выхода (ожидалось [1, 4+nc, num]): [";
            for (size_t i = 0; i < out_shape.size(); ++i) {
                std::cerr << out_shape[i] << (i + 1 == out_shape.size() ? "" : ", ");
            }
            std::cerr << "]\n";
        } else {
            const int64_t channels = out_shape[1];
            const int64_t num = out_shape[2];
            if (channels < 5 || num <= 0) {
                std::cerr << "Некорректная форма выхода (channels/num): [" << out_shape[0] << ", "
                          << out_shape[1] << ", " << out_shape[2] << "]\n";
            } else {
                const int64_t nc = channels - 4;
                if (target_class_id < 0 || target_class_id >= nc) {
                    std::cerr << "В модели nc=" << nc << ", но target_class_id=" << target_class_id
                              << "\n";
                } else {
                    for (int64_t i = 0; i < num; ++i) {
                        const float cx = out_data[i + 0 * num];
                        const float cy = out_data[i + 1 * num];
                        const float w = out_data[i + 2 * num];
                        const float h = out_data[i + 3 * num];
                        const float score = out_data[i + (4 + target_class_id) * num];
                        push_det(cx, cy, w, h, score);
                    }
                }
            }
        }

        const std::vector<int> keep = NmsIndices(detections, iou_threshold);
        const bool person_present = !keep.empty();
        {
            const std::string status_text = person_present ? "PERSON PRESENT" : "NO PERSON";
            const cv::Scalar bg_color = person_present ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 0, 255);
            const cv::Scalar fg_color = cv::Scalar(0, 0, 0);
            const int thickness = 2;
            int baseline = 0;
            const cv::Size ts = cv::getTextSize(status_text, cv::FONT_HERSHEY_SIMPLEX, 0.9, thickness, &baseline);
            cv::Rect bg(10, 10, ts.width + 20, ts.height + baseline + 20);
            cv::rectangle(frame_bgr, bg, bg_color, cv::FILLED);
            cv::putText(frame_bgr, status_text, cv::Point(bg.x + 10, bg.y + bg.height - 10),
                        cv::FONT_HERSHEY_SIMPLEX, 0.9, fg_color, thickness);
        }

        for (int idx : keep) {
            const auto& det = detections[idx];
            const cv::Rect box_i(
                static_cast<int>(det.box_xyxy.x),
                static_cast<int>(det.box_xyxy.y),
                static_cast<int>(det.box_xyxy.width),
                static_cast<int>(det.box_xyxy.height));

            cv::rectangle(frame_bgr, box_i, cv::Scalar(0, 255, 0), 2);
            const std::string label = "person " + std::to_string(det.confidence).substr(0, 4);
            const cv::Size ts = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.6, 1, nullptr);
            cv::Rect bg(box_i.x, std::max(0, box_i.y - ts.height - 6), ts.width + 6,
                        ts.height + 6);
            cv::rectangle(frame_bgr, bg, cv::Scalar(0, 255, 0), cv::FILLED);
            cv::putText(frame_bgr, label, cv::Point(bg.x + 3, bg.y + bg.height - 4),
                        cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 0), 1);
        }

        cv::imshow("YOLO person", frame_bgr);
        const int key = cv::waitKey(1);
        if (key == 27 || key == 'q' || key == 'Q') { // ESC / q
            break;
        }
    }

    return 0;
}