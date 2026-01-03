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

// Полный пайплайн детекции YOLO в C++:
// камера -> препроцессинг (letterbox + RGB + нормализация + CHW) -> ONNX Runtime -> постпроцессинг (декод + NMS)
// Важно: точность детекции крайне чувствительна к препроцессингу и обратному пересчёту координат.

struct LetterboxParams {
    float scale = 1.0F;  // Масштаб, применённый к изображению при letterbox
    float pad_x = 0.0F; // Смещение паддинга по X (для обратного пересчёта координат)
    float pad_y = 0.0F; // Смещение паддинга по Y (для обратного пересчёта координат)
};

struct Detection {
    cv::Rect2f box_xyxy; // Bounding box в координатах исходного изображения (x1, y1, width, height)
    float confidence = 0.0F; // Уверенность детекции [0..1]
};

// IoU (Intersection-over-Union) — мера пересечения двух прямоугольников:
// площадь пересечения / площадь объединения. Используется в NMS для подавления дубликатов.
float IoU(const cv::Rect2f &a, const cv::Rect2f &b) {
    const float inter_x1 = std::max(a.x, b.x);
    const float inter_y1 = std::max(a.y, b.y);
    const float inter_x2 = std::min(a.x + a.width, b.x + b.width);
    const float inter_y2 = std::min(a.y + a.height, b.y + b.height);

    const float inter_w = std::max(0.0F, inter_x2 - inter_x1);
    const float inter_h = std::max(0.0F, inter_y2 - inter_y1);
    const float inter_area = inter_w * inter_h;
    const float union_area = a.area() + b.area() - inter_area;
    return union_area > 0.0F ? inter_area / union_area : 0.0F;
}

// NMS (Non-Maximum Suppression): подавляет дублирующие детекции.
// Алгоритм: 1) сортируем по confidence убыванию, 2) берём самую уверенную,
// 3) подавляем все детекции с IoU > threshold относительно выбранной.
std::vector<int> NmsIndices(const std::vector<Detection> &detections, float iou_threshold) {
    std::vector<int> order(detections.size());
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](int lhs, int rhs) {
        return detections[lhs].confidence > detections[rhs].confidence;
    });

    std::vector<int> keep;
    std::vector<bool> suppressed(order.size(), false);
    for (size_t i = 0; i < order.size(); ++i) {
        if (suppressed[i]) continue;
        keep.push_back(order[i]);
        for (size_t j = i + 1; j < order.size(); ++j) {
            if (!suppressed[j] && IoU(detections[order[i]].box_xyxy, detections[order[j]].box_xyxy) > iou_threshold) {
                suppressed[j] = true;
            }
        }
    }
    return keep;
}

// Letterbox: масштабирует изображение с сохранением пропорций и добавляет паддинги до target_w x target_h.
// Масштаб выбирается минимальным (чтобы изображение влезло), недостающие пиксели заполняются серым (114, 114, 114).
// Сохраняет параметры масштабирования в params для обратного пересчёта координат.
cv::Mat Letterbox(const cv::Mat &bgr, int target_w, int target_h, LetterboxParams &params) {
    const float r = std::min(static_cast<float>(target_w) / bgr.cols, static_cast<float>(target_h) / bgr.rows);
    params.scale = r;

    const int new_w = static_cast<int>(std::round(bgr.cols * r));
    const int new_h = static_cast<int>(std::round(bgr.rows * r));

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
    // 114 — стандартный цвет паддинга в Ultralytics YOLO (серый фон)
    cv::copyMakeBorder(resized, out, pad_top, pad_bottom, pad_left, pad_right, cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));
    return out;
}


int main() {
    // Параметры модели и детекции
    const std::string model_path = "./yolo11n.onnx";
    constexpr int target_class_id = 0; // COCO: person
    constexpr float conf_threshold = 0.35F; // Порог уверенности детекции
    constexpr float iou_threshold = 0.45F;  // Порог IoU для NMS

    if (!std::filesystem::exists(model_path)) {
        std::cerr << "Не найден файл модели: " << model_path << "\n";
        return 1;
    }

    // Инициализация ONNX Runtime: создаём окружение и сессию с оптимизациями графа.
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "yolo_person");
    Ort::SessionOptions session_options;
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    Ort::Session session(env, model_path.c_str(), session_options);
    Ort::AllocatorWithDefaultOptions allocator;

    // Получаем имена входов и выходов модели
    const auto input_name = session.GetInputNameAllocated(0, allocator);
    std::vector<const char *> input_names = {input_name.get()};

    // Читаем форму входного тензора. Обычно для YOLO это NCHW [batch, channels, height, width].
    // Если размеры динамические (-1), фиксируем 640x640 для realtime инференса.
    auto input_shape = session.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
    int64_t input_h = input_shape[2] > 0 ? input_shape[2] : 640;
    int64_t input_w = input_shape[3] > 0 ? input_shape[3] : 640;
    std::vector<int64_t> concrete_input_shape = {1, 3, input_h, input_w};

    // Получаем имена выходов модели
    std::vector<Ort::AllocatedStringPtr> output_name_ptrs;
    std::vector<const char *> output_names;
    for (size_t i = 0; i < session.GetOutputCount(); ++i) {
        output_name_ptrs.emplace_back(session.GetOutputNameAllocated(i, allocator));
        output_names.push_back(output_name_ptrs.back().get());
    }

    // Открываем камеру для захвата кадров
    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        std::cerr << "Не удалось открыть камеру\n";
        return 1;
    }

    cv::namedWindow("YOLO person", cv::WINDOW_NORMAL);

    // Предвыделяем буфер для входного тензора (NCHW: 1 x 3 x H x W)
    std::vector<float> input_tensor_data(3 * input_h * input_w);
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    while (true) {
        cv::Mat frame_bgr;
        if (!cap.read(frame_bgr) || frame_bgr.empty()) {
            break;
        }

        // --- Препроцессинг: подготовка кадра для модели ---
        LetterboxParams lb;
        cv::Mat lb_bgr = Letterbox(frame_bgr, static_cast<int>(input_w), static_cast<int>(input_h), lb);
        
        // BGR -> RGB (модель обучена на RGB)
        cv::Mat lb_rgb;
        cv::cvtColor(lb_bgr, lb_rgb, cv::COLOR_BGR2RGB);
        
        // Нормализация: uint8 [0..255] -> float32 [0..1]
        lb_rgb.convertTo(lb_rgb, CV_32F, 1.0 / 255.0);

        // HWC -> CHW: преобразуем из формата (height, width, channels) в (channels, height, width).
        // Разбиваем на каналы и копируем плоскости подряд в буфер тензора.
        std::vector<cv::Mat> chw(3);
        cv::split(lb_rgb, chw);
        const size_t plane_size = input_h * input_w;
        for (size_t c = 0; c < 3; ++c) {
            std::memcpy(input_tensor_data.data() + c * plane_size, chw[c].ptr<float>(), plane_size * sizeof(float));
        }

        // Создаём тензор ONNX Runtime и запускаем инференс
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            memory_info, input_tensor_data.data(), input_tensor_data.size(),
            concrete_input_shape.data(), concrete_input_shape.size());

        auto output_tensors = session.Run(Ort::RunOptions{nullptr}, input_names.data(), &input_tensor, 1, output_names.data(), output_names.size());
        if (output_tensors.empty()) {
            break;
        }

        // Получаем выход модели: для YOLO v11 это [1, 4+nc, num], где:
        // - 4 координаты (cx, cy, w, h) в формате center-width-height
        // - nc значений confidence по классам
        // - num количество предсказаний (anchor points)
        const auto &out = output_tensors.front();
        const auto out_shape = out.GetTensorTypeAndShapeInfo().GetShape();
        const float *out_data = out.GetTensorData<float>();

        // --- Постпроцессинг: декодирование предсказаний ---
        std::vector<Detection> detections;
        
        // Лямбда для добавления детекции: фильтрует по confidence, переводит cxcywh -> xyxy,
        // пересчитывает координаты из letterbox обратно в исходный кадр, клипует и отбрасывает слишком маленькие боксы.
        auto push_det = [&](float cx, float cy, float w, float h, float conf) {
            if (conf < conf_threshold) return;

            // cxcywh -> xyxy в координатах letterbox-изображения
            float x1 = (cx - w * 0.5F - lb.pad_x) / lb.scale;
            float y1 = (cy - h * 0.5F - lb.pad_y) / lb.scale;
            float x2 = (cx + w * 0.5F - lb.pad_x) / lb.scale;
            float y2 = (cy + h * 0.5F - lb.pad_y) / lb.scale;

            // Клипуем координаты в границы исходного кадра
            x1 = std::clamp(x1, 0.0F, static_cast<float>(frame_bgr.cols - 1));
            y1 = std::clamp(y1, 0.0F, static_cast<float>(frame_bgr.rows - 1));
            x2 = std::clamp(x2, 0.0F, static_cast<float>(frame_bgr.cols - 1));
            y2 = std::clamp(y2, 0.0F, static_cast<float>(frame_bgr.rows - 1));

            const float bw = std::max(0.0F, x2 - x1);
            const float bh = std::max(0.0F, y2 - y1);
            if (bw <= 1.0F || bh <= 1.0F) return; // Отбрасываем слишком маленькие боксы

            detections.push_back(Detection{cv::Rect2f(x1, y1, bw, bh), conf});
        };

        // Декодируем предсказания модели: для каждого anchor point извлекаем координаты и confidence нужного класса
        if (out_shape.size() == 3 && out_shape[0] == 1) {
            const int64_t num = out_shape[2];
            for (int64_t i = 0; i < num; ++i) {
                const float score = out_data[i + (4 + target_class_id) * num];
                push_det(out_data[i + 0 * num], out_data[i + 1 * num], out_data[i + 2 * num], out_data[i + 3 * num], score);
            }
        }

        // Применяем NMS для подавления дублирующих детекций
        const std::vector<int> keep = NmsIndices(detections, iou_threshold);
        const bool person_present = !keep.empty();

        // Визуализация: статус детекции (зелёный = найдено, красный = не найдено)
        const std::string status_text = person_present ? "PERSON PRESENT" : "NO PERSON";
        const cv::Scalar bg_color = person_present ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 0, 255);
        int baseline = 0;
        const cv::Size ts = cv::getTextSize(status_text, cv::FONT_HERSHEY_SIMPLEX, 0.9, 2, &baseline);
        cv::Rect bg(10, 10, ts.width + 20, ts.height + baseline + 20);
        cv::rectangle(frame_bgr, bg, bg_color, cv::FILLED);
        cv::putText(frame_bgr, status_text, cv::Point(bg.x + 10, bg.y + bg.height - 10), cv::FONT_HERSHEY_SIMPLEX, 0.9, cv::Scalar(0, 0, 0), 2);

        // Рисуем bounding boxes для всех оставшихся после NMS детекций
        for (int idx : keep) {
            const auto &det = detections[idx];
            const cv::Rect box(static_cast<int>(det.box_xyxy.x), static_cast<int>(det.box_xyxy.y),
                              static_cast<int>(det.box_xyxy.width), static_cast<int>(det.box_xyxy.height));
            cv::rectangle(frame_bgr, box, cv::Scalar(0, 255, 0), 2);

            // Подпись с confidence над боксом
            const std::string label = "person " + std::to_string(det.confidence).substr(0, 4);
            const cv::Size label_ts = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.6, 1, nullptr);
            cv::Rect label_bg(box.x, std::max(0, box.y - label_ts.height - 6), label_ts.width + 6, label_ts.height + 6);
            cv::rectangle(frame_bgr, label_bg, cv::Scalar(0, 255, 0), cv::FILLED);
            cv::putText(frame_bgr, label, cv::Point(label_bg.x + 3, label_bg.y + label_bg.height - 4), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 0), 1);
        }

        // Показываем кадр в окне.
        cv::imshow("YOLO person", frame_bgr);

        // waitKey:
        // - даёт OpenCV обработать события окна,
        // - возвращает код нажатой клавиши (если была).
        // Здесь выходим по ESC (27).
        if (cv::waitKey(1) == 27) {
            break;
        }
    }

    // Освобождаем ресурсы.
    cap.release();
    cv::destroyAllWindows();
    return 0;
}
