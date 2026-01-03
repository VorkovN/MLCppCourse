#include <http_client.h>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include <algorithm>
#include <chrono>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

// Пример использования Triton Inference Server для инференса YOLO через HTTP API.
// Triton позволяет вынести инференс в отдельный сервис, что удобно для масштабирования и управления ресурсами.

struct LetterboxParams {
    float scale = 1.0F;  // Масштаб, применённый к изображению при letterbox
    float pad_x = 0.0F; // Смещение паддинга по X (для обратного пересчёта координат)
    float pad_y = 0.0F; // Смещение паддинга по Y (для обратного пересчёта координат)
};

// Letterbox до фиксированного размера 640x640: масштабирует с сохранением пропорций,
// недостающие пиксели заполняет серым (114, 114, 114) — стандартный цвет паддинга в Ultralytics YOLO.
cv::Mat Letterbox640(const cv::Mat& bgr, LetterboxParams& params) {
    constexpr int target_w = 640;
    constexpr int target_h = 640;
    const float r = std::min(static_cast<float>(target_w) / bgr.cols, static_cast<float>(target_h) / bgr.rows);
    params.scale = r;

    const int new_w = static_cast<int>(std::round(bgr.cols * r));
    const int new_h = static_cast<int>(std::round(bgr.rows * r));

    cv::Mat resized;
    cv::resize(bgr, resized, cv::Size(new_w, new_h));

    const int pad_w = target_w - new_w;
    const int pad_h = target_h - new_h;
    const int pad_left = pad_w / 2;
    const int pad_right = pad_w - pad_left;
    const int pad_top = pad_h / 2;
    const int pad_bottom = pad_h - pad_top;

    params.pad_x = static_cast<float>(pad_left);
    params.pad_y = static_cast<float>(pad_top);

    cv::Mat out;
    cv::copyMakeBorder(resized, out, pad_top, pad_bottom, pad_left, pad_right, cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));
    return out;
}

// Подготовка кадра под Ultralytics YOLO: letterbox до 640x640, BGR->RGB, нормализация [0..1], преобразование HWC->CHW.
// Возвращает готовый буфер для отправки в Triton.
std::vector<float> PrepareYoloInput(const cv::Mat& frame_bgr, LetterboxParams& lb) {
    cv::Mat lb_bgr = Letterbox640(frame_bgr, lb);
    cv::Mat lb_rgb;
    cv::cvtColor(lb_bgr, lb_rgb, cv::COLOR_BGR2RGB);
    
    // Нормализация: uint8 [0..255] -> float32 [0..1]
    lb_rgb.convertTo(lb_rgb, CV_32F, 1.0 / 255.0);

    // HWC -> CHW: разбиваем на каналы и копируем плоскости подряд
    std::vector<cv::Mat> chw(3);
    cv::split(lb_rgb, chw);

    std::vector<float> input_data(3 * 640 * 640);
    constexpr size_t plane = 640 * 640;
    for (size_t c = 0; c < 3; ++c) {
        std::memcpy(input_data.data() + c * plane, chw[c].ptr<float>(), plane * sizeof(float));
    }
    return input_data;
}


int main() {
    // Параметры подключения к Triton Inference Server
    // Ожидается, что сервер запущен локально: docker run ... -p 8000:8000 ... tritonserver --model-repository=/models
    const std::string triton_url = "localhost:8000";
    const std::string model_name = "yolo_model";
    const std::string input_name = "images";
    const std::string output_name = "output0";
    constexpr int target_class_id = 0; // COCO: person
    constexpr float conf_threshold = 0.35F;

    // Создаём HTTP клиента для подключения к Triton Inference Server
    std::unique_ptr<triton::client::InferenceServerHttpClient> client;
    auto err = triton::client::InferenceServerHttpClient::Create(&client, triton_url, false);
    if (!err.IsOk()) {
        std::cerr << "Не удалось создать Triton HTTP client: " << err.Message() << "\n";
        return 1;
    }

    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        std::cerr << "Не удалось открыть камеру\n";
        return 1;
    }

    cv::namedWindow("Triton", cv::WINDOW_NORMAL);

    // Кеширование результатов: чтобы не нагружать сеть/сервер, делаем инференс не на каждом кадре.
    // Остальные кадры отображаем с последним закешированным результатом.
    int frame_index = 0;
    bool cached_person = false;
    float cached_conf = 0.0F;
    cv::Rect cached_box;
    bool has_cached_box = false;

    auto to_ms = [](auto duration) {
        return std::chrono::duration_cast<std::chrono::milliseconds>(duration).count();
    };

    while (true) {
        const auto t_loop_start = std::chrono::steady_clock::now();

        cv::Mat frame;
        if (!cap.read(frame) || frame.empty()) {
            break;
        }
        ++frame_index;

        const auto t_after_capture = std::chrono::steady_clock::now();
        // Инференс делаем только на каждом 10-м кадре, чтобы снизить нагрузку на сеть и сервер
        const bool should_infer = (frame_index % 10 == 1);

        std::int64_t preprocess_ms = 0;
        std::int64_t infer_ms = 0;
        std::int64_t post_ms = 0;

        if (should_infer) {
            // --- Препроцессинг: подготовка кадра для модели ---
            const auto t_pre_start = std::chrono::steady_clock::now();
            LetterboxParams lb;
            const std::vector<float> input_data = PrepareYoloInput(frame, lb);
            preprocess_ms = to_ms(std::chrono::steady_clock::now() - t_pre_start);

            // Описываем входной тензор Triton: имя, форма [batch, channels, height, width], тип данных FP32
            triton::client::InferInput* infer_input_raw = nullptr;
            auto input_err = triton::client::InferInput::Create(&infer_input_raw, input_name, {1, 3, 640, 640}, "FP32");
            if (!input_err.IsOk()) {
                std::cerr << "InferInput::Create: " << input_err.Message() << "\n";
                return 1;
            }
            std::shared_ptr<triton::client::InferInput> infer_input(infer_input_raw);

            // Передаём сырые байты float32 в запрос
            auto append_err = infer_input->AppendRaw(reinterpret_cast<const uint8_t*>(input_data.data()), input_data.size() * sizeof(float));
            if (!append_err.IsOk()) {
                std::cerr << "InferInput::AppendRaw: " << append_err.Message() << "\n";
                return 1;
            }

            // Описываем запрашиваемый выход модели
            triton::client::InferRequestedOutput* output_raw = nullptr;
            auto out_err = triton::client::InferRequestedOutput::Create(&output_raw, output_name);
            if (!out_err.IsOk()) {
                std::cerr << "InferRequestedOutput::Create: " << out_err.Message() << "\n";
                return 1;
            }
            std::shared_ptr<triton::client::InferRequestedOutput> output(output_raw);

            // Отправляем запрос на инференс в Triton
            triton::client::InferOptions options(model_name);
            triton::client::InferResult* response_raw = nullptr;
            const auto t_infer_start = std::chrono::steady_clock::now();
            auto infer_err = client->Infer(&response_raw, options, {infer_input.get()}, {output.get()});
            const auto t_infer_end = std::chrono::steady_clock::now();
            infer_ms = to_ms(t_infer_end - t_infer_start);
            if (!infer_err.IsOk()) {
                std::cerr << "client->Infer: " << infer_err.Message() << "\n";
                return 1;
            }
            std::unique_ptr<triton::client::InferResult> result(response_raw);

            // Забираем raw output: Triton возвращает byte-buffer, приводим к float*
            const uint8_t* output_bytes = nullptr;
            size_t output_byte_size = 0;
            auto raw_err = result->RawData(output_name, &output_bytes, &output_byte_size);
            if (!raw_err.IsOk()) {
                std::cerr << "InferResult::RawData: " << raw_err.Message() << "\n";
                return 1;
            }

            // YOLO11n @ 640: выход [1, 84, 8400] => 84*8400 float
            // 84 = 4 координаты (cx, cy, w, h) + 80 классов COCO
            constexpr int channels = 84;
            constexpr int num = 8400;
            constexpr size_t expected = static_cast<size_t>(channels) * static_cast<size_t>(num) * sizeof(float);
            if (output_byte_size != expected) {
                std::cerr << "Некорректный размер output0: " << output_byte_size << " bytes (ожидалось " << expected << ")\n";
                return 1;
            }

            const float* out = reinterpret_cast<const float*>(output_bytes);

            // Упрощённый постпроцессинг: вместо NMS берём лучший score по нужному классу.
            // Это минимально рабочий вариант для демонстрации клиента Triton.
            float best_conf = 0.0F;
            float best_cx = 0.0F, best_cy = 0.0F, best_w = 0.0F, best_h = 0.0F;
            for (int i = 0; i < num; ++i) {
                const float score = out[i + (4 + target_class_id) * num];
                if (score > best_conf) {
                    best_conf = score;
                    best_cx = out[i + 0 * num];
                    best_cy = out[i + 1 * num];
                    best_w = out[i + 2 * num];
                    best_h = out[i + 3 * num];
                }
            }

            cached_conf = best_conf;
            cached_person = best_conf >= conf_threshold;

            if (cached_person) {
                // Переводим cxcywh -> xyxy и пересчитываем координаты из letterbox обратно в исходный кадр
                float x1 = (best_cx - best_w * 0.5F - lb.pad_x) / lb.scale;
                float y1 = (best_cy - best_h * 0.5F - lb.pad_y) / lb.scale;
                float x2 = (best_cx + best_w * 0.5F - lb.pad_x) / lb.scale;
                float y2 = (best_cy + best_h * 0.5F - lb.pad_y) / lb.scale;

                // Клипуем координаты в границы исходного кадра
                x1 = std::clamp(x1, 0.0F, static_cast<float>(frame.cols - 1));
                y1 = std::clamp(y1, 0.0F, static_cast<float>(frame.rows - 1));
                x2 = std::clamp(x2, 0.0F, static_cast<float>(frame.cols - 1));
                y2 = std::clamp(y2, 0.0F, static_cast<float>(frame.rows - 1));

                cached_box = cv::Rect(static_cast<int>(x1), static_cast<int>(y1),
                                      static_cast<int>(std::max(0.0F, x2 - x1)), static_cast<int>(std::max(0.0F, y2 - y1)));
                has_cached_box = true;
            } else {
                has_cached_box = false;
            }

            const auto t_post_end = std::chrono::steady_clock::now();
            post_ms = to_ms(t_post_end - t_infer_end);
        }

        // Визуализация: отображаем закешированный результат на каждом кадре
        const std::string status = cached_person ? "PERSON PRESENT" : "NO PERSON";
        const cv::Scalar bg = cached_person ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 0, 255);
        int baseline = 0;
        const cv::Size ts = cv::getTextSize(status, cv::FONT_HERSHEY_SIMPLEX, 0.9, 2, &baseline);
        cv::Rect rect(10, 10, ts.width + 20, ts.height + baseline + 20);
        cv::rectangle(frame, rect, bg, cv::FILLED);
        cv::putText(frame, status, cv::Point(rect.x + 10, rect.y + rect.height - 10), cv::FONT_HERSHEY_SIMPLEX, 0.9, cv::Scalar(0, 0, 0), 2);

        // Рисуем bounding box, если детекция найдена
        if (cached_person && has_cached_box) {
            cv::rectangle(frame, cached_box, cv::Scalar(0, 255, 0), 2);
            cv::putText(frame, "conf=" + std::to_string(cached_conf).substr(0, 4),
                        cv::Point(cached_box.x, std::max(0, cached_box.y - 5)), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 2);
        }

        // Выводим метрики производительности
        const auto t_loop_end = std::chrono::steady_clock::now();
        const auto capture_ms = to_ms(t_after_capture - t_loop_start);
        const auto total_ms = to_ms(t_loop_end - t_loop_start);
        std::cout << "кадр " << frame_index << " захват=" << capture_ms << "мс препроцесс=" << preprocess_ms << "мс инференс=" << infer_ms << "мс пост=" << post_ms << "мс всего=" << total_ms << "мс" << (should_infer ? " [инференс]" : " [пропуск]") << std::endl;

        // Показываем кадр в окне.
        cv::imshow("Triton", frame);

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