#include <http_client.h>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include <algorithm>
#include <cstring>
#include <cmath>
#include <chrono>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

namespace {

struct LetterboxParams {
    float scale = 1.0F;
    float pad_x = 0.0F;
    float pad_y = 0.0F;
};

cv::Mat Letterbox640(const cv::Mat& bgr, LetterboxParams& params) {
    const int target_w = 640;
    const int target_h = 640;

    const int src_w = bgr.cols;
    const int src_h = bgr.rows;

    const float r_w = static_cast<float>(target_w) / static_cast<float>(src_w);
    const float r_h = static_cast<float>(target_h) / static_cast<float>(src_h);
    const float r = std::min(r_w, r_h);
    params.scale = r;

    const int new_w = static_cast<int>(std::round(static_cast<float>(src_w) * r));
    const int new_h = static_cast<int>(std::round(static_cast<float>(src_h) * r));

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
    cv::copyMakeBorder(resized, out, pad_top, pad_bottom, pad_left, pad_right, cv::BORDER_CONSTANT,
                       cv::Scalar(114, 114, 114));
    return out;
}

// Подготовка кадра под Ultralytics YOLO (ONNX):
// - letterbox до 640x640
// - BGR -> RGB
// - float32 [0..1]
// - CHW
std::vector<float> PrepareYoloInput(const cv::Mat& frame_bgr, LetterboxParams& lb) {
    cv::Mat lb_bgr = Letterbox640(frame_bgr, lb);

    cv::Mat lb_rgb;
    cv::cvtColor(lb_bgr, lb_rgb, cv::COLOR_BGR2RGB);
    lb_rgb.convertTo(lb_rgb, CV_32F, 1.0 / 255.0);

    std::vector<cv::Mat> chw(3);
    cv::split(lb_rgb, chw);

    std::vector<float> input_data;
    input_data.resize(3 * 640 * 640);
    const size_t plane = 640 * 640;
    for (size_t c = 0; c < 3; ++c) {
        std::memcpy(input_data.data() + c * plane, chw[c].ptr<float>(), plane * sizeof(float));
    }
    return input_data;
}

} // namespace

int main() {
    // Хардкод под локально поднятый Triton:
    //   docker run ... -p 8000:8000 ... tritonserver --model-repository=/models
    // и модель из examples/Triton/models/yolo_model/config.pbtxt (name: flap_detector)
    const std::string triton_url = "localhost:8000";
    const std::string model_name = "yolo_model";
    const std::string input_name = "images";
    const std::string output_name = "output0";
    const std::string input_dtype = "FP32";
    constexpr int target_class_id = 0; // COCO: person
    constexpr float conf_threshold = 0.35F;

    std::unique_ptr<triton::client::InferenceServerHttpClient> client;
    {
        auto err = triton::client::InferenceServerHttpClient::Create(&client, triton_url, false);
        if (!err.IsOk()) {
            std::cerr << "Не удалось создать Triton HTTP client: " << err.Message() << "\n";
            return 1;
        }
    }

    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        std::cerr << "Не удалось открыть камеру 0\n";
        return 1;
    }

    cv::namedWindow("Triton", cv::WINDOW_NORMAL);

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

        const bool should_infer = (frame_index % 10 == 1);

        std::int64_t preprocess_ms = 0;
        std::int64_t infer_ms = 0;
        std::int64_t post_ms = 0;

        if (should_infer) {
            const auto t_pre_start = std::chrono::steady_clock::now();
            LetterboxParams lb;
            const std::vector<float> input_data = PrepareYoloInput(frame, lb);
            const auto t_pre_end = std::chrono::steady_clock::now();
            preprocess_ms = to_ms(t_pre_end - t_pre_start);

            triton::client::InferInput* infer_input_raw = nullptr;
            auto input_err = triton::client::InferInput::Create(
                &infer_input_raw, input_name, {1, 3, 640, 640}, input_dtype);
            if (!input_err.IsOk()) {
                std::cerr << "InferInput::Create: " << input_err.Message() << "\n";
                return 1;
            }
            std::shared_ptr<triton::client::InferInput> infer_input(infer_input_raw);

            auto append_err = infer_input->AppendRaw(
                reinterpret_cast<const uint8_t*>(input_data.data()), input_data.size() * sizeof(float));
            if (!append_err.IsOk()) {
                std::cerr << "InferInput::AppendRaw: " << append_err.Message() << "\n";
                return 1;
            }

            triton::client::InferRequestedOutput* output_raw = nullptr;
            auto out_err =
                triton::client::InferRequestedOutput::Create(&output_raw, output_name);
            if (!out_err.IsOk()) {
                std::cerr << "InferRequestedOutput::Create: " << out_err.Message() << "\n";
                return 1;
            }
            std::shared_ptr<triton::client::InferRequestedOutput> output(output_raw);

            triton::client::InferOptions options(model_name);
            triton::client::InferResult* response_raw = nullptr;
            const auto t_infer_start = std::chrono::steady_clock::now();
            auto infer_err =
                client->Infer(&response_raw, options, {infer_input.get()}, {output.get()});
            const auto t_infer_end = std::chrono::steady_clock::now();
            if (!infer_err.IsOk()) {
                std::cerr << "client->Infer: " << infer_err.Message() << "\n";
                return 1;
            }
            std::unique_ptr<triton::client::InferResult> result(response_raw);
            infer_ms = to_ms(t_infer_end - t_infer_start);

            const uint8_t* output_bytes = nullptr;
            size_t output_byte_size = 0;
            auto raw_err = result->RawData(output_name, &output_bytes, &output_byte_size);
            if (!raw_err.IsOk()) {
                std::cerr << "InferResult::RawData: " << raw_err.Message() << "\n";
                return 1;
            }
            // YOLO11n @ 640: [1, 84, 8400] => 84*8400 float
            constexpr int channels = 84; // 4 + nc, nc=80 для COCO
            constexpr int num = 8400;
            constexpr size_t expected = static_cast<size_t>(channels) * static_cast<size_t>(num) * sizeof(float);
            if (output_byte_size != expected) {
                std::cerr << "Некорректный размер output0: " << output_byte_size
                          << " bytes (ожидалось " << expected << ")\n";
                return 1;
            }

            const float* out = reinterpret_cast<const float*>(output_bytes);

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
                float x1 = best_cx - best_w * 0.5F;
                float y1 = best_cy - best_h * 0.5F;
                float x2 = best_cx + best_w * 0.5F;
                float y2 = best_cy + best_h * 0.5F;

                x1 = (x1 - lb.pad_x) / lb.scale;
                y1 = (y1 - lb.pad_y) / lb.scale;
                x2 = (x2 - lb.pad_x) / lb.scale;
                y2 = (y2 - lb.pad_y) / lb.scale;

                x1 = std::clamp(x1, 0.0F, static_cast<float>(frame.cols - 1));
                y1 = std::clamp(y1, 0.0F, static_cast<float>(frame.rows - 1));
                x2 = std::clamp(x2, 0.0F, static_cast<float>(frame.cols - 1));
                y2 = std::clamp(y2, 0.0F, static_cast<float>(frame.rows - 1));

                cached_box = cv::Rect(static_cast<int>(x1), static_cast<int>(y1),
                                      static_cast<int>(std::max(0.0F, x2 - x1)),
                                      static_cast<int>(std::max(0.0F, y2 - y1)));
                has_cached_box = true;
            } else {
                has_cached_box = false;
            }

            post_ms = to_ms(std::chrono::steady_clock::now() - t_infer_end);
        }

        const std::string status = cached_person ? "PERSON PRESENT" : "NO PERSON";
        const cv::Scalar bg = cached_person ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 0, 255);
        const cv::Scalar fg = cv::Scalar(0, 0, 0);
        int baseline = 0;
        const cv::Size ts = cv::getTextSize(status, cv::FONT_HERSHEY_SIMPLEX, 0.9, 2, &baseline);
        cv::Rect rect(10, 10, ts.width + 20, ts.height + baseline + 20);
        cv::rectangle(frame, rect, bg, cv::FILLED);
        cv::putText(frame, status, cv::Point(rect.x + 10, rect.y + rect.height - 10),
                    cv::FONT_HERSHEY_SIMPLEX, 0.9, fg, 2);

        if (cached_person && has_cached_box) {
            cv::rectangle(frame, cached_box, cv::Scalar(0, 255, 0), 2);
            cv::putText(frame, "conf=" + std::to_string(cached_conf).substr(0, 4),
                        cv::Point(cached_box.x, std::max(0, cached_box.y - 5)), cv::FONT_HERSHEY_SIMPLEX, 0.6,
                        cv::Scalar(0, 255, 0), 2);
        }

        const auto t_loop_end = std::chrono::steady_clock::now();
        const auto capture_ms = to_ms(t_after_capture - t_loop_start);
        const auto total_ms = to_ms(t_loop_end - t_loop_start);
        std::cout << "кадр " << frame_index
                  << " захват=" << capture_ms << "мс"
                  << " препроцесс=" << preprocess_ms << "мс"
                  << " инференс=" << infer_ms << "мс"
                  << " пост=" << post_ms << "мс"
                  << " всего=" << total_ms << "мс"
                  << (should_infer ? " [инференс]" : " [пропуск]")
                  << std::endl;

        cv::imshow("Triton", frame);
        const int key = cv::waitKey(1);
        if (key == 27 || key == 'q' || key == 'Q') {
            break;
        }
    }

    return 0;
}