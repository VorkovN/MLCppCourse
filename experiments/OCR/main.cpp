#include <opencv2/opencv.hpp>
#include <tesseract/baseapi.h>
#include <iostream>
#include <string>

int main(int argc, char** argv) {
    tesseract::TessBaseAPI ocr;
    // LSTM-движок обычно заметно лучше различает похожие глифы (i/l/1) на печатном тексте
    if (ocr.Init(nullptr, "eng", tesseract::OEM_LSTM_ONLY) != 0) {
        std::cerr << "Ошибка: не удалось инициализировать Tesseract (нужен eng.traineddata и корректный TESSDATA_PREFIX)" << std::endl;
        return -1;
    }

    ocr.SetPageSegMode(tesseract::PSM_AUTO);
    // Помогаем Tesseract корректнее оценивать размер шрифта на изображениях без DPI
    ocr.SetVariable("user_defined_dpi", "300");

    (void)argc;
    (void)argv;
    const std::string imagePath = "./test_image.jpg";
    cv::Mat image = cv::imread(imagePath, cv::IMREAD_COLOR);
    if (image.empty()) {
        std::cerr << "Ошибка: не удалось прочитать картинку: " << imagePath << std::endl;
        return -1;
    }

    cv::Mat gray;
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    // Для текста на скриншотах/фото Otsu + сильный blur часто "ломает" тонкие штрихи (i превращается в 1).
    // Более мягкий пайплайн: апскейл + адаптивный порог.
    cv::resize(gray, gray, cv::Size(), 2.0, 2.0, cv::INTER_CUBIC);
    cv::adaptiveThreshold(gray, gray, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY, 35, 11);

    ocr.SetImage(gray.data, gray.cols, gray.rows, 1, static_cast<int>(gray.step));
    std::string text = ocr.GetUTF8Text();

    std::cout << "Файл: " << imagePath << std::endl;
    std::cout << "Распознанный текст:" << std::endl;
    std::cout << text << std::endl;

    return 0;
}