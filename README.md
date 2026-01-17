## План курса про инференс ML-моделей на C++

<a id="toc"></a>
### Оглавление

- [Введение](#intro)
- [OpenCV](#opencv)
  - [CMake](#opencv-cmake)
  - [Conan](#opencv-conan)
- [MediaPipe](#mediapipe)
  - [MediaPipe С++](#mediapipe-conan)
  - [MediaPipe Pybind11](#mediapipe-pybind11)
- [YOLO](#yolo)
  - [ONNX Runtime](#yolo-onnxruntime)
  - [Triton Inference Server](#yolo-triton)
- [OCR](#ocr)
  - [Tesseract](#ocr-tesseract)

---

<a id="intro"></a>
## Введение

Я 5 лет работаю C++‑разработчиком, последний год я много работал с внедрением машинного обучения. И часто реализация инференса модели оказывается сложнее реализации обучения.

Обычно модели обучаются на Python, где зависимости подключаются сравнительно просто. Но что делать, если проект, в котором должен происходить инференс, написан на C++ и собирается под разные платформы? Например, десктоп нужно собрать под linux, mac_arm, mac_x86, windows. В таком случае кроссплатформенная сборка библиотек для инференса становится задачей не менее сложной, чем обучение модели.

В этом небольшом курсе я хочу рассказать о базовых библиотеках и инструментах для инференса ML‑моделей и главное показать готовые решения самых частых задач. Основной идеей этого курса было сэкономить время разработчикам, которые только столкнулись с ML в C++‑проектах. Весь код — упрощённая, но рабочая версия. Предлагаю ее использовать как универсальный бейзлайн, а дальше адаптировать под вашу бизнес‑логику и стиль оформления.

Я не буду рассказывать об архитектурах моделей и их обучении, этот курс только лишь о запуске моделей в проде. 

В данном репозитории я выложил всю кодовую базу проекта: https://github.com/VorkovN/MLCppCourse Так что можете запустить код и убедиться, что он работает, прежде чем интегрировать какие-то куски кода в ваши проекты. Курс ориентирован на сборку под **macOS arm64** и **Linux x86_64**. Под Windows подход в целом тот же, но больше времени уйдёт на настройку зависимостей, настройки CMake и Conan‑профилей. Также замечу, что весь код в этом репозитории собирается в режиме Debug. В Release он тоже будет работать, если соберете релизный conan профиль, о conan я расскажу далее. Debug режим был выбран с целью возможности дебага в этом обучающем проекте. Так что не пугайтесь, что инференс в некоторых ML библиотеках будет медленный в режиме Debug. В Release версиях инференс ускоряется, как правило, в 2-10 раз!

---

<a id="opencv"></a>
## OpenCV

![Пример детекции окружностей с помощью OpenCV](https://raw.githubusercontent.com/VorkovN/MLCppCourse/main/images/OpenCV.png)

#### Темы урока

- [CMake](#opencv-cmake) — генератор сборки, targets, зависимости, типы сборки.
- [Conan](#opencv-conan) — пакетный менеджер для C++, профили, lock-файлы, интеграция с CMake.

#### Актуальность

[OpenCV](https://opencv.org) — это универсальный инструмент для всего, что связано с изображениями в C++. Начиная от работы с видеопотоками, заканчивая ML‑интеграциями, где OpenCV почти неизбежно появляется как инструмент для препроцессинга и постпроцессинга.

Где она встречается чаще всего:

- промышленная автоматизация,
- робототехника,
- медицина,
- видеонаблюдение.

OpenCV — отличный старт для курса, потому что на её примере удобно разобрать, как собрать проект и как подключать зависимости.

#### Теория

- **OpenCV**: (Open Source Computer Vision Library) — библиотека для обработки изображений/видео. Основной язык — C++, также есть биндинги для Python/Java и др.
- **Ключевая абстракция**: `cv::Mat` — это «картинка как матрица» (2D/3D массив), но не просто `uint8[]`. Внутри важно помнить про:
  - **layout в памяти**: непрерывная ли память (`isContinuous()`) и какой **шаг строки** (stride/step);
  - **каналы**: сколько их и в каком порядке (в OpenCV по умолчанию часто **BGR**, а не RGB);
  - **тип данных**: `uint8`, `float32` и т. п. (это влияет и на точность, и на скорость);
  - **владение памятью**: `cv::Mat` может быть «видом» на чужой буфер, поэтому копирование (`clone()`) и slicing нужно делать осознанно.
- **Модули OpenCV (в общем виде)**:
  - `core` — фундамент: `cv::Mat`, базовые типы (`Point`/`Size`/`Rect`/`Scalar`), операции над массивами, базовая математика.
  - `imgcodecs` — чтение/запись изображений.
  - `highgui` — окна и ввод (клавиатура/мышь).
  - `imgproc` — ключевой модуль обработки изображений: преобразования цветовых пространств, фильтрация, пороги, морфология, контуры, геометрические преобразования.
  - `videoio` — работа с видео и камерами.
  - `video` — анализ движения (трекинг, оптический поток).
  - `features2d` — ключевые точки/дескрипторы и матчинги.
  - `calib3d` — калибровка камер и 3D‑геометрия.
  - `dnn` — инференс нейросетей (CPU/OpenCL/CUDA — зависит от сборки). Форматы: ONNX, TensorFlow, Caffe.

---

<a id="opencv-cmake"></a>
### CMake

##### Актуальность

[CMake](https://cmake.org) — это язык описания графа зависимостей проекта. В ML‑интеграции вы подключаете множество библиотек. Чаще всего сборка этих библиотек изначально написана на CMake, сейчас это самый популярный генератор билд системы. Потому начиная любой не БигТех проект намного проще используя CMake.

##### Теория

CMake — генератор build‑системы: вы описываете цели (targets), зависимости и свойства, а CMake генерирует проект для Ninja/Makefile/Xcode/Visual Studio.

База, без которой в продовой интеграции быстро становится некомфортно:

- **Targets как центр мира**
  - **Что это**: в CMake вы описываете не «команды компилятора», а **цели** (targets) — исполняемые файлы и библиотеки.
  - **Как выглядит**:
    - `add_executable(app ...)` / `add_library(lib ...)` — создаём цель: испольняемый файл с main() или библиотека;
    - `find_package(OpenCV REQUIRED)` — находим внешнюю зависимость, которую создали через `add_executable/add_library`;
    - `target_link_libraries(app PRIVATE OpenCV::opencv)` — связываем цели.
  - **Почему важно**: когда всё оформлено через targets, CMake сам распространяет include‑пути, зависимости и настройки линковки (в рамках правил видимости).

- **Видимость: `PRIVATE` / `PUBLIC` / `INTERFACE`**
  - **Что это**: правило «кто видит зависимость».
  - **Смысл**:
    - `PRIVATE` — зависимость нужна **только внутри** текущей цели;
    - `PUBLIC` — нужна и **внутри**, и **потребителям** (кто линкуется с вашей библиотекой);
    - `INTERFACE` — нужна **только потребителям** (типично для header-only или «наследуемых» настроек).
  - **Как думать**: это способ честно описать API вашей библиотеки: что является частью публичного контракта, а что — внутренности.

- **Тип сборки: `Debug` / `Release` / `RelWithDebInfo`**
  - **Что это**: набор разных флагов компиляции/линковки.
  - **Чем отличаются**:
    - `Debug` — отладочная информация и обычно минимальные оптимизации;
    - `Release` — оптимизации и минимальная отладка;
    - `RelWithDebInfo` — компромисс: оптимизации + отладочные символы.
  - **Почему важно**: тип сборки влияет на производительность и на совместимость окружения сборки зависимостей (в том числе когда зависимости собираются отдельным инструментом).

- **Toolchain**
  - **Что это**: файл, который задаёт «каким компилятором/под какую платформу/с какими путями» конфигурируется проект. Например, в этом курсе мы в каждом проекте будем подключать conan_toolchain.cmake
  - **Как выглядит**: обычно это один `*.cmake`, который подключается при конфигурации (`-DCMAKE_TOOLCHAIN_FILE=...`).
  - **Почему важно**: toolchain фиксирует окружение и делает конфигурацию предсказуемой (особенно в кроссплатформенных сценариях и когда зависимости приходят через менеджер пакетов).

---

<a id="opencv-conan"></a>
### Conan

##### Актуальность

Пакетный менеджер [Conan](https://conan.io) — инструмент для управления внешними зависимостями в C++‑проектах, особенно когда проект нужно собирать под несколько платформ/компиляторов.

Альтернатива Conan — собирать и раскладывать зависимости вручную (скриптами или руками). На практике это быстро приводит к тому, что сборка перестаёт быть воспроизводимой и предсказуемой.

##### Теория

Conan собирает или скачивает **пакеты** (готовые бинарники библиотек) под конкретную конфигурацию: `OS/arch/compiler/build_type/options`. Дальше эти пакеты кэшируются локально и (опционально) публикуются в удалённый Conan‑remote (например, в корпоративный GitLab/Artifactory/Nexus или свой сервер), чтобы на CI можно было получать те же самые артефакты.

Ключевые сущности:

- **Recipe (рецепт пакета)**: описание того, *как собрать* библиотеку и какие у неё зависимости/настройки. Обычно это `conanfile.py` (полноценный рецепт) или `conanfile.txt` (упрощённое перечисление зависимостей для приложения, а также опций: shared/static, включение/выключение фич).
- **Package (пакет/бинарник)**: результат сборки recipe под конкретные `settings/options` (условно: «собранная библиотека под нужную ОС/архитектуру/компилятор/тип сборки»).
- **profiles**: файл, который фиксирует OS/arch/compiler/build_type и окружение компилятора. Профиль нужен, чтобы сборка была повторяемой на разных машинах.
- **build/host contexts**:
  - `build` — где выполняется сборка (инструменты, компилятор);
  - `host` — где будет запускаться результат.
- **lock‑файл** (`conan.lock`): фиксирует полный граф зависимостей (включая транзитивные), часто разработчики перетаскивают тег библиотеки на новый коммит из-за чего может упасть сборка, избежать этой проблемы как раз и помогает lock файл.
- **remotes и cache**:
  - локальный cache — куда Conan складывает скачанные/собранные пакеты;
  - remote — сервер, откуда пакеты скачиваются и куда публикуются (удобно для CI).
- **интеграция с CMake**: Conan генерирует toolchain и файлы описания зависимостей (например, `conan_toolchain.cmake`), после чего CMake видит библиотеки как обычные `find_package`/targets.

В нашем курсе потребуется понимание некоторых conan команд:
- `conan export` — сохранить самостоятельно написанный рецепт сборки внешней библиотеки в кэш, чтобы при дальнейшей сборке вашего проекта рецепт этой библиотеки подтянулся.
- `conan install` — установить зависимости проекта по `conanfile.txt`/`conanfile.py` под конкретный профиль (и сгенерировать файлы интеграции для CMake).


#### Практика

Полный код проекта можно посмотреть по ссылке: https://github.com/VorkovN/MLCppCourse/blob/main/experiments/OpenCV/main.cpp

Сборка и запуск:

```bash
cmake -DCMAKE_BUILD_TYPE=Debug -G "Unix Makefiles" -B build
cmake --build build --target OpenCV
./build/OpenCV
```

Ключевая идея практики — снять кадр с камеры, подготовить изображение под поиск окружностей и визуализировать результат.

```cpp
cv::VideoCapture camera(0);
if (!camera.isOpened()) {
    std::cerr << "Ошибка: не удалось открыть камеру" << std::endl;
    return -1;
}
camera.set(cv::CAP_PROP_FRAME_WIDTH, 1280);
camera.set(cv::CAP_PROP_FRAME_HEIGHT, 720);
camera.set(cv::CAP_PROP_FPS, 30);
```

Дальше — подготовка кадра и поиск окружностей через HoughCircles:

```cpp
camera >> frame;
cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
cv::GaussianBlur(gray, gray, cv::Size(9, 9), 2.0, 2.0);

std::vector<cv::Vec3f> circles;
cv::HoughCircles(gray, circles, cv::HOUGH_GRADIENT_ALT, 20.0, 10, 10, 0.9, 50, 2000);
```

Если окружности найдены, выбираем самую большую и рисуем её на кадре:

```cpp
auto maxCircle = *std::max_element(circles.begin(), circles.end(),
                                   [](const cv::Vec3f &a, const cv::Vec3f &b) { return a[2] < b[2]; });
const auto center = cv::Point(maxCircle[0], maxCircle[1]);
const auto radius = maxCircle[2];
cv::circle(frame, center, radius, cv::Scalar(255, 0, 0), 2);
cv::circle(frame, center, 3, cv::Scalar(0, 0, 255), -1);
```


#### Задание на усвоение

- Подключить библиотеку Eigen через Conan и с помощью кватернионов повернуть изображение найденной окружности на произвольный угол.

---

<a id="mediapipe"></a>
## MediaPipe

![Визуализация лендмарков лица MediaPipe](https://raw.githubusercontent.com/VorkovN/MLCppCourse/main/images/MediaPipe.png)

<a id="mediapipe-topics"></a>
#### Темы урока

- [Через Conan](#mediapipe-conan) — сборка и подключение MediaPipe как C++ зависимости.
- [Через pybind11](#mediapipe-pybind11) — биндинги C++ ↔ Python и запуск через Python-обвязку.

#### Актуальность

[MediaPipe](https://developers.google.com/mediapipe) — Очень известный фреймворк, предоставляющий незаменимую наиболее детализированную детекцию тела и маски лица (478 лендмарков лица).
Типовые прикладные задачи:
- ключевые точки лица (face mesh) — моргания, мимика, поза головы;
- руки/жесты — управление, интерактив;
- pose estimation — скелет, движение, фитнес и т. п.

#### Теория

**MediaPipe** — фреймворк от Google для построения real-time пайплайнов обработки медиа‑данных (видео/изображения/аудио). Его главная идея: вы описываете **граф обработки**, а дальше движок сам гонит по нему данные и следит за временем.

Базовые термины:

- **Graph** (обычно `.pbtxt`) — описание конвейера: какие шаги выполняются, в каком порядке, как соединены потоки;
- **Calculator** — узел графа (маленькая логическая «функция»): получил пакет → посчитал → отдал дальше;
- **Streams** — каналы передачи данных между calculator‑ами;
- **Packets** — данные в потоках, и важная деталь: у каждого пакета есть **timestamp**.

Почему timestamp важен: MediaPipe — это не «один вызов функции», а система, которая синхронизирует события во времени. Это влияет на latency, стабильность трекинга и даже на то, как вы будете отлаживать баги.

Типовой паттерн внутри готовых пайплайнов:
- **детекция** (тяжёлая модель) запускается редко;
- **трекинг** (лёгкая модель/математика) работает на каждом кадре.

Так получается стабильный FPS и меньше дрожания. Пример для Face Mesh обычно выглядит так: детектор находит лицо → ROI → модель считает 478 точек → сглаживание → лендмарки пользователю.

Библиотека быстрая, точная, многофункциональная. Но сборка на практике: страх и ужас С++ разработчика. Собирается под Bazel, отсутствует в Conan, тянет множество зависимостей, включая TensorFlowLite, собрать который под windows на С++20 испытание не для слабонервных, а конвертировать модель tflite в другой формат не получится. Если встречаетесь с этой библиотекой в первый раз, то на сборку под все десктоп платформы можно смело закладывать неделю. Но рекомендую запустить ИИ агента, который сам подберет подходящий рецепт сборки, чтобы ночные кошмары вас не беспокоили.
Но если вы хотите запустить в проекте медиапайп не для прод версии, а для своих экспериментов, то можно немного упростить себе жизнь, об этом я расскажу в следующем разделе:

<a id="mediapipe-conan"></a>
#### Практика

Полный код проекта можно посмотреть по ссылке: https://github.com/VorkovN/MLCppCourse/blob/main/experiments/Mediapipe/main.cpp

**Mediapipe C++**

Медиапайп библиотеки нет в Conan, поэтому рецепт лежит в `experiments/Mediapipe/conan/mediapipe`.

```bash
conan export experiments/Mediapipe/conan/mediapipe
```

Код делает следующее: открывает камеру, инициализирует `FaceLandmarker`, конвертирует кадр в RGB, передаёт кадр в `DetectForVideo` и рисует лендмарки.

Инициализация FaceLandmarker:

```cpp
auto options = std::make_unique<FaceLandmarkerOptions>();
options->base_options.model_asset_path = model_path;
options->running_mode = VIDEO;
options->num_faces = 1;

auto landmarker_or = FaceLandmarker::Create(std::move(options));
auto landmarker = std::move(landmarker_or.value());
```

Создание `mediapipe::Image` из OpenCV‑кадра и запуск инференса:

```cpp
cv::cvtColor(frame, rgb_frame, cv::COLOR_BGR2RGB);
auto image_frame = std::make_shared<mediapipe::ImageFrame>(
    mediapipe::ImageFormat::SRGB, rgb_frame.cols, rgb_frame.rows,
    mediapipe::ImageFrame::kDefaultAlignmentBoundary);
std::memcpy(image_frame->MutablePixelData(), rgb_frame.data, rgb_frame.rows * rgb_frame.cols * 3);

mediapipe::Image image(image_frame);
auto detection_result_or = landmarker->DetectForVideo(image, timestamp_ms);
```

Отрисовка лендмарков:

```cpp
const auto& face_landmarks = detection_result_or.value().face_landmarks[0].landmarks;
for (const auto& landmark : face_landmarks) {
    int x = static_cast<int>(landmark.x * frame.cols);
    int y = static_cast<int>(landmark.y * frame.rows);
    cv::circle(frame, cv::Point(x, y), 2, cv::Scalar(0, 255, 0), -1);
}
```

---

<a id="mediapipe-pybind11"></a>
### Интеграция через pybind11

##### Актуальность

Иногда честный ответ на вопрос «как быстро вывести фичу в продукт» — это не героически тащить всю интеграцию в C++, а оставить часть логики на Python.

pybind11 помогает сделать гибридный режим:
- C++ отвечает за скорость, доступ к данным, низкоуровневые детали;
- Python — за удобную бизнес‑логику, эксперименты и быстрые правки;
- всё это без отдельного сервиса, протоколов и лишней сетевой латентности.
- Если вы хотите запустить медиапайп в тестовом режиме или у вас горят сроки на разработку прототипа, то pybind позволит запустить медиапайп через питон.

##### Теория

**[pybind11](https://pybind11.readthedocs.io)** — библиотека для Python‑биндингов: вы компилируете C++ код в модуль (`.so`/`.pyd`), а Python импортирует его как обычный пакет.

Если говорить по сути: pybind11 позволяет собрать «слоёный пирог», где Python остаётся удобным уровнем управления, а C++ — уровнем производительности и контроля над данными.

##### Практика

Полный код проекта можно посмотреть по ссылке: https://github.com/VorkovN/MLCppCourse/blob/main/experiments/MediapipePybind/main.cpp

**Mediapipe Pybind**

Команды перед запуском:

```bash
cd experiments/MediapipePybind
./setup_venv.sh
```

Здесь медиапайп поднимается через embedded Python. Схема такая: C++ кадр → numpy → `mediapipe.Image` → `detect_for_video`.

Инициализация embedded Python и создание детектора:

```cpp
py::module_ sys = py::module_::import("sys");
sys.attr("path").attr("insert")(0, MEDIAPIPEPYBIND_VENV_SITE_PACKAGES);

py::module_ mediapipe = py::module_::import("mediapipe");
py::module_ vision = py::module_::import("mediapipe.tasks.python.vision");
py::module_ base_options_module = py::module_::import("mediapipe.tasks.python.core.base_options");

py::object FaceLandmarker = vision.attr("FaceLandmarker");
py::object FaceLandmarkerOptions = vision.attr("FaceLandmarkerOptions");
py::object BaseOptions = base_options_module.attr("BaseOptions");
py::object RunningMode = vision.attr("RunningMode");

py::kwargs base_options_kwargs;
base_options_kwargs["model_asset_path"] = model_path;
py::object base_options = BaseOptions(**base_options_kwargs);

py::kwargs options_kwargs;
options_kwargs["base_options"] = base_options;
options_kwargs["running_mode"] = RunningMode.attr("VIDEO");
options_kwargs["num_faces"] = 1;
py::object options = FaceLandmarkerOptions(**options_kwargs);

landmarker_ = FaceLandmarker.attr("create_from_options")(options);
```

Подготовка кадра и запуск инференса:

```cpp
py::array_t<uint8_t> img_array({frame.rows, frame.cols, 3});
std::memcpy(img_array.mutable_data(), frame.data, frame.rows * frame.cols * 3);

py::module_ mediapipe = py::module_::import("mediapipe");
py::object Image = mediapipe.attr("Image");
py::object ImageFormat = mediapipe.attr("ImageFormat");
py::kwargs image_kwargs;
image_kwargs["image_format"] = ImageFormat.attr("SRGB");
image_kwargs["data"] = img_array;
py::object image = Image(**image_kwargs);

py::object result = landmarker_.attr("detect_for_video")(image, timestamp_ms_);
```

Далее результат разбирается и переводится в пиксельные координаты:

```cpp
py::object first_face = result.attr("face_landmarks").attr("__getitem__")(0);
for (size_t i = 0; i < py::len(first_face); ++i) {
    py::object landmark = first_face.attr("__getitem__")(py::cast(i));
    const auto x = landmark.attr("x").cast<double>();
    const auto y = landmark.attr("y").cast<double>();
    landmarks.emplace_back(static_cast<int>(x * frame.cols),
                           static_cast<int>(y * frame.rows));
}
```

#### Задание на усвоение

- В реальной жизни требуется оценивать яркость освещения лица, чтобы гарантировать стабильность выдачи лендмарков: с помощью OpenCV оценить яркость изображения только по пикселям, принадлежащим лицу (по маске/области, построенной из лендмарков).

---

<a id="yolo"></a>
## YOLO

![Пример детекции объектов YOLO](https://raw.githubusercontent.com/VorkovN/MLCppCourse/main/images/YOLO.png)

<a id="yolo-topics"></a>
#### Темы урока

- [ONNX Runtime](#yolo-onnxruntime) — инференс ONNX-моделей в процессе приложения.
- [Triton Inference Server](#yolo-triton) — инференс как сервис по HTTP/gRPC.

#### Актуальность

[YOLO](https://github.com/ultralytics/ultralytics) — одно из самых популярных семейств моделей для object detection. Её часто выбирают как «первую продовую» модель для детектора: она даёт хороший баланс скорости и качества и имеет огромное количество готовых весов/примеров.

В рамках курса YOLO удобна тем, что на её примере можно разобрать полный цикл инференса: подготовка входа, запуск модели, разбор выходов и получение финальных боксов.

#### Теория

**YOLO** — это one‑stage detector: сеть за один проход выдаёт набор кандидатов на объекты: **координаты бокса + “уверенность” + класс**.

Чтобы понимать, что происходит внутри, полезно держать в голове три уровня:

- **Backbone** — извлекает признаки из изображения (условно «сжимает картинку в информативные карты признаков»).
- **Neck** — объединяет признаки разных масштабов (идея feature pyramid): крупные объекты удобнее ловить на “грубых” картах, мелкие — на “детальных”.
- **Head** — превращает карты признаков в предсказания: координаты боксов и классы (и иногда дополнительные головы, например для масок/позы).

Что модель возвращает на выходе (на уровне идеи):

- **BBox** — координаты прямоугольника объекта (часто либо `xywh` — центр и размеры, либо `xyxy` — углы).
- **Confidence** — число, отражающее уверенность модели.
- **Class** — вероятности/оценки по классам.

Дальше почти всегда идёт **постпроцессинг**: из множества кандидатов нужно получить финальный список детекций. Классический шаг здесь — **NMS (Non‑Maximum Suppression)**: алгоритм, который оставляет наиболее сильные боксы и удаляет “дубликаты”, сильно перекрывающиеся по IoU.

Коротко про NMS: сортируем боксы по confidence, берём самый уверенный и выкидываем все боксы с IoU выше порога; повторяем, пока кандидаты не закончатся. Итог — компактный набор детекций без сильных пересечений.

Отдельно: YOLO‑семейство — это не только детекция. В зависимости от модели/версии могут быть варианты для segmentation/pose/OBB, и тогда к боксу добавляются дополнительные выходы (маска, ключевые точки, угол и т. п.).

---

<a id="yolo-onnxruntime"></a>
### ONNX Runtime

##### Актуальность

- [ONNX Runtime](https://onnxruntime.ai) — универсальный рантайм для инференса. Как правило, модели тренируют в разных фреймворках (PyTorch, TensorFlow, scikit-learn), а в проде хочется один рантайм и предсказуемую интеграцию.

##### Теория

Если упростить до практики, то связка ONNX + ONNX Runtime решает две задачи:

- **[ONNX](https://onnx.ai)** — как мы упаковываем модель в переносимый формат;
- **[ONNX Runtime](https://onnxruntime.ai) (ORT)** — как мы запускаем эту модель в C++ (CPU/GPU/и т. д.) и получаем тензоры на выходе.

- **ONNX**: это описание вычислительного графа (операции + тензоры + веса). У модели есть opset — версия набора операций «каким языком ops написан граф», его важно помнить и учитывать при выборе версии onnx runtime.
- **ONNX Runtime**: Высокопроизводительный движок/инференс-рантайм для выполнения ONNX моделей

- **Execution Provider** — «на чём исполняем граф». Частые варианты:
  - CPU (почти везде),
  - CUDA/TensorRT (на NVIDIA),
  - CoreML (на Apple),
  - DirectML (на Windows).
- **Graph optimization level** — насколько агрессивно ORT оптимизирует граф перед запуском (фьюзинг, констант‑фолдинг и т. п.).

Как это выглядит в коде: вы создаёте `Ort::Session` с `Ort::SessionOptions`, выбираете provider, затем на каждый вызов формируете входные `Ort::Value` (тензоры) и получаете массив выходных `Ort::Value`.

##### Практика

Полный код проекта можно посмотреть по ссылке: https://github.com/VorkovN/MLCppCourse/blob/main/experiments/YOLO/main.cpp

**ONNX Runtime**

Проект демонстрирует полный пайплайн: камера → letterbox → RGB/нормализация → CHW → ORT → декодирование → NMS → визуализация.

Инициализация сессии и чтение формы входа:

```cpp
Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "yolo_person");
Ort::SessionOptions session_options;
session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
Ort::Session session(env, model_path.c_str(), session_options);

auto input_shape = session.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
int64_t input_h = input_shape[2] > 0 ? input_shape[2] : 640;
int64_t input_w = input_shape[3] > 0 ? input_shape[3] : 640;
```

Препроцессинг (letterbox + RGB + нормализация + CHW):

```cpp
LetterboxParams lb;
cv::Mat lb_bgr = Letterbox(frame_bgr, static_cast<int>(input_w), static_cast<int>(input_h), lb);
cv::cvtColor(lb_bgr, lb_rgb, cv::COLOR_BGR2RGB);
lb_rgb.convertTo(lb_rgb, CV_32F, 1.0 / 255.0);
std::vector<cv::Mat> chw(3);
cv::split(lb_rgb, chw);
```

Запуск инференса и извлечение выходного тензора:

```cpp
Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
    memory_info, input_tensor_data.data(), input_tensor_data.size(),
    concrete_input_shape.data(), concrete_input_shape.size());

auto output_tensors = session.Run(Ort::RunOptions{nullptr},
                                  input_names.data(), &input_tensor, 1,
                                  output_names.data(), output_names.size());
const float *out_data = output_tensors.front().GetTensorData<float>();
```

---

<a id="yolo-triton"></a>
### Triton Inference Server

##### Актуальность

- Triton полезен в тех случаях, когда на бэке вам необходимо крутить несколько моделей. Вместо того чтобы разворачивать отдельный инференс под каждую модель, удобно настроить инференс моделей через конфиги тритон сервера.
##### Теория

- **[Triton Inference Server](https://github.com/triton-inference-server/server)** — это инференс‑сервер: вы запускаете его как отдельный сервис, а модели «подкладываете» в специальную папку (**model repository**). Дальше к моделям можно обращаться по HTTP/gRPC.

- **Плюсы: централизованное управление моделями**
  - **Версионирование (A/B, rollback)**: одна и та же модель может иметь несколько версий (папки `1/`, `2/`, `3/`…), и Triton может переключаться между ними. Это удобно для A/B‑сравнений и быстрого отката на предыдущую версию.
  - **Dynamic batching**: Triton умеет автоматически объединять близкие по времени запросы в батч, чтобы эффективнее загружать CPU/GPU (и без необходимости собирать батчи вручную в коде сервиса).
  - **Model ensemble**: можно декларативно описать цепочку «препроцессинг → модель → постпроцессинг» как единый пайплайн (ensemble) и вызывать его как одну модель.
  - **Метрики**: Triton отдаёт метрики (обычно под Prometheus), поэтому проще наблюдать latency/throughput, загрузку и поведение моделей.

- **Минусы**: сетевые накладные расходы и инфраструктура (Docker/Kubernetes).

**Model repository: структура папки моделей**

Минимальная структура обычно такая:

```text
models/
  yolo_model/                 # имя модели (как вы будете к ней обращаться)
    config.pbtxt              # конфиг Triton для этой модели
    1/                        # версия модели
      model.onnx              # файл модели (пример для ONNX)
    2/
      model.onnx
```

Главная идея: **имя папки = имя модели**, а **вложенные числовые папки = версии**. Это и есть базовая «единица версионирования» в Triton.

**`config.pbtxt`: декларативное описание того, *как Triton должен понимать модель*: какие входы/выходы, какие размеры, какой бэкенд, как инстанцировать и (опционально) как батчить.

Минимальный каркас для ONNX‑модели выглядит примерно так:

```text
name: "yolo_model"
platform: "onnxruntime_onnx"
max_batch_size: 0

input [
  { name: "images" data_type: TYPE_FP32 dims: [ 3, 640, 640 ] }
]
output [
  { name: "output0" data_type: TYPE_FP32 dims: [ ... ] }
]
```

Ключевые поля, которые обычно приходится понимать:

- `name`: имя модели (обычно совпадает с именем папки).
- `platform`/`backend`: чем исполнять модель (для ONNX часто `onnxruntime_onnx`).
- `max_batch_size`: 0 означает «без батча»; >0 — модель поддерживает batching (тогда появляется смысл в `dynamic_batching`).
- `input[]` / `output[]`: имена, типы и размеры тензоров.

Если сравнивать:

- ONNX Runtime — напрямую встраивается в код (дает меньшие задержки, проще деплой на клиент);
- Triton — это инференс как отдельный микросервис (удобнее управлять моделями, мониторить, масштабировать).

##### Практика

Полный код проекта можно посмотреть по ссылке: https://github.com/VorkovN/MLCppCourse/blob/main/experiments/Triton/main.cpp

**Triton клиент**

Перед запуском клиента поднимаем сервер:


```bash
docker run --rm --name triton --platform=linux/arm64 -p 8000:8000 \
  -v ./experiments/Triton/models:/models \
  nvcr.io/nvidia/tritonserver:24.10-py3 tritonserver --model-repository=/models
```

Инициализация клиента и подключение к серверу:

```cpp
const std::string triton_url = "localhost:8000";
std::unique_ptr<triton::client::InferenceServerHttpClient> client;
auto err = triton::client::InferenceServerHttpClient::Create(&client, triton_url, false);
```

Препроцессинг (letterbox → RGB → нормализация → CHW):

```cpp
LetterboxParams lb;
const std::vector<float> input_data = PrepareYoloInput(frame, lb);
```

Формирование запроса и запуск инференса:

```cpp
auto input_err = triton::client::InferInput::Create(&infer_input_raw, input_name,
                                                    {1, 3, 640, 640}, "FP32");
infer_input->AppendRaw(reinterpret_cast<const uint8_t*>(input_data.data()),
                       input_data.size() * sizeof(float));

triton::client::InferOptions options(model_name);
auto infer_err = client->Infer(&response_raw, options, {infer_input.get()}, {output.get()});
```

Извлечение raw‑выхода и минимальный постпроцессинг:

```cpp
const uint8_t* output_bytes = nullptr;
size_t output_byte_size = 0;
result->RawData(output_name, &output_bytes, &output_byte_size);

const float* out = reinterpret_cast<const float*>(output_bytes);
// В примере выбираем лучший score по классу и переводим cxcywh -> xyxy
```


#### Задание на усвоение

- Добавить в Triton ещё одну модель (любую на выбор) и проверить, что она корректно отвечает через HTTP или gRPC.

---
<a id="ocr"></a>
## OCR

![Пример работы OCR с Tesseract](https://raw.githubusercontent.com/VorkovN/MLCppCourse/main/images/OCR.png)

<a id="ocr-topics"></a>
#### Темы урока

- [Tesseract](#ocr-tesseract) — OCR-движок, модели `*.traineddata`, предобработка, PSM.

#### Актуальность

[OCR](https://en.wikipedia.org/wiki/Optical_character_recognition) (Optical Character Recognition) нужен, когда требуется автоматически извлекать текст из изображений: скриншотов, сканов, кадров видео, фотографий документов и интерфейсов.

В прикладных задачах OCR часто используется как часть более широкой системы: поиск по скриншотам, валидация отображаемых данных и т. д.

#### Теория

OCR‑пайплайн обычно состоит из четырёх этапов:

- **предобработка**: нормализация размера/контраста, перевод в grayscale, подавление шума, бинаризация, исправление наклона (deskew);
- **детекция текста**: выделение областей, где расположен текст;
- **распознавание**: преобразование области текста в строку символов;
- **постобработка**: склейка строк, нормализация форматов, фильтрация по уверенности, применение правил/словарей под конкретную задачу.

Важно учитывать, что качество OCR сильно зависит от домена данных: скриншоты IDE, таблицы, документы, фото с камеры — это разные условия и для них обычно требуются разные настройки OCR.

---

<a id="ocr-tesseract"></a>
### Tesseract

##### Актуальность

[Tesseract](https://github.com/tesseract-ocr/tesseract) — практичный вариант OCR‑движка, когда требуется:

- простой и воспроизводимый способ интеграции в C++‑проект;
- поддержка нескольких языков через `*.traineddata`;
- возможность управлять параметрами сегментации и качеством через пайплайн предобработки.

##### Теория

- Tesseract — OCR‑движок общего назначения.
- `*.traineddata` — языковые модели. Можно комбинировать языки, коих на выбор более сотни.
- PSM (Page Segmentation Mode) задаёт предположение о структуре страницы (строка/блок/авто‑режим) и влияет на качество сегментации и распознавания.
- Предобработка обычно включает `resize`, `cvtColor` в grayscale, `threshold` (Otsu/adaptive), а также морфологические операции для удаления шума.
- Confidence — оценка уверенности распознавания (часто используется для фильтрации результата и принятия решений в прикладной логике).

#### Практика

Полный код проекта можно посмотреть по ссылке: https://github.com/VorkovN/MLCppCourse/blob/main/experiments/OCR/main.cpp

Инициализация Tesseract и базовые параметры:

```cpp
tesseract::TessBaseAPI ocr;
if (ocr.Init(nullptr, "eng", tesseract::OEM_LSTM_ONLY) != 0) {
    std::cerr << "Ошибка: не удалось инициализировать Tesseract" << std::endl;
    return -1;
}
ocr.SetPageSegMode(tesseract::PSM_AUTO);
ocr.SetVariable("user_defined_dpi", "300");
```

Препроцессинг и передача буфера в Tesseract:

```cpp
cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
cv::resize(gray, gray, cv::Size(), 2.0, 2.0, cv::INTER_CUBIC);
cv::adaptiveThreshold(gray, gray, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C,
                      cv::THRESH_BINARY, 35, 11);

ocr.SetImage(gray.data, gray.cols, gray.rows, 1, gray.step);
std::string text = ocr.GetUTF8Text();
```

#### Задание на усвоение

- Реализовать сервис, который по входящим скриншотам определяет: работает программист или занимается отвлечёнными от задачи делами.
- Вход: набор скринов. Выход: список пар (имя скрина, работает/филонит).
