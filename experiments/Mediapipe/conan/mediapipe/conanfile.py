from conan import ConanFile
from conan.tools.files import copy, chdir, save
from conan.tools.layout import basic_layout
from conan.tools.scm import Git
from conan.tools.cmake import CMake
import os
import shutil
import re
import json

class Mediapipe(ConanFile):
    name = "mediapipe"
    version = "0.10.26"
    package_type = "static-library"
    description = "Google MediaPipe (Tasks API) packaged via CMake wrapper"
    license = "Apache-2.0"
    settings = "os", "arch", "compiler", "build_type"

    generators = ("CMakeToolchain", "CMakeDeps")
    requires = (
        "abseil/20240116.1",
        "eigen/3.4.0",
        "protobuf/3.21.12",
        "glog/0.7.1",
        "flatbuffers/23.5.26",
        "tensorflow-lite/2.15.0",
        "opencv/4.11.0",
    )

    def configure(self):
        # Отключаем ffmpeg для OpenCV, чтобы избежать проблем с зависимостями
        self.options["opencv"].with_ffmpeg = False

    def layout(self):
        basic_layout(self)

    def build_requirements(self):
        # Нужен protoc для генерации *.pb.h (в исходниках mediapipe они не всегда закоммичены).
        self.tool_requires("protobuf/3.21.12")
        # Нужен flatc для генерации metadata_schema_generated.h
        self.tool_requires("flatbuffers/23.5.26")

    def source(self):
        src_dir = os.path.join(self.source_folder, "mediapipe")
        repo_url = "https://gitlab.axoning.com/nikita_vorkov/mediapipe.git"
        ref = f"v{self.version}"


        git = Git(self)
        git.clone(url=repo_url, target=src_dir, args=["--depth", "1"])
        with chdir(self, src_dir):
            git.checkout(ref)

    def build(self):
        # Собираем минимально необходимую часть MediaPipe для Tasks Face Landmarker (CPU).
        src_dir = os.path.join(self.source_folder, "mediapipe")
        wrap_dir = os.path.join(self.build_folder, "cmake-wrap")
        os.makedirs(wrap_dir, exist_ok=True)

        # MediaPipe регистрирует графы/калькуляторы через статическую инициализацию.
        # Для статических библиотек линкер может не вытянуть объектные файлы с регистрацией,
        # если из них не запрашиваются символы напрямую. Поэтому "привязываем" нужные .cc
        # к символам, которые вызываются из FaceLandmarker::Create().
        try:
            face_landmarker_cc = os.path.join(
                src_dir, "mediapipe", "tasks", "cc", "vision", "face_landmarker", "face_landmarker.cc"
            )
            face_landmarker_graph_cc = os.path.join(
                src_dir, "mediapipe", "tasks", "cc", "vision", "face_landmarker", "face_landmarker_graph.cc"
            )
            marker = "ADVANCIFY_CONAN_FORCE_LINK_TASKS_REGISTRY"

            # Обязательные графы/калькуляторы для face_landmarker.task (CPU).
            force_items = [
                ("mediapipe/tasks/cc/vision/face_landmarker/face_landmarker_graph.cc",
                 "advancify_mediapipe_force_link_face_landmarker_graph"),
                ("mediapipe/tasks/cc/vision/face_detector/face_detector_graph.cc",
                 "advancify_mediapipe_force_link_face_detector_graph"),
                ("mediapipe/tasks/cc/vision/face_landmarker/face_landmarks_detector_graph.cc",
                 "advancify_mediapipe_force_link_face_landmarks_detector_graph"),
                ("mediapipe/tasks/cc/components/processors/image_preprocessing_graph.cc",
                 "advancify_mediapipe_force_link_image_preprocessing_graph"),
                ("mediapipe/tasks/cc/components/processors/detection_postprocessing_graph.cc",
                 "advancify_mediapipe_force_link_detection_postprocessing_graph"),
                ("mediapipe/tasks/cc/core/model_resources_calculator.cc",
                 "advancify_mediapipe_force_link_model_resources_calculator"),
                ("mediapipe/calculators/core/flow_limiter_calculator.cc",
                 "advancify_mediapipe_force_link_flow_limiter_calculator"),
                ("mediapipe/calculators/core/real_time_flow_limiter_calculator.cc",
                 "advancify_mediapipe_force_link_real_time_flow_limiter_calculator"),
                ("mediapipe/calculators/core/gate_calculator.cc",
                 "advancify_mediapipe_force_link_gate_calculator"),
                ("mediapipe/calculators/core/pass_through_calculator.cc",
                 "advancify_mediapipe_force_link_pass_through_calculator"),
                ("mediapipe/calculators/core/previous_loopback_calculator.cc",
                 "advancify_mediapipe_force_link_previous_loopback_calculator"),
                ("mediapipe/calculators/core/clip_vector_size_calculator.cc",
                 "advancify_mediapipe_force_link_clip_vector_size_calculator"),
                ("mediapipe/calculators/image/image_properties_calculator.cc",
                 "advancify_mediapipe_force_link_image_properties_calculator"),
                ("mediapipe/calculators/util/association_norm_rect_calculator.cc",
                 "advancify_mediapipe_force_link_association_norm_rect_calculator"),
                ("mediapipe/calculators/util/collection_has_min_size_calculator.cc",
                 "advancify_mediapipe_force_link_collection_has_min_size_calculator"),
                ("mediapipe/calculators/tflite/ssd_anchors_calculator.cc",
                 "advancify_mediapipe_force_link_ssd_anchors_calculator"),
                ("mediapipe/calculators/tensor/image_to_tensor_calculator.cc",
                 "advancify_mediapipe_force_link_image_to_tensor_calculator"),
                ("mediapipe/calculators/tensor/inference_calculator.cc",
                 "advancify_mediapipe_force_link_inference_calculator"),
                ("mediapipe/calculators/tensor/inference_calculator_cpu.cc",
                 "advancify_mediapipe_force_link_inference_calculator_cpu"),
                ("mediapipe/calculators/tensor/tensors_to_detections_calculator.cc",
                 "advancify_mediapipe_force_link_tensors_to_detections_calculator"),
                ("mediapipe/calculators/util/non_max_suppression_calculator.cc",
                 "advancify_mediapipe_force_link_non_max_suppression_calculator"),
                ("mediapipe/calculators/util/detection_projection_calculator.cc",
                 "advancify_mediapipe_force_link_detection_projection_calculator"),
                ("mediapipe/calculators/util/detections_to_rects_calculator.cc",
                 "advancify_mediapipe_force_link_detections_to_rects_calculator"),
                ("mediapipe/calculators/util/rect_transformation_calculator.cc",
                 "advancify_mediapipe_force_link_rect_transformation_calculator"),
                ("mediapipe/calculators/util/detection_transformation_calculator.cc",
                 "advancify_mediapipe_force_link_detection_transformation_calculator"),
                ("mediapipe/calculators/core/begin_loop_calculator.cc",
                 "advancify_mediapipe_force_link_begin_loop_calculator"),
                ("mediapipe/calculators/core/end_loop_calculator.cc",
                 "advancify_mediapipe_force_link_end_loop_calculator"),
                ("mediapipe/tasks/cc/components/calculators/end_loop_calculator.cc",
                 "advancify_mediapipe_force_link_end_loop_calculator_tasks"),
                ("mediapipe/calculators/core/split_vector_calculator.cc",
                 "advancify_mediapipe_force_link_split_vector_calculator"),
                ("mediapipe/calculators/core/concatenate_vector_calculator.cc",
                 "advancify_mediapipe_force_link_concatenate_vector_calculator"),
                ("mediapipe/calculators/core/get_vector_item_calculator.cc",
                 "advancify_mediapipe_force_link_get_vector_item_calculator"),
                ("mediapipe/calculators/image/image_clone_calculator.cc",
                 "advancify_mediapipe_force_link_image_clone_calculator"),
                ("mediapipe/calculators/tensor/tensors_to_floats_calculator.cc",
                 "advancify_mediapipe_force_link_tensors_to_floats_calculator"),
                ("mediapipe/calculators/util/landmarks_smoothing_calculator.cc",
                 "advancify_mediapipe_force_link_landmarks_smoothing_calculator"),
                ("mediapipe/tasks/cc/vision/face_landmarker/face_blendshapes_graph.cc",
                 "advancify_mediapipe_force_link_face_blendshapes_graph"),
                ("mediapipe/tasks/cc/vision/face_landmarker/tensors_to_face_landmarks_graph.cc",
                 "advancify_mediapipe_force_link_tensors_to_face_landmarks_graph"),
                ("mediapipe/calculators/util/thresholding_calculator.cc",
                 "advancify_mediapipe_force_link_thresholding_calculator"),
                ("mediapipe/calculators/util/landmark_letterbox_removal_calculator.cc",
                 "advancify_mediapipe_force_link_landmark_letterbox_removal_calculator"),
                ("mediapipe/calculators/util/landmark_projection_calculator.cc",
                 "advancify_mediapipe_force_link_landmark_projection_calculator"),
                ("mediapipe/calculators/util/landmarks_to_detection_calculator.cc",
                 "advancify_mediapipe_force_link_landmarks_to_detection_calculator"),
                ("mediapipe/calculators/tensor/landmarks_to_tensor_calculator.cc",
                 "advancify_mediapipe_force_link_landmarks_to_tensor_calculator"),
                ("mediapipe/calculators/tensor/tensors_to_classification_calculator.cc",
                 "advancify_mediapipe_force_link_tensors_to_classification_calculator"),
                ("mediapipe/calculators/tensor/tensors_to_landmarks_calculator.cc",
                 "advancify_mediapipe_force_link_tensors_to_landmarks_calculator"),
                ("mediapipe/calculators/core/split_proto_list_calculator.cc",
                 "advancify_mediapipe_force_link_split_proto_list_calculator"),
                ("mediapipe/framework/stream_handler/in_order_output_stream_handler.cc",
                 "advancify_mediapipe_force_link_in_order_output_stream_handler"),
                ("mediapipe/framework/stream_handler/default_input_stream_handler.cc",
                 "advancify_mediapipe_force_link_default_input_stream_handler"),
                ("mediapipe/framework/stream_handler/immediate_input_stream_handler.cc",
                 "advancify_mediapipe_force_link_immediate_input_stream_handler"),
            ]

            available_symbols = []
            # Добавляем "якорные" символы в соответствующие .cc (один раз).
            for rel_path, sym in force_items:
                abs_path = os.path.join(src_dir, rel_path)
                if not os.path.isfile(abs_path):
                    continue
                with open(abs_path, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
                if ('extern "C" void ' + sym + "()") not in content:
                    content += (
                        "\n\n// " + marker + "\n"
                        'extern "C" void ' + sym + "() {}\n"
                    )
                    save(self, abs_path, content)
                available_symbols.append(sym)

            if os.path.isfile(face_landmarker_cc):
                with open(face_landmarker_cc, "r", encoding="utf-8", errors="ignore") as f:
                    fl_content = f.read()
                # 1) Декларации "якорных" символов (добавляем только отсутствующие).
                if marker not in fl_content:
                    decl_block = "\n// " + marker + "\n"
                    for sym in available_symbols:
                        decl_block += 'extern "C" void ' + sym + "();\n"
                    decl_block += "\n"
                    fl_content = decl_block + fl_content
                else:
                    # marker уже есть — докладываем недостающие декларации.
                    insert_pos = fl_content.find(marker)
                    for sym in available_symbols:
                        decl = 'extern "C" void ' + sym + "();"
                        if decl not in fl_content:
                            # добавим сразу после marker-комментария
                            line_end = fl_content.find("\n", insert_pos)
                            if line_end != -1:
                                fl_content = fl_content[: line_end + 1] + decl + "\n" + fl_content[line_end + 1 :]

                # 2) Вставляем вызовы в FaceLandmarker::Create(), чтобы компоновщик точно
                # подтянул объекты с регистрацией графов/калькуляторов.
                create_pos = fl_content.find("FaceLandmarker::Create")
                if create_pos != -1:
                    brace_pos = fl_content.find("{", create_pos)
                    if brace_pos != -1:
                        missing_calls = []
                        for sym in available_symbols:
                            call_line = sym + "();"
                            if ("\n  " + call_line + "\n") not in fl_content and ("\n" + call_line + "\n") not in fl_content:
                                missing_calls.append(call_line)
                        if missing_calls:
                            call_block = "\n" + "\n".join("  " + c for c in missing_calls) + "\n"
                            fl_content = fl_content[: brace_pos + 1] + call_block + fl_content[brace_pos + 1 :]

                save(self, face_landmarker_cc, fl_content)
        except Exception:
            pass

        # На macOS файл `scheduler_queue.cc` использует `@autoreleasepool`, но компилируется как C++.
        # Меняем условие на `__OBJC__`, чтобы не ломать сборку без Objective-C++.
        scheduler_queue_path = os.path.join(src_dir, "mediapipe", "framework", "scheduler_queue.cc")
        try:
            if os.path.isfile(scheduler_queue_path):
                with open(scheduler_queue_path, "r", encoding="utf-8", errors="ignore") as f:
                    scheduler_queue_content = f.read()
                patched_scheduler_queue_content = scheduler_queue_content.replace(
                    "#ifdef __APPLE__\n#define AUTORELEASEPOOL @autoreleasepool",
                    "#ifdef __OBJC__\n#define AUTORELEASEPOOL @autoreleasepool",
                )
                if patched_scheduler_queue_content != scheduler_queue_content:
                    save(self, scheduler_queue_path, patched_scheduler_queue_content)
        except Exception:
            pass

        # Генерируем `metadata_parser.h` из шаблона (в оригинале делается на стороне Bazel).
        metadata_parser_template_path = os.path.join(
            src_dir, "mediapipe", "tasks", "cc", "metadata", "metadata_parser.h.template"
        )
        metadata_parser_path = os.path.join(
            src_dir, "mediapipe", "tasks", "cc", "metadata", "metadata_parser.h"
        )
        try:
            if os.path.isfile(metadata_parser_template_path) and not os.path.isfile(metadata_parser_path):
                with open(metadata_parser_template_path, "r", encoding="utf-8", errors="ignore") as f:
                    template_content = f.read()
                # Версия нужна как строковая константа и валидируется при парсинге метаданных.
                # Для текущих моделей (.task) минимально требуется 1.0.0.
                save(self, metadata_parser_path, template_content.replace("{LATEST_METADATA_PARSER_VERSION}", "1.0.0"))
        except Exception:
            pass

        # Генерация protobuf (*.pb.h/*.pb.cc) для протоколов, используемых Tasks API.
        protobuf_dependency = self.dependencies.build["protobuf"]
        protoc_executable = os.path.join(
            protobuf_dependency.package_folder,
            "bin",
            "protoc.exe" if str(self.settings.os) == "Windows" else "protoc",
        )
        proto_output_dir = os.path.join(self.build_folder, "proto-gen")
        if os.path.isdir(proto_output_dir):
            shutil.rmtree(proto_output_dir)
        os.makedirs(proto_output_dir, exist_ok=True)

        # В mediapipe есть proto, которые зависят от внешних деревьев (например tensorflow/...),
        # поэтому не генерируем такие зависимости.
        proto_roots = (
            os.path.join(src_dir, "mediapipe", "framework"),
            os.path.join(src_dir, "mediapipe", "tasks"),
            os.path.join(src_dir, "mediapipe", "calculators", "core"),
            os.path.join(src_dir, "mediapipe", "calculators", "internal"),
            os.path.join(src_dir, "mediapipe", "calculators", "image"),
            os.path.join(src_dir, "mediapipe", "calculators", "tensor"),
            os.path.join(src_dir, "mediapipe", "calculators", "util"),
            os.path.join(src_dir, "mediapipe", "util"),
            os.path.join(src_dir, "mediapipe", "gpu"),
        )
        required_proto_relpaths = set()


        for proto_root in proto_roots:
            if not os.path.isdir(proto_root):
                continue
            for root, dirs, files in os.walk(proto_root):
                dirs[:] = [d for d in dirs if d not in (".git", ".github")]
                for filename in files:
                    if not filename.endswith(".proto"):
                        continue
                    absolute_path = os.path.join(root, filename)
                    relpath = os.path.relpath(absolute_path, src_dir)
                    if "/tensorflow/" in relpath or relpath.startswith("tensorflow/"):
                        continue
                    required_proto_relpaths.add(relpath)

        # Добираем транзитивные зависимости через `import` в proto, иначе можно получить
        # ситуацию, когда сгенерированный *.pb.h включает другой *.pb.h, которого нет в пакете.
        # Пример: calculator.pb.h -> packet_factory.pb.h.
        import_regex = re.compile(r'^\s*import\s+(?:public\s+)?\"([^\"]+)\"', re.MULTILINE)
        proto_queue = list(required_proto_relpaths)
        while proto_queue:
            proto_relpath = proto_queue.pop()
            proto_abs = os.path.join(src_dir, proto_relpath)
            try:
                with open(proto_abs, "r", encoding="utf-8", errors="ignore") as f:
                    proto_content = f.read()
            except OSError:
                continue

            for imported in import_regex.findall(proto_content):
                # Генерируем только proto из дерева исходников mediapipe.
                # tensorflow/* (как в calculators/tensorflow) намеренно не поддерживаем.
                if not imported.endswith(".proto"):
                    continue
                if "/tensorflow/" in imported or imported.startswith("tensorflow/"):
                    continue
                if not imported.startswith("mediapipe/"):
                    continue
                imported_abs = os.path.join(src_dir, imported)
                if not os.path.isfile(imported_abs):
                    continue
                if imported not in required_proto_relpaths:
                    required_proto_relpaths.add(imported)
                    proto_queue.append(imported)

        proto_files = sorted(required_proto_relpaths)

        proto_include_dirs = [src_dir]
        protobuf_res_dir = os.path.join(protobuf_dependency.package_folder, "res")
        if os.path.isdir(protobuf_res_dir):
            proto_include_dirs.append(protobuf_res_dir)
        protobuf_include_dir = os.path.join(protobuf_dependency.package_folder, "include")
        if os.path.isdir(protobuf_include_dir):
            proto_include_dirs.append(protobuf_include_dir)


        with chdir(self, src_dir):
            proto_path_args = " ".join([f'-I"{p}"' for p in proto_include_dirs])
            for proto_relpath in proto_files:
                self.run(
                    f'"{protoc_executable}" {proto_path_args} --cpp_out="{proto_output_dir}" "{proto_relpath}"',
                    env="conanbuild",
                )

        # Генерация flatbuffers-хедера, который ожидается Tasks metadata (metadata_schema_generated.h).
        flatbuffers_dependency = self.dependencies.build["flatbuffers"]
        flatc_executable = os.path.join(
            flatbuffers_dependency.package_folder,
            "bin",
            "flatc.exe" if str(self.settings.os) == "Windows" else "flatc",
        )
        metadata_schema_relpaths = [
            os.path.join("mediapipe", "tasks", "metadata", "metadata_schema.fbs"),
            os.path.join("mediapipe", "tasks", "metadata", "object_detector_metadata_schema.fbs"),
        ]
        flatbuffers_gen_root = os.path.join(self.build_folder, "flatbuffers-gen")
        flatbuffers_output_dir = os.path.join(flatbuffers_gen_root, "mediapipe", "tasks", "metadata")
        if os.path.isdir(flatbuffers_gen_root):
            shutil.rmtree(flatbuffers_gen_root)
        os.makedirs(flatbuffers_output_dir, exist_ok=True)

        if os.path.isfile(flatc_executable):
            with chdir(self, src_dir):
                for schema_relpath in metadata_schema_relpaths:
                    schema_abs = os.path.join(src_dir, schema_relpath)
                    if not os.path.isfile(schema_abs):
                        continue
                    self.run(
                        f'"{flatc_executable}" --cpp -o "{flatbuffers_output_dir}" "{schema_relpath}"',
                        env="conanbuild",
                    )

        cmakelists = f"""
cmake_minimum_required(VERSION 3.22)
project(mediapipe LANGUAGES CXX)

include(GNUInstallDirs)

find_package(protobuf REQUIRED)
find_package(glog REQUIRED)
find_package(tensorflowlite REQUIRED)
find_package(absl REQUIRED)
find_package(flatbuffers REQUIRED)
find_package(Eigen3 REQUIRED)

set(MEDIAPIPE_SRC "${{MEDIAPIPE_SRC}}")
set(PROTO_GEN "${{PROTO_GEN}}")
set(FLATBUFFERS_GEN "${{FLATBUFFERS_GEN}}")

file(GLOB_RECURSE MP_SOURCES
  "${{MEDIAPIPE_SRC}}/mediapipe/framework/*.cc"
  "${{MEDIAPIPE_SRC}}/mediapipe/framework/tool/*.cc"
  "${{MEDIAPIPE_SRC}}/mediapipe/tasks/cc/common.cc"
  "${{MEDIAPIPE_SRC}}/mediapipe/tasks/cc/core/*.cc"
  "${{MEDIAPIPE_SRC}}/mediapipe/tasks/cc/components/containers/*.cc"
  "${{MEDIAPIPE_SRC}}/mediapipe/tasks/cc/components/containers/*/*.cc"
  "${{MEDIAPIPE_SRC}}/mediapipe/tasks/cc/components/utils/*.cc"
  "${{MEDIAPIPE_SRC}}/mediapipe/tasks/cc/components/utils/*/*.cc"
  "${{MEDIAPIPE_SRC}}/mediapipe/tasks/cc/components/processors/image_preprocessing_graph.cc"
  "${{MEDIAPIPE_SRC}}/mediapipe/tasks/cc/components/processors/detection_postprocessing_graph.cc"
  "${{MEDIAPIPE_SRC}}/mediapipe/tasks/cc/components/calculators/*.cc"
  "${{MEDIAPIPE_SRC}}/mediapipe/tasks/cc/metadata/metadata_extractor.cc"
  "${{MEDIAPIPE_SRC}}/mediapipe/tasks/cc/metadata/metadata_version.cc"
  "${{MEDIAPIPE_SRC}}/mediapipe/tasks/cc/metadata/metadata_version_utils.cc"
  "${{MEDIAPIPE_SRC}}/mediapipe/tasks/cc/vision/core/*.cc"
  "${{MEDIAPIPE_SRC}}/mediapipe/tasks/cc/vision/utils/image_tensor_specs.cc"
  "${{MEDIAPIPE_SRC}}/mediapipe/tasks/cc/vision/custom_ops/*.cc"
  "${{MEDIAPIPE_SRC}}/mediapipe/tasks/cc/vision/face_detector/*.cc"
  "${{MEDIAPIPE_SRC}}/mediapipe/tasks/cc/vision/face_geometry/*.cc"
  "${{MEDIAPIPE_SRC}}/mediapipe/tasks/cc/vision/face_geometry/*/*.cc"
  "${{MEDIAPIPE_SRC}}/mediapipe/tasks/cc/vision/face_geometry/*/*/*.cc"
  "${{MEDIAPIPE_SRC}}/mediapipe/tasks/cc/vision/face_landmarker/*.cc"
  "${{MEDIAPIPE_SRC}}/mediapipe/calculators/core/*.cc"
  "${{MEDIAPIPE_SRC}}/mediapipe/calculators/image/image_clone_calculator.cc"
  "${{MEDIAPIPE_SRC}}/mediapipe/calculators/image/image_properties_calculator.cc"
  "${{MEDIAPIPE_SRC}}/mediapipe/calculators/tensor/image_to_tensor_calculator.cc"
  "${{MEDIAPIPE_SRC}}/mediapipe/calculators/tensor/image_to_tensor_utils.cc"
  "${{MEDIAPIPE_SRC}}/mediapipe/calculators/tensor/*.cc"
  "${{MEDIAPIPE_SRC}}/mediapipe/calculators/tensor/landmarks_to_tensor_calculator.cc"
  "${{MEDIAPIPE_SRC}}/mediapipe/calculators/tensor/tensors_dequantization_calculator.cc"
  "${{MEDIAPIPE_SRC}}/mediapipe/calculators/tensor/tensors_to_classification_calculator.cc"
  "${{MEDIAPIPE_SRC}}/mediapipe/calculators/tensor/tensors_to_detections_calculator.cc"
  "${{MEDIAPIPE_SRC}}/mediapipe/calculators/tensor/tensors_to_floats_calculator.cc"
  "${{MEDIAPIPE_SRC}}/mediapipe/calculators/tensor/tensors_to_landmarks_calculator.cc"
  "${{MEDIAPIPE_SRC}}/mediapipe/calculators/tflite/ssd_anchors_calculator.cc"
  "${{MEDIAPIPE_SRC}}/mediapipe/calculators/util/association_norm_rect_calculator.cc"
  "${{MEDIAPIPE_SRC}}/mediapipe/calculators/util/collection_has_min_size_calculator.cc"
  "${{MEDIAPIPE_SRC}}/mediapipe/calculators/util/detection_projection_calculator.cc"
  "${{MEDIAPIPE_SRC}}/mediapipe/calculators/util/detection_transformation_calculator.cc"
  "${{MEDIAPIPE_SRC}}/mediapipe/calculators/util/detections_to_rects_calculator.cc"
  "${{MEDIAPIPE_SRC}}/mediapipe/calculators/util/landmark_letterbox_removal_calculator.cc"
  "${{MEDIAPIPE_SRC}}/mediapipe/calculators/util/landmark_projection_calculator.cc"
  "${{MEDIAPIPE_SRC}}/mediapipe/calculators/util/landmarks_refinement_calculator.cc"
  "${{MEDIAPIPE_SRC}}/mediapipe/calculators/util/landmarks_smoothing_calculator.cc"
  "${{MEDIAPIPE_SRC}}/mediapipe/calculators/util/landmarks_smoothing_calculator_utils.cc"
  "${{MEDIAPIPE_SRC}}/mediapipe/calculators/util/landmarks_to_detection_calculator.cc"
  "${{MEDIAPIPE_SRC}}/mediapipe/calculators/util/non_max_suppression_calculator.cc"
  "${{MEDIAPIPE_SRC}}/mediapipe/calculators/util/rect_transformation_calculator.cc"
  "${{MEDIAPIPE_SRC}}/mediapipe/calculators/util/thresholding_calculator.cc"
  "${{MEDIAPIPE_SRC}}/mediapipe/framework/formats/*.cc"
  "${{MEDIAPIPE_SRC}}/mediapipe/framework/stream_handler/*.cc"
  "${{MEDIAPIPE_SRC}}/mediapipe/util/graph_builder_utils.cc"
  "${{MEDIAPIPE_SRC}}/mediapipe/util/filtering/low_pass_filter.cc"
  "${{MEDIAPIPE_SRC}}/mediapipe/util/filtering/one_euro_filter.cc"
  "${{MEDIAPIPE_SRC}}/mediapipe/util/filtering/relative_velocity_filter.cc"
  "${{MEDIAPIPE_SRC}}/mediapipe/util/header_util.cc"
  "${{MEDIAPIPE_SRC}}/mediapipe/util/label_map_util.cc"
  "${{MEDIAPIPE_SRC}}/mediapipe/util/rectangle_util.cc"
  "${{MEDIAPIPE_SRC}}/mediapipe/util/str_util.cc"
  "${{MEDIAPIPE_SRC}}/mediapipe/util/cpu_util.cc"
  "${{MEDIAPIPE_SRC}}/mediapipe/util/resource_util.cc"
  "${{MEDIAPIPE_SRC}}/mediapipe/util/resource_util_default.cc"
  "${{MEDIAPIPE_SRC}}/mediapipe/util/resource_util_custom.cc"
  "${{MEDIAPIPE_SRC}}/mediapipe/util/tflite/*.cc"
  "${{MEDIAPIPE_SRC}}/mediapipe/gpu/gpu_buffer.cc"
  "${{MEDIAPIPE_SRC}}/mediapipe/gpu/gpu_buffer_storage.cc"
  "${{MEDIAPIPE_SRC}}/mediapipe/gpu/gpu_buffer_storage_image_frame.cc"
  "${{MEDIAPIPE_SRC}}/mediapipe/gpu/gpu_buffer_format.cc"
)

file(GLOB_RECURSE MP_PROTO_SOURCES "${{PROTO_GEN}}/*.pb.cc")
list(APPEND MP_SOURCES ${{MP_PROTO_SOURCES}})

list(FILTER MP_SOURCES EXCLUDE REGEX ".*/test/.*")
list(FILTER MP_SOURCES EXCLUDE REGEX ".*/tests/.*")
list(FILTER MP_SOURCES EXCLUDE REGEX ".*_test\\\\.cc$")
list(FILTER MP_SOURCES EXCLUDE REGEX ".*_tests\\\\.cc$")
list(FILTER MP_SOURCES EXCLUDE REGEX ".*_unittest\\\\.cc$")
list(FILTER MP_SOURCES EXCLUDE REGEX ".*_benchmark\\\\.cc$")
list(FILTER MP_SOURCES EXCLUDE REGEX ".*_gl\\\\.cc$")
list(FILTER MP_SOURCES EXCLUDE REGEX ".*/gpu/.*gl_.*\\\\.cc$")
list(FILTER MP_SOURCES EXCLUDE REGEX ".*/framework/formats/motion/.*\\\\.cc$")
list(FILTER MP_SOURCES EXCLUDE REGEX ".*/framework/debug/.*\\\\.cc$")
list(FILTER MP_SOURCES EXCLUDE REGEX ".*/framework/profiler/.*\\\\.cc$")
list(FILTER MP_SOURCES EXCLUDE REGEX ".*/framework/tool/.*_template\\\\.cc$")
list(FILTER MP_SOURCES EXCLUDE REGEX ".*/framework/tool/.*test.*\\\\.cc$")
list(FILTER MP_SOURCES EXCLUDE REGEX ".*/tasks/cc/metadata/python/.*\\\\.cc$")
list(FILTER MP_SOURCES EXCLUDE REGEX ".*/mediapipe_builtin_op_resolver\\\\.cc$")
list(FILTER MP_SOURCES EXCLUDE REGEX ".*/mediapipe/util/tflite/op_resolver\\\\.cc$")
list(FILTER MP_SOURCES EXCLUDE REGEX ".*/mediapipe/util/tflite/operations/.*\\\\.cc$")
list(FILTER MP_SOURCES EXCLUDE REGEX ".*/mediapipe/util/tflite/.*gpu.*\\\\.cc$")
list(FILTER MP_SOURCES EXCLUDE REGEX ".*bert.*\\\\.cc$")
list(FILTER MP_SOURCES EXCLUDE REGEX ".*tokenizer.*\\\\.cc$")
list(FILTER MP_SOURCES EXCLUDE REGEX ".*audio.*\\\\.cc$")
list(FILTER MP_SOURCES EXCLUDE REGEX ".*metal.*\\\\.cc$")
list(FILTER MP_SOURCES EXCLUDE REGEX ".*/tasks/cc/text/.*\\\\.cc$")
list(FILTER MP_SOURCES EXCLUDE REGEX ".*gpu.*delegate.*\\\\.cc$")
list(FILTER MP_SOURCES EXCLUDE REGEX ".*tflite_gpu_runner.*\\\\.cc$")
list(FILTER MP_SOURCES EXCLUDE REGEX ".*inference_on_disk_cache.*\\\\.cc$")
list(FILTER MP_SOURCES EXCLUDE REGEX ".*_gl.*\\\\.cc$")
list(FILTER MP_SOURCES EXCLUDE REGEX ".*regex_preprocessor.*\\\\.cc$")
list(FILTER MP_SOURCES EXCLUDE REGEX ".*_test_utils\\\\.cc$")

list(APPEND MP_SOURCES "${{CMAKE_CURRENT_LIST_DIR}}/mediapipe_builtin_op_resolver.cc")
list(APPEND MP_SOURCES "${{CMAKE_CURRENT_LIST_DIR}}/zip_utils.cc")

add_library(mediapipe STATIC ${{MP_SOURCES}})
if (NOT WIN32)
  set_property(TARGET mediapipe PROPERTY POSITION_INDEPENDENT_CODE ON)
endif()
target_compile_features(mediapipe PUBLIC cxx_std_20)
target_compile_definitions(mediapipe PUBLIC MEDIAPIPE_DISABLE_GPU=1)
target_include_directories(mediapipe
  PUBLIC
    $<BUILD_INTERFACE:${{MEDIAPIPE_SRC}}>
    $<BUILD_INTERFACE:${{PROTO_GEN}}>
    $<BUILD_INTERFACE:${{FLATBUFFERS_GEN}}>
    $<INSTALL_INTERFACE:include>
)
find_package(OpenCV REQUIRED)
target_link_libraries(mediapipe PUBLIC opencv::opencv)
target_link_libraries(mediapipe
  PUBLIC
    protobuf::protobuf
    glog::glog
    tensorflow::tensorflowlite
    abseil::abseil
    flatbuffers::flatbuffers
    Eigen3::Eigen
)

install(DIRECTORY "${{MEDIAPIPE_SRC}}/mediapipe" DESTINATION ${{CMAKE_INSTALL_INCLUDEDIR}} PATTERN "*.bazel" EXCLUDE PATTERN "*.BUILD" EXCLUDE)
install(DIRECTORY "${{MEDIAPIPE_SRC}}/third_party" DESTINATION ${{CMAKE_INSTALL_INCLUDEDIR}} PATTERN "*.bazel" EXCLUDE PATTERN "*.BUILD" EXCLUDE)
install(DIRECTORY "${{PROTO_GEN}}/mediapipe" DESTINATION ${{CMAKE_INSTALL_INCLUDEDIR}})
install(FILES "${{FLATBUFFERS_GEN}}/mediapipe/tasks/metadata/metadata_schema_generated.h" DESTINATION ${{CMAKE_INSTALL_INCLUDEDIR}}/mediapipe/tasks/metadata OPTIONAL)

install(TARGETS mediapipe EXPORT mediapipeTargets
  ARCHIVE DESTINATION ${{CMAKE_INSTALL_LIBDIR}}
  LIBRARY DESTINATION ${{CMAKE_INSTALL_LIBDIR}}
  RUNTIME DESTINATION ${{CMAKE_INSTALL_BINDIR}}
)
export(EXPORT mediapipeTargets)
"""
        # Минимальная реализация OpResolver: только то, что нужно для face_landmarker.
        # Это позволяет не тащить текстовые кастом-опы (re2/sentencepiece и т.п.).
        mediapipe_builtin_op_resolver_cc = r"""
/* Автогенерация рецептом Conan: минимальный набор кастомных TFLite-операторов. */

#include "mediapipe/tasks/cc/core/mediapipe_builtin_op_resolver.h"

#include "mediapipe/tasks/cc/vision/custom_ops/fused_batch_norm.h"

namespace mediapipe {
namespace tasks {
namespace core {

MediaPipeBuiltinOpResolver::MediaPipeBuiltinOpResolver() {
  AddCustom("FusedBatchNormV3",
            mediapipe::tflite_operations::Register_FusedBatchNorm());
}

}  // namespace core
}  // namespace tasks
}  // namespace mediapipe
"""
        zip_utils_cc = r"""
/* Автогенерация рецептом Conan: zip-утилиты для .task (ZIP).
 *
 * MediaPipe Tasks использует .task как zip-контейнер, где ассеты лежат как STORED
 * (без сжатия). Оригинальный mediapipe использует minizip; здесь делаем небольшой
 * ZIP-парсер под STORED записи, чтобы не тянуть внешнюю зависимость.
 */

#include "mediapipe/tasks/cc/metadata/utils/zip_utils.h"

#include <cstdint>
#include <cstring>
#include <string>

#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "mediapipe/tasks/cc/common.h"

#include <chrono>
#include <cstdio>
#include <fstream>
#include <sstream>

namespace mediapipe {
namespace tasks {
namespace metadata {

namespace {

constexpr uint32_t kZipLocalFileHeaderSig = 0x04034b50u;   // PK\003\004
constexpr uint32_t kZipCentralDirHeaderSig = 0x02014b50u;  // PK\001\002
constexpr uint32_t kZipEndOfCentralDirSig = 0x06054b50u;   // PK\005\006

inline uint16_t ReadU16LE(const unsigned char* p) {
  return static_cast<uint16_t>(p[0]) | (static_cast<uint16_t>(p[1]) << 8);
}
inline uint32_t ReadU32LE(const unsigned char* p) {
  return static_cast<uint32_t>(p[0]) | (static_cast<uint32_t>(p[1]) << 8) |
         (static_cast<uint32_t>(p[2]) << 16) | (static_cast<uint32_t>(p[3]) << 24);
}

absl::Status ZipError(absl::string_view message) {
  return CreateStatusWithPayload(absl::StatusCode::kUnknown, std::string(message),
                                 MediaPipeTasksStatus::kFileZipError);
}

}  // namespace

absl::Status ExtractFilesfromZipFile(
    const char* buffer_data, const size_t buffer_size,
    absl::flat_hash_map<std::string, absl::string_view>* files) {
  if (files == nullptr) {
    return ZipError("Output map is null.");
  }
  files->clear();
  if (buffer_data == nullptr || buffer_size < 4) {
    return absl::OkStatus();
  }

  const unsigned char* udata = reinterpret_cast<const unsigned char*>(buffer_data);

  // .task иногда приходит с небольшим префиксом (например, 2 байта 0x00),
  // поэтому ищем начало ZIP по сигнатуре Local File Header (PK\003\004)
  // в начале буфера. Если сигнатуры нет — считаем это обычным .tflite.
  size_t zip_base = static_cast<size_t>(-1);
  const size_t scan_limit = (buffer_size > 4096) ? 4096 : buffer_size;
  for (size_t i = 0; i + 4 <= scan_limit; ++i) {
    if (ReadU32LE(udata + i) == kZipLocalFileHeaderSig) {
      zip_base = i;
      break;
    }
  }
  if (zip_base == static_cast<size_t>(-1)) {
    return absl::OkStatus();
  }

  // Ищем End Of Central Directory (EOCD) в последних 64KB.
  const size_t kMaxComment = 0xFFFF;
  const size_t search_window = (buffer_size > (kMaxComment + 22)) ? (kMaxComment + 22) : buffer_size;
  size_t eocd_pos = static_cast<size_t>(-1);
  for (size_t i = buffer_size - search_window; i + 4 <= buffer_size; ++i) {
    if (ReadU32LE(udata + i) == kZipEndOfCentralDirSig) {
      eocd_pos = i;
      break;  // берём первое найденное (обычно оно одно)
    }
  }
  if (eocd_pos == static_cast<size_t>(-1)) {
    return ZipError("Unable to find End of Central Directory record.");
  }
  if (eocd_pos + 22 > buffer_size) {
    return ZipError("Corrupted EOCD record.");
  }

  const uint16_t total_entries = ReadU16LE(udata + eocd_pos + 10);
  const uint32_t central_dir_size = ReadU32LE(udata + eocd_pos + 12);
  const uint32_t central_dir_offset = ReadU32LE(udata + eocd_pos + 16);


  // Встречаются два варианта: оффсеты в EOCD могут быть как абсолютными от начала файла,
  // так и относительными к "началу zip" внутри буфера. Пробуем оба.
  uint64_t central_dir_begin = static_cast<uint64_t>(central_dir_offset);
  bool central_dir_ok = (central_dir_begin + static_cast<uint64_t>(central_dir_size) <= buffer_size) &&
                        (ReadU32LE(udata + central_dir_begin) == kZipCentralDirHeaderSig);
  if (!central_dir_ok) {
    const uint64_t alt_begin = static_cast<uint64_t>(zip_base) + static_cast<uint64_t>(central_dir_offset);
    const bool alt_ok = (alt_begin + static_cast<uint64_t>(central_dir_size) <= buffer_size) &&
                        (ReadU32LE(udata + alt_begin) == kZipCentralDirHeaderSig);
    if (!alt_ok) {
      return ZipError("Invalid central directory header signature.");
    }
    central_dir_begin = alt_begin;
  }

  size_t cd = static_cast<size_t>(central_dir_begin);
  for (uint16_t idx = 0; idx < total_entries; ++idx) {
    if (cd + 46 > buffer_size) {
      return ZipError("Central directory entry is out of bounds.");
    }
    if (ReadU32LE(udata + cd) != kZipCentralDirHeaderSig) {
      return ZipError("Invalid central directory header signature.");
    }

    const uint16_t compression_method = ReadU16LE(udata + cd + 10);
    const uint32_t uncompressed_size = ReadU32LE(udata + cd + 24);
    const uint16_t file_name_len = ReadU16LE(udata + cd + 28);
    const uint16_t extra_len = ReadU16LE(udata + cd + 30);
    const uint16_t comment_len = ReadU16LE(udata + cd + 32);
    const uint32_t local_header_offset_rel = ReadU32LE(udata + cd + 42);

    const size_t name_pos = cd + 46;
    const size_t next_cd = name_pos + static_cast<size_t>(file_name_len) + extra_len + comment_len;
    if (name_pos + file_name_len > buffer_size || next_cd > buffer_size) {
      return ZipError("Central directory name/extra/comment is out of bounds.");
    }

    if (compression_method != 0) {
      return ZipError("Expected uncompressed (STORED) zip archive.");
    }

    const std::string file_name(reinterpret_cast<const char*>(udata + name_pos),
                                reinterpret_cast<const char*>(udata + name_pos + file_name_len));
    // пропускаем директории
    if (!file_name.empty() && file_name.back() != '/') {
      // Local file header
      uint64_t local_header_offset = static_cast<uint64_t>(local_header_offset_rel);
      bool local_ok = (local_header_offset + 30 <= buffer_size) &&
                      (ReadU32LE(udata + local_header_offset) == kZipLocalFileHeaderSig);
      if (!local_ok) {
        const uint64_t alt_local = static_cast<uint64_t>(zip_base) + static_cast<uint64_t>(local_header_offset_rel);
        const bool alt_ok = (alt_local + 30 <= buffer_size) &&
                            (ReadU32LE(udata + alt_local) == kZipLocalFileHeaderSig);
        if (!alt_ok) {
          return ZipError("Invalid local file header signature.");
        }
        local_header_offset = alt_local;
      }
      const uint16_t lfh_name_len = ReadU16LE(udata + local_header_offset + 26);
      const uint16_t lfh_extra_len = ReadU16LE(udata + local_header_offset + 28);
      const size_t data_offset = static_cast<size_t>(local_header_offset) + 30 +
                                 static_cast<size_t>(lfh_name_len) + static_cast<size_t>(lfh_extra_len);
      if (static_cast<uint64_t>(data_offset) + static_cast<uint64_t>(uncompressed_size) > buffer_size) {
        return ZipError("Local file data is out of bounds.");
      }

      (*files)[file_name] = absl::string_view(reinterpret_cast<const char*>(udata + data_offset),
                                              static_cast<size_t>(uncompressed_size));
    }

    cd = next_cd;
  }

  return absl::OkStatus();
}

void SetExternalFile(const absl::string_view& file_content,
                     core::proto::ExternalFile* model_file,
                     bool is_copy) {
  if (model_file == nullptr) return;
  if (is_copy) {
    model_file->set_file_content(std::string(file_content));
    return;
  }
  const auto pointer = reinterpret_cast<uint64_t>(file_content.data());
  model_file->mutable_file_pointer_meta()->set_pointer(pointer);
  model_file->mutable_file_pointer_meta()->set_length(file_content.length());
}

}  // namespace metadata
}  // namespace tasks
}  // namespace mediapipe
"""
        save(self, os.path.join(wrap_dir, "mediapipe_builtin_op_resolver.cc"), mediapipe_builtin_op_resolver_cc)
        save(self, os.path.join(wrap_dir, "zip_utils.cc"), zip_utils_cc)
        save(self, os.path.join(wrap_dir, "CMakeLists.txt"), cmakelists)

        cmake = CMake(self)
        cmake.configure(build_script_folder=wrap_dir, variables={
            "MEDIAPIPE_SRC": src_dir,
            "PROTO_GEN": proto_output_dir,
            "FLATBUFFERS_GEN": flatbuffers_gen_root,
        })
        cmake.build()


    def package(self):
        src_dir = os.path.join(self.source_folder, "mediapipe")
        # Устанавливаем через cmake.install (библиотека + include)
        cmake = CMake(self)
        cmake.install()

        # Чтобы потребитель мог подключать mediapipe-хедеры без прямого подключения Abseil,
        # докладываем публичные хедеры Abseil в include-дерево пакета (absl/...).
        # Это устраняет ошибки вида: `fatal error: 'absl/status/statusor.h' file not found`.
        try:
            abseil_dependency = self.dependencies.get("abseil")
            if abseil_dependency is not None and abseil_dependency.package_folder:
                abseil_include_dir = os.path.join(abseil_dependency.package_folder, "include")
                if os.path.isdir(abseil_include_dir):
                    copy(
                        self,
                        pattern=os.path.join("absl", "**"),
                        src=abseil_include_dir,
                        dst=os.path.join(self.package_folder, "include"),
                        keep_path=True,
                    )
        except Exception:
            pass

        # Аналогично докладываем protobuf-хедеры (google/protobuf/...), т.к. сгенерированные *.pb.h
        # инклюдят их напрямую через <google/protobuf/...>.
        try:
            protobuf_dependency = self.dependencies.get("protobuf")
            if protobuf_dependency is not None and protobuf_dependency.package_folder:
                protobuf_include_dir = os.path.join(protobuf_dependency.package_folder, "include")
                if os.path.isdir(protobuf_include_dir):
                    copy(
                        self,
                        pattern=os.path.join("google", "protobuf", "**"),
                        src=protobuf_include_dir,
                        dst=os.path.join(self.package_folder, "include"),
                        keep_path=True,
                    )
        except Exception:
            pass

        # MediaPipe использует glog (mediapipe/framework/port/logging.h -> glog/logging.h).
        try:
            glog_dependency = self.dependencies.get("glog")
            if glog_dependency is not None and glog_dependency.package_folder:
                glog_include_dir = os.path.join(glog_dependency.package_folder, "include")
                if os.path.isdir(glog_include_dir):
                    copy(
                        self,
                        pattern=os.path.join("glog", "**"),
                        src=glog_include_dir,
                        dst=os.path.join(self.package_folder, "include"),
                        keep_path=True,
                    )
        except Exception:
            pass

        # glog может зависеть от gflags на уровне хедеров.
        try:
            gflags_dependency = self.dependencies.get("gflags")
            if gflags_dependency is not None and gflags_dependency.package_folder:
                gflags_include_dir = os.path.join(gflags_dependency.package_folder, "include")
                if os.path.isdir(gflags_include_dir):
                    copy(
                        self,
                        pattern=os.path.join("gflags", "**"),
                        src=gflags_include_dir,
                        dst=os.path.join(self.package_folder, "include"),
                        keep_path=True,
                    )
        except Exception:
            pass

        # Хедеры flatbuffers нужны для tasks metadata.
        try:
            flatbuffers_dependency = self.dependencies.get("flatbuffers")
            if flatbuffers_dependency is not None and flatbuffers_dependency.package_folder:
                flatbuffers_include_dir = os.path.join(flatbuffers_dependency.package_folder, "include")
                if os.path.isdir(flatbuffers_include_dir):
                    copy(
                        self,
                        pattern=os.path.join("flatbuffers", "**"),
                        src=flatbuffers_include_dir,
                        dst=os.path.join(self.package_folder, "include"),
                        keep_path=True,
                    )
        except Exception:
            pass

        # Хедеры TensorFlow Lite нужны для tasks/core и custom_ops.
        try:
            tflite_dependency = self.dependencies.get("tensorflow-lite")
            if tflite_dependency is not None and tflite_dependency.package_folder:
                tflite_include_dir = os.path.join(tflite_dependency.package_folder, "include")
                if os.path.isdir(tflite_include_dir):
                    copy(
                        self,
                        pattern=os.path.join("tensorflow", "lite", "**"),
                        src=tflite_include_dir,
                        dst=os.path.join(self.package_folder, "include"),
                        keep_path=True,
                    )
        except Exception:
            pass


        # Докладываем сгенерированные *.pb.h в include-дерево пакета.
        proto_output_dir = os.path.join(self.build_folder, "proto-gen")
        if os.path.isdir(proto_output_dir):
            copy(
                self,
                pattern="*.pb.h",
                src=proto_output_dir,
                dst=os.path.join(self.package_folder, "include"),
                keep_path=True,
            )

        # Докладываем сгенерированный flatbuffers header для metadata schema.
        flatbuffers_output_dir = os.path.join(self.build_folder, "flatbuffers-gen", "mediapipe", "tasks", "metadata")
        metadata_generated_header = os.path.join(flatbuffers_output_dir, "metadata_schema_generated.h")
        if os.path.isfile(metadata_generated_header):
            copy(
                self,
                pattern="metadata_schema_generated.h",
                src=flatbuffers_output_dir,
                dst=os.path.join(self.package_folder, "include", "mediapipe", "tasks", "metadata"),
                keep_path=False,
            )

        # Лицензия
        copy(self, "LICENSE",
             src=src_dir,
             dst=os.path.join(self.package_folder, "licenses"),
             keep_path=False)

    def package_info(self):
        # Статическая библиотека + include.
        self.cpp_info.set_property("cmake_file_name", "mediapipe")
        self.cpp_info.set_property("cmake_target_name", "mediapipe::mediapipe")
        self.cpp_info.includedirs = ["include"]
        self.cpp_info.defines = ["MEDIAPIPE_DISABLE_GPU=1"]
        self.cpp_info.libs = ["mediapipe"]
        self.cpp_info.requires = [
            "opencv::opencv",
            # Нужен как минимум StatusOr для Tasks API (и он протаскивает include/ Abseil).
            "abseil::absl_statusor",
            "eigen::eigen",
            "glog::glog",
            "protobuf::protobuf",
            "flatbuffers::flatbuffers",
            "tensorflow-lite::tensorflow-lite",
        ]

        # В проекте медиапайп может использоваться через include-path без линковки CMake-таргета,
        # поэтому докладываем include Abseil напрямую, чтобы сборка хедеров была стабильной.
        # (Иначе `#include "absl/..."` не находится при компиляции потребителя.)
        try:
            abseil_dependency = self.dependencies.get("abseil")
            if abseil_dependency is not None and abseil_dependency.package_folder:
                abseil_include_dir = os.path.join(abseil_dependency.package_folder, "include")
                if os.path.isdir(abseil_include_dir):
                    # Важно: часть потребителей берет include dirs из переменных CMakeDeps,
                    # не используя таргеты и транзитивные зависимости. Поэтому добавляем
                    # include-dir напрямую в includedirs (и продублируем как system include).
                    self.cpp_info.includedirs.append(abseil_include_dir)
                    self.cpp_info.system_includedirs.append(abseil_include_dir)
        except Exception:
            pass

