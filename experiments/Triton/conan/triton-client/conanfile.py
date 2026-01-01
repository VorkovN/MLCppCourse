from conan import ConanFile
from conan.tools.cmake import CMake, cmake_layout
from conan.tools.files import get
import os

class TritonClientConan(ConanFile):
    name = "triton-client"
    version = "2.57.1"
    package_type = "library"
    description = "Triton Inference Server C++ Client Libraries"
    homepage = "https://github.com/triton-inference-server/server"
    settings = "os", "arch", "compiler", "build_type"

    options = {
        "shared": [True, False],
    }
    default_options = {
        "shared": False,
    }
    generators = "CMakeToolchain", "CMakeDeps"

    def layout(self):
        cmake_layout(self)

    def requirements(self):
        self.requires("rapidjson/cci.20230929")

    def source(self):
        get(
            self,
            url="https://github.com/triton-inference-server/client/archive/refs/heads/r24.10.tar.gz",
            strip_root=True,
        )

    def build(self):
        rapidjson_dep = self.dependencies["rapidjson"]
        rapidjson_root = rapidjson_dep.package_folder
        rapidjson_cmake_dir = os.path.join(rapidjson_root, "lib", "cmake", "RapidJSON")

        # macOS: version-script не поддерживается, выключаем для httpclient
        lib_cmake = os.path.join(self.source_folder, "src", "c++", "library", "CMakeLists.txt")
        if os.path.exists(lib_cmake):
            with open(lib_cmake, "r", encoding="utf-8") as f:
                content = f.read()
            content = content.replace("if (NOT WIN32)", "if (NOT WIN32 AND NOT APPLE)")
            with open(lib_cmake, "w", encoding="utf-8") as f:
                f.write(content)

        cmake = CMake(self)
        cmake.configure(
            build_script_folder=self.source_folder,
            variables={
                "CMAKE_PREFIX_PATH": self.generators_folder,
                "RapidJSON_DIR": rapidjson_cmake_dir,
                "RapidJSON_ROOT": rapidjson_root,
                "RAPIDJSON_INCLUDE_DIR": os.path.join(rapidjson_root, "include"),
                "TRITON_ENABLE_CC_HTTP": "ON",
                "TRITON_ENABLE_CC_GRPC": "OFF",
                "TRITON_ENABLE_GRPC": "OFF",
                "TRITON_ENABLE_GRPC_V2": "OFF",
                "TRITON_ENABLE_PERF_ANALYZER": "OFF",
                "TRITON_ENABLE_CC_S3": "OFF",
                "TRITON_COMMON_REPO_TAG": "r24.10",
                "TRITON_THIRD_PARTY_REPO_TAG": "r24.10",
                "TRITON_CORE_REPO_TAG": "r24.10",
                "BUILD_SHARED_LIBS": "ON" if self.options.shared else "OFF",
            },
        )

        cache_dir = os.path.join(self.build_folder, "cc-clients", "tmp")
        cache_file = os.path.join(cache_dir, f"cc-clients-cache-{self.settings.build_type}.cmake")
        os.makedirs(cache_dir, exist_ok=True)
        with open(cache_file, "a", encoding="utf-8") as f:
            f.write("\n")
            f.write(f'set(RapidJSON_DIR "{rapidjson_cmake_dir}" CACHE PATH "Initial cache" FORCE)\n')
            f.write(f'set(RapidJSON_ROOT "{rapidjson_root}" CACHE PATH "Initial cache" FORCE)\n')
            f.write(
                f'set(RAPIDJSON_INCLUDE_DIR "{os.path.join(rapidjson_root, "include")}" CACHE PATH "Initial cache" FORCE)\n'
            )
            f.write(
                f'set(RAPIDJSON_INCLUDE_DIRS "{os.path.join(rapidjson_root, "include")}" CACHE PATH "Initial cache" FORCE)\n'
            )

        cmake.build(target="cc-clients")

    def package(self):
        cmake = CMake(self)
        cmake.install()

    def package_info(self):
        # Устанавливаем стандартные пути
        self.cpp_info.includedirs = ["include"]
        self.cpp_info.libdirs = ["lib"]

        # self.cpp_info.components["common"].set_property("cmake_target_name", "TritonCommon::TritonCommon")
        # self.cpp_info.components["common"].libs = ["tritoncommon"]
        #
        # self.cpp_info.components["core"].set_property("cmake_target_name", "TritonCore::TritonCore")
        # self.cpp_info.components["core"].libs = ["tritoncore"]

        # self.cpp_info.components["grpc_client"].set_property("cmake_target_name", "TritonClient::grpcclient")
        # self.cpp_info.components["grpc_client"].libs = ["grpcclient"]
        # self.cpp_info.components["grpc_client"].requires = ["common", "core"]

        self.cpp_info.components["http_client"].set_property("cmake_target_name", "TritonClient::httpclient")
        self.cpp_info.components["http_client"].libs = ["httpclient"]
        # self.cpp_info.components["http_client"].requires = ["common", "core"]

        # CMake интеграция
        self.cpp_info.set_property("cmake_file_name", "TritonClient")
        self.cpp_info.set_property("cmake_target_name", "TritonClient::TritonClient")
        self.cpp_info.builddirs = ["lib/cmake"]