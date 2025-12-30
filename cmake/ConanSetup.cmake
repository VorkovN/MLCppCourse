function(setup_conan)
    if(NOT EXISTS "${CMAKE_BINARY_DIR}/conan_toolchain.cmake")
        if(APPLE)
            set(CONAN_PROFILE "macos_release")
        else()
            set(CONAN_PROFILE "linux_release")
        endif()

        find_program(CONAN_EXECUTABLE NAMES conan REQUIRED)

        execute_process(
                COMMAND ${CONAN_EXECUTABLE} install ${CMAKE_SOURCE_DIR}/conan/conanfile.txt
                --output-folder=${CMAKE_BINARY_DIR}
                --build=missing
                --profile:build ${CMAKE_SOURCE_DIR}/conan/profile_${CONAN_PROFILE}
                --profile:host ${CMAKE_SOURCE_DIR}/conan/profile_${CONAN_PROFILE}
                --lockfile ${CMAKE_SOURCE_DIR}/conan/conan.lock
                RESULT_VARIABLE CONAN_RESULT
        )

        if(NOT CONAN_RESULT EQUAL 0)
            message(FATAL_ERROR "Conan install failed with code ${CONAN_RESULT}")
        endif()
    endif()

    include(${CMAKE_BINARY_DIR}/conan_toolchain.cmake)
endfunction()