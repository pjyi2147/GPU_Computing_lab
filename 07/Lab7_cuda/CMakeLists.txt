# 3.10 for CUDA language support
# 3.17 for FindCUDAToolkit
# 3.18 for CMAKE_CUDA_ARCHITECTURES
cmake_minimum_required(VERSION 3.17 FATAL_ERROR)

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_COLOR_MAKEFILE ON)
set(VERBOSE_BUILD OFF)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
set(CMAKE_MACOSX_RPATH TRUE)
set_property(GLOBAL PROPERTY USE_FOLDERS ON)


project(teachingkit-labs CXX CUDA)

set(BUILD_LIBGPUTK_LIBRARY ON CACHE BOOL "Turn on build of libgputk library")
set(BUILD_LOGTIME ON CACHE BOOL "Turn on build of time logging")

list(INSERT CMAKE_MODULE_PATH 0 ${PROJECT_SOURCE_DIR}/support/cmake/modules)

if (BUILD_LIBGPUTK_LIBRARY)
  find_package(CUDAToolkit REQUIRED)
else()
  find_package(CUDAToolkit)
endif()

if (NOT WIN32)
    find_package(EnableCxx11 REQUIRED)
endif (NOT WIN32)


set(CUDA_ATTACH_VS_BUILD_RULE_TO_CUDA_FILE OFF)
set(CUDA_USE_STATIC_CUDA_RUNTIME ON)
set(CUDA_VERBOSE_BUILD OFF)

# allow C99 for cmake versions less than 3.1
set (CMAKE_C_STANDARD 99)


# if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
#   if(CMAKE_CXX_COMPILER_VERSION VERSION_GREATER "4.9.3")
#     if(CUDA_VERSION VERSION_LESS "8.0")
#       message(STATUS "Found gcc >=5 and CUDA <= 7.5, adding workaround C++ flags")
#       set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_FORCE_INLINES")
#     endif(CUDA_VERSION VERSION_LESS "8.0")
#   endif(CMAKE_CXX_COMPILER_VERSION VERSION_GREATER "4.9.3")
# endif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")


# set(CUDA_PROPAGATE_HOST_FLAGS TRUE)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
# https://cmake.org/cmake/help/v3.10/prop_tgt/CUDA_RESOLVE_DEVICE_SYMBOLS.html
# enable device static linking on all targets?
set(CMAKE_CUDA_RESOLVE_DEVICE_SYMBOLS ON)

# For some MPs it may be necessary to specify the compute capability of your
# NVIDIA GPU. In that case, simply uncomment the following two lines that start
# with 'set' and change the value of COMPUTE_CAPABILITY to one of the following
# hardware architectures: Tesla-class '11', '12', or '13'; Fermi-class '20'
# or '21'; Kepler-class '30', '32', '35'; or Maxwell-class '50' or '52'
# CUDA_SELECT_NVCC_ARCH_FLAGS(ARCH_FLAGS 3.5+PTX 5.0 Maxwell 6.1 7.0 7.5)
# list(APPEND CUDA_NVCC_FLAGS ${ARCH_FLAGS})
# https://cmake.org/cmake/help/latest/variable/CMAKE_CUDA_ARCHITECTURES.html
# set(CMAKE_CUDA_ARCHITECTURES 50)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=sm_50")
message(STATUS ${CMAKE_CUDA_FLAGS})

# Set warning levels for host compilation
if (UNIX)
     add_compile_options(-Wno-unused-function)
     add_compile_options(-Wno-unused-parameter)
     add_compile_options(-Wextra)
     add_compile_options(-Wno-unused-variable)
endif ()


add_definitions(-DGPUTK_USE_CUDA)

# Select the correct host compiler on OS X
if (APPLE)
    if (${CMAKE_SYSTEM_VERSION} LESS 12.0)
        # OS X 10.7 or less
        find_program(GCC_EXECUTABLE NAMES g++ PATHS ENV PATH)
        if (NOT GCC_EXECUTABLE)
            message(FATAL_ERROR "Could not find host compiler 'g++'")
        endif(NOT GCC_EXECUTABLE)
        set(CUDA_HOST_COMPILER ${GCC_EXECUTABLE})
    else (${CMAKE_SYSTEM_VERSION} LESS 12.0)
        # OS X 10.8+
        find_program(CLANG_EXECUTABLE NAMES clang++-3.6 clang++ PATHS ENV PATH)
        if (NOT CLANG_EXECUTABLE)
            message(FATAL_ERROR "Could not find host compiler 'clang++'")
        endif ()
        # set(CUDA_HOST_COMPILER ${CLANG_EXECUTABLE})
        set(CMAKE_CUDA_HOST_COMPILER ${CLANG_EXECUTABLE})
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -isystem ${CUDA_INCLUDE_DIRS}")
		set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -stdlib=libc++" )
    endif (${CMAKE_SYSTEM_VERSION} LESS 12.0)
endif (APPLE)

# Set the correct linker options on UNIX platforms
if (UNIX)
    set(LINK_LIBRARIES "-lm -lcuda")
    if (APPLE)
        find_library(CUDA_LIBRARY NAMES CUDA)
        get_filename_component(FRAMEWORK_DIRECTORY ${CUDA_LIBRARY} DIRECTORY)
        set(LINK_LIBRARIES "${LINK_LIBRARIES} -F${FRAMEWORK_DIRECTORY} -framework CUDA")
    else (APPLE)
        include(CheckLibraryExists)
        check_library_exists(rt clock_gettime "time.h" HAVE_CLOCK_GETTIME)
        if (HAVE_CLOCK_GETTIME)
            set(LINK_LIBRARIES "${LINK_LIBRARIES} -lrt")
        else (HAVE_CLOCK_GETTIME)
            message(FATAL_ERROR "clock_gettime() not found")
        endif (HAVE_CLOCK_GETTIME)
    endif (APPLE)
endif (UNIX)

macro(add_lab_helper NAME SOURCES)
    add_executable(${NAME} ${SOURCES})
    target_link_libraries(${NAME} ${GPUTKLIB} ${LINK_LIBRARIES})
    target_link_libraries(${NAME} CUDA::cudart)
endmacro(add_lab_helper)

macro(add_lab_libs NAME LIBRARIES)
    target_link_libraries(${NAME} ${LIBRARIES})
endmacro(add_lab_libs)

include_directories(${PROJECT_SOURCE_DIR}/libgputk)
# CUDA_INCLUDE_DIRECTORIES(${PROJECT_SOURCE_DIR}/libgputk)

if (BUILD_LIBGPUTK_LIBRARY)
    include(libgputk/sources.cmake)
    add_library(${GPUTKLIB} ${LIBGPUTK_SOURCE_FILES}) # TODO: static?
    if (WIN32)
        target_link_libraries(${GPUTKLIB} ${CUDA_cuda_LIBRARY})
    endif (WIN32)
    target_link_libraries(${GPUTKLIB} ${LINK_LIBRARIES})
    target_link_libraries(${GPUTKLIB} CUDA::cudart)
    if (BUILD_LOGTIME)
        set_target_properties(${GPUTKLIB} PROPERTIES COMPILE_FLAGS "${COMPILE_FLAGS} -DgpuTKLogger_TimerLog" )
    endif()
endif (BUILD_LIBGPUTK_LIBRARY)


# include(Module24/sources.cmake)

# These are the OpenCL and OpenACC
# labs and are not supported by the
# current cmake configuration
# include(Module21/sources.cmake)
# include(Module22/sources.cmake)

# Add a special target to clean nvcc generated files.
# CUDA_BUILD_CLEAN_TARGET()
