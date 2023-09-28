
cmake_minimum_required(VERSION 3.25.2)
project(Food-recognition VERSION 0.1.0)

include(CTest)
enable_testing() 
find_package(OpenCV REQUIRED)

include_directories( ${OpenCV_INCLUDE_DIRS})


set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack) 

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include_directories( ${OpenCV_INCLUDE_DIRS})

add_library(Detector_lib STATIC Utilities/Detector_lib.cpp)  # Aggiunta della libreria statica
target_link_libraries(Detector_lib ${OpenCV_LIBS})

add_library(Plate_rec STATIC Utilities/Plate_rec.cpp)  # Aggiunta della libreria statica
target_link_libraries(Plate_rec ${OpenCV_LIBS})


add_executable(main main.cpp Utilities/Utilities.cpp Utilities/mAP.cpp)
target_link_libraries( main ${OpenCV_LIBS}  Detector_lib Plate_rec)
