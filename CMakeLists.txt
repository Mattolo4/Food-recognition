cmake_minimum_required(VERSION 3.25.2)
project(Food-recognition VERSION 0.1.0)

include(CTest)
enable_testing() 
find_package(OpenCV REQUIRED)

include_directories( ${OpenCV_INCLUDE_DIRS})

# Current dir
add_executable(FProj FProj.cpp Utilities.cpp) 
target_link_libraries( FProj ${OpenCV_LIBS} )

add_executable(3DHist 3DHist.cpp Utilities.cpp)
target_link_libraries( 3DHist ${OpenCV_LIBS} ${OPENGL_LIBRARIES} ${GLUT_LIBRARY})

add_executable(main main.cpp)
target_link_libraries( main ${OpenCV_LIBS})


# Sub-dir
# Add the subdirectory containing your source file
add_subdirectory(Performance-measurements)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack) 
