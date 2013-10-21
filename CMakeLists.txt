cmake_minimum_required(VERSION 2.8)
project( proj )
find_package( OpenCV REQUIRED )
add_executable( proj proj.cpp )
target_link_libraries( proj ${OpenCV_LIBS} )
# TEMP: testing laplacian blend
add_executable(laplacian_blend laplacian_blend.cpp)
target_link_libraries(laplacian_blend ${OpenCV_LIBS})

# Add online example TEMP
add_executable(example_homography example_homography.cpp)
target_link_libraries(example_homography ${OpenCV_LIBS})

