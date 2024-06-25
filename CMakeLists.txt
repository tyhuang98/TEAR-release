cmake_minimum_required(VERSION 3.10)
project(TEAR)

#compile
#set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_BUILD_TYPE Release)
set (CMAKE_CXX_STANDARD 14)


if(NOT DEFINED CMAKE_SUPPRESS_DEVELOPER_WARNINGS)
    set(CMAKE_SUPPRESS_DEVELOPER_WARNINGS 1 CACHE INTERNAL "No dev warnings")
endif()

# find libraries
find_package(Eigen3 REQUIRED)
find_package(Boost REQUIRED)
find_package(PCL REQUIRED)   #note
find_package(OpenMP REQUIRED)


include_directories(
        include
        ${PCL_INCLUDE_DIRS}
        ${EIGEN3_INCLUDE_DIR}
        ${CMAKE_BINARY_DIR}/pmc-src
)

link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})


# build a lib for the project
FILE(GLOB_RECURSE IncFiles "include/*")
set(SOURCES
        src/registration.cpp
        src/simu_process.cpp
)
add_library(${PROJECT_NAME}_lib SHARED ${IncFiles} ${SOURCES})
target_link_libraries(${PROJECT_NAME}_lib ${PCL_LIBRARIES} Eigen3::Eigen)

# openmp
if (OPENMP_FOUND)
    message("OPENMP FOUND")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif ()


# Change this line to include your own executable file
#add_executable(TEAR test_scalability.cpp)
add_executable(TEAR demo.cpp)
target_link_libraries(TEAR ${PROJECT_NAME}_lib)


