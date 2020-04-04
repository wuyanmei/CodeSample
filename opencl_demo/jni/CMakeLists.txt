cmake_minimum_required(VERSION 3.6)

# Enable C++11
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

set(target SobelFilter)
#project(${target} CXX)

message(STATUS "ANDROID_ABI=${ANDROID_ABI}")

find_library( # Defines the name of the path variable that stores the
        # location of the NDK library.
        log-lib

        # Specifies the name of the NDK library that
        # CMake needs to locate.
        log )

file(GLOB srcs *.cpp *.c)
file(GLOB hdrs *.hpp *.h ./CL/*.h ./CL/*.hpp)

include_directories("${CMAKE_CURRENT_LIST_DIR}")
add_library(${target} SHARED ${srcs} ${hdrs} )
target_link_libraries(${target}
        ${log-lib}
        )
