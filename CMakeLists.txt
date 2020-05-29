cmake_minimum_required( VERSION 3.2 )

## Specify a project name
project(mpc++)

## Load CMAKE configuration from environment variables
set(CMAKE_MODULE_PATH $ENV{CMAKE_MODULE_PATH})
set(CMAKE_PREFIX_PATH $ENV{CMAKE_PREFIX_PATH})                      

## Find Eigen3 to build the library
find_package (Eigen3 3.3 REQUIRED NO_MODULE)

## Include the Eigen3 library to the project
include_directories(${EIGEN3_INCLUDE_DIR})
include_directories(include)

## On Apple system Eigen3 is not installed in the default path
## so we include the installed path
if(APPLE)
    include_directories(/usr/local/include)
    link_directories(/usr/local/lib/)
endif()

if(WIN32)
    include_directories("../eigen")
    include_directories("../nlopt/build")
    include_directories("../nlopt/src/api")
    link_directories("../nlopt/build/Release")
endif()

## Creating mpc++ interface library
add_library(mpc++ INTERFACE)
target_link_libraries(mpc++ INTERFACE nlopt)
target_include_directories(mpc++ INTERFACE ${EIGEN3_INCLUDE_DIR})

include(CMakePackageConfigHelpers)
write_basic_package_version_file(
    "${PROJECT_BINARY_DIR}/mpc++ConfigVersion.cmake"
    VERSION 0.1
    COMPATIBILITY AnyNewerVersion
)

install(TARGETS mpc++
    EXPORT mpc++Targets
    LIBRARY DESTINATION lib COMPONENT Runtime
    ARCHIVE DESTINATION lib COMPONENT Development
    RUNTIME DESTINATION bin COMPONENT Runtime
    PUBLIC_HEADER DESTINATION include COMPONENT Development
    BUNDLE DESTINATION bin COMPONENT Runtime
)

include(CMakePackageConfigHelpers)
configure_package_config_file(
    "${PROJECT_SOURCE_DIR}/cmake/mpc++Config.cmake.in"
    "${PROJECT_BINARY_DIR}/mpc++Config.cmake"
    INSTALL_DESTINATION lib/cmake/mpc++
)

install(EXPORT mpc++Targets DESTINATION lib/cmake/mpc++)
install(FILES "${PROJECT_BINARY_DIR}/mpc++ConfigVersion.cmake"
              "${PROJECT_BINARY_DIR}/mpc++Config.cmake"
        DESTINATION lib/cmake/mpc++)

install(DIRECTORY ${PROJECT_SOURCE_DIR}/include/ DESTINATION /usr/local/include)

## Testing area
if(NOT WIN32)
    set(CMAKE_CXX_FLAGS "-Wall -Wextra -std=c++1z -Wfatal-errors -g -O0 -rdynamic")
endif()
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)

find_package(OpenMP)
if (OPENMP_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

add_executable(vanderpol test/vanderpol.cpp)
target_link_libraries(vanderpol mpc++ ${OpenMP_CXX_LIBRARIES})
target_compile_definitions(vanderpol PUBLIC debug)

add_executable(linearProblem test/linearProblem.cpp)
target_link_libraries(linearProblem mpc++ ${OpenMP_CXX_LIBRARIES})
target_compile_definitions(linearProblem PUBLIC debug)

#if(NOT APPLE)
#    include_directories(${PROJECT_SOURCE_DIR}/deps/stacktrace)
#    add_library(stacktrace SHARED ${PROJECT_SOURCE_DIR}/deps/stacktrace/stacktrace.cpp)
#    set_property(TARGET stacktrace PROPERTY POSITION_INDEPENDENT_CODE ON)
#    target_link_libraries(vanderpol stacktrace bfd iberty)
#endif()
