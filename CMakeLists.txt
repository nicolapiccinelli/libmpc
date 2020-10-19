cmake_minimum_required(VERSION 3.2)
set(CMAKE_CXX_STANDARD 17)

## Specify a project name
project(mpc++ VERSION 1.0.0 LANGUAGES CXX)

## Load CMAKE configuration from environment variables
set(CMAKE_MODULE_PATH $ENV{CMAKE_MODULE_PATH})
set(CMAKE_PREFIX_PATH $ENV{CMAKE_PREFIX_PATH})                      

## On Apple system Eigen3 is not installed in the default path
## so we include the installed path
if(APPLE)
    include_directories("/usr/local/include")
    link_directories("/usr/local/lib/")
endif()

if(WIN32)
    include_directories(
        "../eigen"
        "../nlopt/include"
        "../Catch2/build")
    link_directories("../nlopt/bin")
endif()

## Find Eigen3 to build the library
find_package (Eigen3 REQUIRED NO_MODULE)

## Include the Eigen3 library to the project
# set(EIGEN3_INCLUDE_DIR ../eigen)
include_directories(${EIGEN3_INCLUDE_DIR})
include_directories("include")

set(MPC_INCLUDE_DIR "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>/include/mpc/")
file(GLOB MPC_HEADERS "${MPC_INCLUDE_DIR}/*")

## Creating mpc++ interface library
add_library(mpc++ INTERFACE)
target_link_libraries(mpc++ INTERFACE "nlopt")
target_include_directories(mpc++ INTERFACE ${MPC_INCLUDE_DIR} ${EIGEN3_INCLUDE_DIR})
target_sources(mpc++ INTERFACE ${MPC_HEADERS})

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
    INSTALL_DESTINATION "lib/cmake/mpc++"
)

install(EXPORT mpc++Targets 
    DESTINATION "lib/cmake/mpc++")
install(FILES 
    "${PROJECT_BINARY_DIR}/mpc++ConfigVersion.cmake"
    "${PROJECT_BINARY_DIR}/mpc++Config.cmake"
    DESTINATION "lib/cmake/mpc++")

install(DIRECTORY "${PROJECT_SOURCE_DIR}/include/" 
    DESTINATION "/usr/local/include")

## Adding test subdirectory
enable_testing()
add_subdirectory(test)