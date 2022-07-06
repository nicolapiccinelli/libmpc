cmake_minimum_required(VERSION 3.2)
set(CMAKE_CXX_STANDARD 20)

## Specify a project name
project(mpc++ VERSION 1.0.0 LANGUAGES CXX)
set(USE_SHOW_STACKTRACE false)

# Disabling Eigen stack allocation warning
add_definitions(-DEIGEN_STACK_ALLOCATION_LIMIT=0)

## Load CMAKE configuration from environment variables
set(CMAKE_MODULE_PATH $ENV{CMAKE_MODULE_PATH})
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR}/cmake/modules)                  
set(CMAKE_PREFIX_PATH $ENV{CMAKE_PREFIX_PATH})    

# NLopt by default is not installed in the system 
# so we need to provide the right path
if(APPLE OR UNIX)
    link_directories("/usr/local/lib/")
endif()

## On Apple system Eigen3 is not installed in the default path
## so we include the installed path
if(APPLE)
    include_directories("/usr/local/include")
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
# Find OSQP library and headers
find_package(osqp REQUIRED)
# Find nlopt library and headers
find_package(NLopt REQUIRED)

## Include the Eigen3 library to the project
include_directories(${EIGEN3_INCLUDE_DIR})
include_directories(${OSQP_INCLUDE_DIR})
include_directories("include")

## Enable definition to enable stacktrace print
## https://www.boost.org/doc/libs/1_65_0/doc/html/stacktrace/configuration_and_build.html
if(USE_SHOW_STACKTRACE)
    find_package(Boost REQUIRED COMPONENTS stacktrace_basic stacktrace_backtrace stacktrace_addr2line stacktrace_noop)
    add_definitions(-DSHOW_STACKTRACE=1 -DBOOST_STACKTRACE_USE_ADDR2LINE)
else()
    add_definitions(-DSHOW_STACKTRACE=0)
endif()

## Creating mpc++ interface library
set(MPC_INCLUDE_DIR "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>/include/mpc")
file(GLOB MPC_HEADERS "${MPC_INCLUDE_DIR}/*")

add_library(mpc++ INTERFACE)
target_link_libraries(mpc++ INTERFACE nlopt m osqp)
target_include_directories(mpc++ INTERFACE ${EIGEN3_INCLUDE_DIR} ${OSQP_INCLUDE_DIR})
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

install(EXPORT mpc++Targets DESTINATION "lib/cmake/mpc++")
install(FILES "${PROJECT_BINARY_DIR}/mpc++ConfigVersion.cmake"
              "${PROJECT_BINARY_DIR}/mpc++Config.cmake"
        DESTINATION "lib/cmake/mpc++")

install(DIRECTORY "${PROJECT_SOURCE_DIR}/include/" DESTINATION "include")

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    ## Adding test subdirectory just in case of debug
    enable_testing()
    add_subdirectory(test)
endif()