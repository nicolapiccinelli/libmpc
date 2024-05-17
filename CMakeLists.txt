cmake_minimum_required(VERSION 3.2)
set(CMAKE_CXX_STANDARD 20)
set(LIBMPCC_VERSION 0.5.0)

## Specify a project name
project(mpc++ VERSION ${LIBMPCC_VERSION} LANGUAGES CXX)
set(USE_SHOW_STACKTRACE false)

# Disabling Eigen stack allocation warning
add_definitions(-DEIGEN_STACK_ALLOCATION_LIMIT=0)

## Load CMAKE configuration from environment variables
set(CMAKE_MODULE_PATH $ENV{CMAKE_MODULE_PATH})
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR}/cmake/modules)                  
set(CMAKE_PREFIX_PATH $ENV{CMAKE_PREFIX_PATH})

if(CMAKE_SYSTEM_NAME STREQUAL "Windows")
include_directories(
    "../eigen"
    "../nlopt/include"
    "../Catch2/build")
    link_directories("../nlopt/bin")
endif()

# Find Eigen3 to build the library
find_package(Eigen3 REQUIRED NO_MODULE)
if(Eigen3_FOUND)
    message(STATUS "Found Eigen3")
    message(STATUS "Eigen3 include dir: ${EIGEN3_INCLUDE_DIRS}")
else()
    message(FATAL_ERROR "Could not locate Eigen3")
endif()

# Find OSQP library and headers
find_package(osqp REQUIRED)
if(osqp_FOUND)
    message(STATUS "Found osqp")
    message(STATUS "osqp include dir: ${OSQP_INCLUDE_DIR}")
    message(STATUS "osqp library dir: ${OSQP_LIBRARIES}")
else()
    message(FATAL_ERROR "Could not locate osqp")
endif()

# Find nlopt library and headers
find_package(NLopt REQUIRED)
if(NLopt_FOUND)
    message(STATUS "Found NLopt")
    message(STATUS "NLopt include dir: ${NLOPT_INCLUDE_DIRS}")
    message(STATUS "NLopt library dir: ${NLOPT_LIBRARIES}")
else()
    message(FATAL_ERROR "Could not locate NLopt")
endif()

# Include the external libraries to the project
# This is necessary to include the headers of the external libraries
set(EXTERN_INCLUDE_DIRS ${EIGEN3_INCLUDE_DIRS} ${OSQP_INCLUDE_DIR} ${NLOPT_INCLUDE_DIRS}) 

include_directories(${EXTERN_INCLUDE_DIRS} "include")

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
target_sources(mpc++ INTERFACE ${MPC_HEADERS})
target_include_directories(mpc++ INTERFACE ${EXTERN_INCLUDE_DIRS})
target_link_libraries(mpc++ INTERFACE ${NLOPT_LIBRARIES} m osqp::osqp)

include(CMakePackageConfigHelpers)
write_basic_package_version_file(
    "${PROJECT_BINARY_DIR}/mpc++ConfigVersion.cmake"
    VERSION ${LIBMPCC_VERSION}
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

message(STATUS "Installation prefix: ${CMAKE_INSTALL_PREFIX}")
set(MPC++_INSTALL_INCLUDE_DIR "${CMAKE_INSTALL_PREFIX}/include")

include(CMakePackageConfigHelpers)
configure_package_config_file(
    "${PROJECT_SOURCE_DIR}/cmake/mpc++Config.cmake.in"
    "${PROJECT_BINARY_DIR}/mpc++Config.cmake"
    INSTALL_DESTINATION "lib/cmake/mpc++"
    PATH_VARS MPC++_INSTALL_INCLUDE_DIR
)

install(EXPORT mpc++Targets DESTINATION "lib/cmake/mpc++")
install(FILES "${PROJECT_BINARY_DIR}/mpc++ConfigVersion.cmake"
              "${PROJECT_BINARY_DIR}/mpc++Config.cmake"
        DESTINATION "lib/cmake/mpc++")

install(DIRECTORY "${PROJECT_SOURCE_DIR}/include/" DESTINATION "${MPC++_INSTALL_INCLUDE_DIR}")

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    ## Adding test subdirectory just in case of debug
    enable_testing()
    add_subdirectory(test)
endif()