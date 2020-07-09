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

## Testing area
enable_testing()

if(UNIX)
    set(CMAKE_CXX_FLAGS "-Wall -Wextra -std=c++17 -Wfatal-errors -Werror=return-type -g -O0")
endif()

if(WIN32)
    set(CMAKE_CXX_FLAGS "/std:c++17")
endif()

set(EXECUTABLE_OUTPUT_PATH "${PROJECT_SOURCE_DIR}/bin")

find_package(OpenMP)
if (OPENMP_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

find_package(Catch2 REQUIRED)
include_directories(test/include)

if(UNIX)
    set(CMAKE_CXX_FLAGS "-rdynamic")
    include_directories("${PROJECT_SOURCE_DIR}/deps/stacktrace")
    add_library(stacktrace SHARED "${PROJECT_SOURCE_DIR}/deps/stacktrace/stacktrace.cpp")
    set_property(TARGET stacktrace PROPERTY POSITION_INDEPENDENT_CODE ON)
endif()


include(CTest)
include(Catch)


set(MPC_LINK_LIB 
    "mpc++" 
    "Catch2::Catch2" 
    ${OpenMP_CXX_LIBRARIES})

set(MPC_TEST_LIB_SOURCES
    "test/test_constraints.cpp" 
    "test/test_common.cpp"
    "test/test_objective.cpp"
    "test/test_main.cpp")

add_executable(test_lib_dynamic ${MPC_TEST_LIB_SOURCES})
target_link_libraries(test_lib_dynamic ${MPC_LINK_LIB})
target_compile_definitions(test_lib_dynamic PUBLIC debug)
target_compile_definitions(test_lib_dynamic PUBLIC MPC_DYNAMIC)
catch_discover_tests(test_lib_dynamic)

add_executable(test_lib_static ${MPC_TEST_LIB_SOURCES})
target_link_libraries(test_lib_static ${MPC_LINK_LIB})
target_compile_definitions(test_lib_static PUBLIC debug)
catch_discover_tests(test_lib_static)


set(MPC_TEST_LOGGER_SOURCES
"test/test_logger.cpp" 
"test/test_main.cpp")

add_executable(test_logger ${MPC_TEST_LOGGER_SOURCES})
target_link_libraries(test_logger ${MPC_LINK_LIB})
target_compile_definitions(test_logger PUBLIC debug)
catch_discover_tests(test_logger)

set(MPC_TEST_CASES_SOURCES
    "test/test_main.cpp" 
    "test/test_discrete_lti_siso.cpp"
    "test/test_vanderpol.cpp")

add_executable(test_cases_dynamic ${MPC_TEST_CASES_SOURCES})
target_link_libraries(test_cases_dynamic ${MPC_LINK_LIB})
target_compile_definitions(test_cases_dynamic PUBLIC debug)
target_compile_definitions(test_cases_dynamic PUBLIC MPC_DYNAMIC)
catch_discover_tests(test_cases_dynamic)

add_executable(test_cases_static ${MPC_TEST_CASES_SOURCES})
target_link_libraries(test_cases_static ${MPC_LINK_LIB})
target_compile_definitions(test_cases_static PUBLIC debug)
catch_discover_tests(test_cases_static)

if(UNIX)
    set(MPC_TEST_UNIX_LINK_LIB
        "stacktrace" 
        "bfd" 
        "iberty")

    target_link_libraries(test_lib_dynamic ${MPC_TEST_UNIX_LINK_LIB})
    target_link_libraries(test_lib_static ${MPC_TEST_UNIX_LINK_LIB})
    target_link_libraries(test_cases_dynamic ${MPC_TEST_UNIX_LINK_LIB})
    target_link_libraries(test_cases_static ${MPC_TEST_UNIX_LINK_LIB})
    target_link_libraries(test_logger ${MPC_TEST_UNIX_LINK_LIB})
endif()