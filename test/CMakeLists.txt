if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    set(CMAKE_CXX_FLAGS "-Wsuggest-override -Wall -Werror=suggest-override -Wextra -Wfatal-errors -Werror=return-type -g -O0 -no-pie -fno-pie")
    message("Adding g++ specific flags")
endif()

IF(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
    set(CMAKE_CXX_FLAGS "-Wunused-command-line-argument -std=c++20 -Wall -Wextra -Wfatal-errors -Werror=return-type -g -O0")
    message("Adding clang specific flags")
endif()

if(CMAKE_SYSTEM_NAME STREQUAL "Windows")
    set(CMAKE_CXX_FLAGS "/std:c++20")
    message("Adding mvcc specific flags")
endif()

set(EXECUTABLE_OUTPUT_PATH "${PROJECT_SOURCE_DIR}/bin")

find_package(OpenMP)
if (OPENMP_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

find_package(Catch2 REQUIRED)

include_directories(include)
include(CTest)
include(Catch)

set(MPC_LINK_LIB 
    "mpc++" 
    "Catch2::Catch2" 
    ${OpenMP_CXX_LIBRARIES})

if(CMAKE_COMPILER_IS_GNUCXX)
    include(CodeCoverage)
    set(CMAKE_BUILD_TYPE Coverage)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fprofile-arcs -ftest-coverage --coverage")
    setup_target_for_coverage(test_lib_dynamic_coverage test_lib_dynamic coverage)
    setup_target_for_coverage(test_lib_static_coverage test_lib_static coverage)
endif()

set(MPC_TEST_LIB_SOURCES
    "NLMPC/test_constraints.cpp"
    "NLMPC/test_objective.cpp"
    "NLMPC/test_common.cpp"
    "NLMPC/test_nloptimizer.cpp"
    "LMPC/test_lmpc.cpp"
    "LMPC/test_mutiple_instances.cpp"
    "test_utils.cpp"
    "test_profiler.cpp"
    "test_logger.cpp"
    "test_main.cpp")

set(MPC_TEST_CASES_SOURCES
    "NLMPC/test_discrete_lti_siso.cpp"
    "NLMPC/test_vanderpol.cpp"
    "LMPC/test_quadrotor.cpp"
    "test_main.cpp")

add_executable(test_lib_dynamic ${MPC_TEST_LIB_SOURCES})
target_link_libraries(test_lib_dynamic ${MPC_LINK_LIB})
target_compile_definitions(test_lib_dynamic PUBLIC debug)
target_compile_definitions(test_lib_dynamic PUBLIC MPC_DYNAMIC)
catch_discover_tests(test_lib_dynamic)

add_executable(test_lib_static ${MPC_TEST_LIB_SOURCES})
target_link_libraries(test_lib_static ${MPC_LINK_LIB})
target_compile_definitions(test_lib_static PUBLIC debug)
catch_discover_tests(test_lib_static)

add_executable(test_cases_dynamic ${MPC_TEST_CASES_SOURCES})
target_link_libraries(test_cases_dynamic ${MPC_LINK_LIB})
target_compile_definitions(test_cases_dynamic PUBLIC debug)
target_compile_definitions(test_cases_dynamic PUBLIC MPC_DYNAMIC)
catch_discover_tests(test_cases_dynamic)

add_executable(test_cases_static ${MPC_TEST_CASES_SOURCES})
target_link_libraries(test_cases_static ${MPC_LINK_LIB})
target_compile_definitions(test_cases_static PUBLIC debug)
catch_discover_tests(test_cases_static)

if(USE_SHOW_STACKTRACE)
    set(STACKTRACE_LIBS 
        dl
        ${Boost_STACKTRACE_BASIC_LIBRARY}
        ${Boost_STACKTRACE_BACKTRACE_LIBRARY}
        ${Boost_STACKTRACE_ADDR2LINE_LIBRARY}
        ${Boost_STACKTRACE_NOOP_LIBRARY})

    target_link_libraries(test_lib_dynamic ${STACKTRACE_LIBS})
    target_link_libraries(test_lib_static ${STACKTRACE_LIBS})
    target_link_libraries(test_cases_dynamic ${STACKTRACE_LIBS})
    target_link_libraries(test_cases_static ${STACKTRACE_LIBS})
endif()