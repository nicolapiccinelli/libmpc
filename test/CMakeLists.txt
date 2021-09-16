if(UNIX)
    set(CMAKE_CXX_FLAGS "-Wall -Werror -Wextra -std=c++17 -Wfatal-errors -Werror=return-type -g -O0 -no-pie -fno-pie")
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
    setup_target_for_coverage(test_cases_dynamic_coverage test_cases_dynamic coverage)
    setup_target_for_coverage(test_cases_static_coverage test_cases_static coverage)
    setup_target_for_coverage(test_logger_coverage test_logger coverage)
endif()

set(MPC_TEST_LIB_SOURCES
    "test_constraints.cpp" 
    "test_common.cpp"
    "test_objective.cpp"
    "test_main.cpp")

set(MPC_TEST_LOGGER_SOURCES
    "test_logger.cpp" 
    "test_main.cpp")

set(MPC_TEST_CASES_SOURCES
    "test_main.cpp" 
    "test_discrete_lti_siso.cpp"
    "test_vanderpol.cpp")

set(MPC_TEST_LINEAR_SOURCES
    "test_lmpc.cpp"
    "test_main.cpp")

set(MPC_TEST_UTILS_SOURCES
    "test_utils.cpp"
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

add_executable(test_logger ${MPC_TEST_LOGGER_SOURCES})
target_link_libraries(test_logger ${MPC_LINK_LIB})
target_compile_definitions(test_logger PUBLIC debug)
catch_discover_tests(test_logger)

add_executable(test_cases_dynamic ${MPC_TEST_CASES_SOURCES})
target_link_libraries(test_cases_dynamic ${MPC_LINK_LIB})
target_compile_definitions(test_cases_dynamic PUBLIC debug)
target_compile_definitions(test_cases_dynamic PUBLIC MPC_DYNAMIC)
catch_discover_tests(test_cases_dynamic)

add_executable(test_cases_static ${MPC_TEST_CASES_SOURCES})
target_link_libraries(test_cases_static ${MPC_LINK_LIB})
target_compile_definitions(test_cases_static PUBLIC debug)
catch_discover_tests(test_cases_static)

add_executable(test_lmpc_static ${MPC_TEST_LINEAR_SOURCES})
target_link_libraries(test_lmpc_static ${MPC_LINK_LIB})
target_compile_definitions(test_lmpc_static PUBLIC debug)
catch_discover_tests(test_lmpc_static)

add_executable(test_lmpc_dynamic ${MPC_TEST_LINEAR_SOURCES})
target_link_libraries(test_lmpc_dynamic ${MPC_LINK_LIB})
target_compile_definitions(test_lmpc_dynamic PUBLIC debug)
target_compile_definitions(test_lmpc_dynamic PUBLIC MPC_DYNAMIC)
catch_discover_tests(test_lmpc_dynamic)

add_executable(test_utils ${MPC_TEST_UTILS_SOURCES})
target_link_libraries(test_utils ${MPC_LINK_LIB})
target_compile_definitions(test_utils PUBLIC debug)
catch_discover_tests(test_utils)

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
    target_link_libraries(test_logger ${STACKTRACE_LIBS})
    target_link_libraries(test_lmpc_static ${STACKTRACE_LIBS})
    target_link_libraries(test_lmpc_dynamic ${STACKTRACE_LIBS})
    target_link_libraries(test_utils ${STACKTRACE_LIBS})
endif()