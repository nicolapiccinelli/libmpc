#include "basic.hpp"
#include <catch2/catch_test_macros.hpp>

TEST_CASE("Checking logger prefix", "[logging]")
{
    std::stringstream ss;
    mpc::Logger::instance().
        setStream(&ss).
        setPrefix("test").
        setLevel(mpc::Logger::log_level::NORMAL).
        log(mpc::Logger::log_type::INFO) << "a";

    REQUIRE(ss.str().find("test") != std::string::npos);
}

TEST_CASE("Checking logger verbosity", "[logging]")
{
    std::stringstream ss;
    mpc::Logger::instance().reset().setStream(&ss);

    SECTION("Level NONE")
    {
        mpc::Logger::instance().setLevel(mpc::Logger::log_level::NONE);
        mpc::Logger::instance().log(mpc::Logger::log_type::DETAIL) << "a";
        REQUIRE(ss.str().size() == 0);

        ss.str("");

        mpc::Logger::instance().log(mpc::Logger::log_type::INFO) << "a";
        REQUIRE(ss.str().size() == 0);

        ss.str("");

        mpc::Logger::instance().log(mpc::Logger::log_type::ERROR) << "a";
        REQUIRE(ss.str().size() == 0);
    }

    SECTION("Level DEEP")
    {
        mpc::Logger::instance().setLevel(mpc::Logger::log_level::DEEP);
        mpc::Logger::instance().log(mpc::Logger::log_type::DETAIL) << "a";
        REQUIRE(ss.str().size() > 0);

        ss.str("");

        mpc::Logger::instance().log(mpc::Logger::log_type::INFO) << "a";
        REQUIRE(ss.str().size() > 0);

        ss.str("");

        mpc::Logger::instance().log(mpc::Logger::log_type::ERROR) << "a";
        REQUIRE(ss.str().size() > 0);
    }

    SECTION("Level INFO")
    {
        mpc::Logger::instance().setLevel(mpc::Logger::log_level::NORMAL);
        mpc::Logger::instance().log(mpc::Logger::log_type::DETAIL) << "a";
        REQUIRE(ss.str().size() == 0);

        ss.str("");

        mpc::Logger::instance().log(mpc::Logger::log_type::INFO) << "a";
        REQUIRE(ss.str().size() > 0);

        ss.str("");

        mpc::Logger::instance().log(mpc::Logger::log_type::ERROR) << "a";
        REQUIRE(ss.str().size() > 0);
    }

    SECTION("Level ERROR")
    {
        mpc::Logger::instance().setLevel(mpc::Logger::log_level::ALERT);
        mpc::Logger::instance().log(mpc::Logger::log_type::DETAIL) << "a";
        REQUIRE(ss.str().size() == 0);

        ss.str("");

        mpc::Logger::instance().log(mpc::Logger::log_type::INFO) << "a";
        REQUIRE(ss.str().size() == 0);

        ss.str("");

        mpc::Logger::instance().log(mpc::Logger::log_type::ERROR) << "a";
        REQUIRE(ss.str().size() > 0);
    }
}