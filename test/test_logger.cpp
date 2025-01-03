/*
 *   Copyright (c) 2023 Nicola Piccinelli
 *   All rights reserved.
 */
#include "basic.hpp"
#include <catch2/catch_test_macros.hpp>

TEST_CASE("Logger outputs log messages based on log level", "[Logger]")
{
    // Set up the logger
    mpc::Logger &logger = mpc::Logger::instance();
    std::ostringstream oss;
    logger.setStream(&oss);

    SECTION("Log level set to NONE")
    {
        logger.setLevel(mpc::Logger::LogLevel::NONE);

        SECTION("Log messages are not output regardless of log type")
        {
            logger.log(mpc::Logger::LogType::DETAIL) << "Detail message";
            logger.log(mpc::Logger::LogType::INFO) << "Info message";
            logger.log(mpc::Logger::LogType::ERROR) << "Error message";

            REQUIRE(oss.str().empty());
        }
    }

    SECTION("Log level set to DEEP")
    {
        logger.setLevel(mpc::Logger::LogLevel::DEEP);

        SECTION("Log messages are output when log level allows")
        {
            logger.log(mpc::Logger::LogType::DETAIL) << "Detail message";
            logger.log(mpc::Logger::LogType::INFO) << "Info message";
            logger.log(mpc::Logger::LogType::ERROR) << "Error message";

            REQUIRE(oss.str() == "[MPC++] Detail message[MPC++] Info message[MPC++] Error message");
        }
    }

    SECTION("Log level set to NORMAL")
    {
        logger.setLevel(mpc::Logger::LogLevel::NORMAL);

        SECTION("Log messages are output for INFO and ERROR log types")
        {
            logger.log(mpc::Logger::LogType::DETAIL) << "Detail message";
            logger.log(mpc::Logger::LogType::INFO) << "Info message";
            logger.log(mpc::Logger::LogType::ERROR) << "Error message";

            REQUIRE(oss.str() == "[MPC++] Info message[MPC++] Error message");
        }
    }

    SECTION("Log level set to ALERT")
    {
        logger.setLevel(mpc::Logger::LogLevel::ALERT);

        SECTION("Log messages are output for ERROR log type")
        {
            logger.log(mpc::Logger::LogType::DETAIL) << "Detail message";
            logger.log(mpc::Logger::LogType::INFO) << "Info message";
            logger.log(mpc::Logger::LogType::ERROR) << "Error message";

            REQUIRE(oss.str() == "[MPC++] Error message");
        }
    }
}

TEST_CASE("Logger sets prefix for log messages", "[Logger]")
{
    // Set up the logger
    mpc::Logger &logger = mpc::Logger::instance();
    std::ostringstream oss;
    logger.setLevel(mpc::Logger::LogLevel::NORMAL);
    logger.setStream(&oss);

    SECTION("Prefix is set")
    {
        logger.setPrefix("PREFIX");

        SECTION("Log messages include the prefix")
        {
            logger.log(mpc::Logger::LogType::INFO) << "Message";

            REQUIRE(oss.str() == "[MPC++ PREFIX] Message");
        }
    }

    SECTION("Prefix is not set")
    {
        logger.setPrefix("");

        SECTION("Log messages do not include the prefix")
        {
            logger.log(mpc::Logger::LogType::INFO) << "Message";

            REQUIRE(oss.str() == "[MPC++] Message");
        }
    }
}

TEST_CASE("Logger reset function", "[Logger]")
{
    // Set up the logger
    mpc::Logger &logger = mpc::Logger::instance();
    std::ostringstream oss;
    logger.setStream(&oss);
    logger.setPrefix("PREFIX");
    logger.setLevel(mpc::Logger::LogLevel::DEEP);

    SECTION("Logger is reset to default configuration")
    {
        logger.reset();
        logger.log(mpc::Logger::LogType::DETAIL) << "Info message";

        REQUIRE(oss.str().empty());
    }
}