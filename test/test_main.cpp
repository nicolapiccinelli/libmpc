/*
 *   Copyright (c) 2023-2025 Nicola Piccinelli
 *   All rights reserved.
 */
#include <catch2/catch_session.hpp>

int main(int argc, char *argv[])
{
    int result = Catch::Session().run(argc, argv);
    return result;
}