/*
 * SPDX-FileCopyrightText: Copyright (c) 2021, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef CUCIM_HASH_FUNCTION_H
#define CUCIM_HASH_FUNCTION_H

#include <cstdint>

namespace cucim::codec
{

/**
 * @brief splitmix64 hash function with three input values
 *
 * This function based on the code from https://xorshift.di.unimi.it/splitmix64.c which is released in the public
 * domain (http://creativecommons.org/publicdomain/zero/1.0/).
 *
 * @param x input state value
 * @return uint64_t
 */
inline uint64_t splitmix64(uint64_t x)
{

    uint64_t z = (x += 0x9e3779b97f4a7c15);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9;
    z = (z ^ (z >> 27)) * 0x94d049bb133111eb;
    return z ^ (z >> 31);
}

/**
 * @brief splitmix64 hash function with three input values
 *
 * This function based on the code from https://xorshift.di.unimi.it/splitmix64.c which is released in the public
 * domain (http://creativecommons.org/publicdomain/zero/1.0/).
 *
 * @param a
 * @param b
 * @param c
 * @return uint64_t
 */
inline uint64_t splitmix64_3(uint64_t a, uint64_t b, uint64_t c)
{

    uint64_t z = (a += 0x9e3779b97f4a7c15);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9;
    z = (z ^ (z >> 27)) * 0x94d049bb133111eb;
    z = z ^ (z >> 31);

    z += (b += 0x9e3779b97f4a7c15);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9;
    z = (z ^ (z >> 27)) * 0x94d049bb133111eb;
    z = z ^ (z >> 31);

    z += (c += 0x9e3779b97f4a7c15);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9;
    z = (z ^ (z >> 27)) * 0x94d049bb133111eb;
    z = z ^ (z >> 31);

    return z;
}

} // namespace cucim::codec

#endif // CUCIM_HASH_FUNCTION_H
