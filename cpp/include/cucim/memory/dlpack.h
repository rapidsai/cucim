/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef CUCIM_DLPACK_H
#define CUCIM_DLPACK_H

#include "dlpack/dlpack.h"
#include <fmt/format.h>

namespace cucim::memory
{

class DLTContainer
{
public:
    DLTContainer() = delete;
    DLTContainer(DLTensor* handle, char* shm_name = nullptr) : tensor_(handle), shm_name_(shm_name)
    {
    }

    /**
     * @brief Return the size of memory required to store the contents of data.
     *
     * @return size_t Required size for the tensor.
     */
    size_t size()
    {
        size_t size = 1;
        for (int i = 0; i < tensor_->ndim; ++i)
        {
            size *= tensor_->shape[i];
        }
        size *= (tensor_->dtype.bits * tensor_->dtype.lanes + 7) / 8;
        return size;
    }

    /**
     * @brief Return a string providing the basic type of the homogenous array in NumPy.
     *
     * Note: This method assumes little-endian for now.
     *
     * @return A const character pointer that represents a string
     */
    const char* numpy_dtype() {
        // TODO: consider bfloat16: https://github.com/dmlc/dlpack/issues/45
        // TODO: consider other byte-order
        if (!tensor_) {
            return "";
        }

        const DLDataType& dtype = tensor_->dtype;
        uint8_t code = dtype.code;
        uint8_t bits = dtype.bits;
        switch(code) {

        case kDLInt:
            switch(bits) {
            case 8:
                return "|i1";
            case 16:
                return "<i2";
            case 32:
                return "<i4";
            case 64:
                return "<i8";
            }
            throw std::logic_error(fmt::format("DLDataType(code: kDLInt, bits: {}) is not supported!", bits));
        case kDLUInt:
            switch(bits) {
            case 8:
                return "|u1";
            case 16:
                return "<u2";
            case 32:
                return "<u4";
            case 64:
                return "<u8";
            }
            throw std::logic_error(fmt::format("DLDataType(code: kDLUInt, bits: {}) is not supported!", bits));
        case kDLFloat:
            switch(bits) {
            case 16:
                return "<f2";
            case 32:
                return "<f4";
            case 64:
                return "<f8";
            }
            break;
        case kDLBfloat:
            throw std::logic_error(fmt::format("DLDataType(code: kDLBfloat, bits: {}) is not supported!", bits));
        }
        throw std::logic_error(fmt::format("DLDataType(code: {}, bits: {}) is not supported!", code, bits));
    }

    operator DLTensor() const
    {
        if (tensor_)
        {
            return *tensor_;
        }
        else
        {
            return DLTensor{};
        }
    }

    operator DLTensor*() const
    {
        return tensor_;
    }

private:
    DLTensor* tensor_ = nullptr;
    char* shm_name_ = nullptr;
};

} // namespace cucim::memory

#endif // CUCIM_DLPACK_H
