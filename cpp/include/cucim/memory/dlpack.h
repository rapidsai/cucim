/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2022, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef CUCIM_DLPACK_H
#define CUCIM_DLPACK_H

#include "dlpack/dlpack.h"
#include <fmt/format.h>

namespace cucim::memory
{

/**
* @brief Return a string providing the basic type of the homogeneous array in NumPy.
*
* Note: This method assumes little-endian for now.
*
* @return A const character pointer that represents a string from a given `DLDataType` data.
*/
inline const char* to_numpy_dtype(const DLDataType& dtype) {
    // TODO: consider bfloat16: https://github.com/dmlc/dlpack/issues/45
    // TODO: consider other byte-order
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
    size_t size() const
    {
        size_t size = 1;
        for (int i = 0; i < tensor_->ndim; ++i)
        {
            size *= tensor_->shape[i];
        }
        size *= (tensor_->dtype.bits * tensor_->dtype.lanes + 7) / 8;
        return size;
    }

    DLDataType dtype() const {
        if (!tensor_) {
            return DLDataType({ DLDataTypeCode::kDLUInt, 8, 1 });
        }
        return tensor_->dtype;
    }
    /**
     * @brief Return a string providing the basic type of the homogeneous array in NumPy.
     *
     * Note: This method assumes little-endian for now.
     *
     * @return A const character pointer that represents a string
     */
    const char* numpy_dtype() const {
        // TODO: consider bfloat16: https://github.com/dmlc/dlpack/issues/45
        // TODO: consider other byte-order
        if (!tensor_) {
            return "";
        }
        return to_numpy_dtype(tensor_->dtype);
    }

    operator bool() const
    {
        return static_cast<bool>(tensor_);
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
