/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef PYCUCIM_COMPAT_H
#define PYCUCIM_COMPAT_H

#include <cstdint>
#include <stdexcept>
#include <string>
#include <type_traits>

#include <fmt/format.h>
#include <nanobind/nanobind.h>

namespace nb = nanobind;

namespace cucim
{

/**
 * A bool parameter that converts like pybind11 did: besides True and False, it accepts None and any object whose
 * type defines `__bool__` (e.g., numpy.bool_, 0, 1). nanobind's own bool caster accepts only True and False.
 */
struct Bool
{
    bool value = false;
    operator bool() const
    {
        return value;
    }
};

} // namespace cucim

NAMESPACE_BEGIN(NB_NAMESPACE)
NAMESPACE_BEGIN(detail)

template <>
struct type_caster<cucim::Bool>
{
    NB_TYPE_CASTER(cucim::Bool, const_name("bool"))

    bool from_python(handle src, uint32_t flags, cleanup_list*) noexcept
    {
        if (src.is(Py_True) || src.is(Py_False))
        {
            value.value = src.is(Py_True);
            return true;
        }
        if (!(flags & (uint32_t)cast_flags::convert))
        {
            return false;
        }
        if (src.is_none())
        {
            value.value = false;
            return true;
        }
        auto nb_bool = reinterpret_cast<inquiry>(PyType_GetSlot(Py_TYPE(src.ptr()), Py_nb_bool));
        int result = nb_bool ? nb_bool(src.ptr()) : -1;
        if (result == 0 || result == 1)
        {
            value.value = result == 1;
            return true;
        }
        PyErr_Clear();
        return false;
    }

    static handle from_cpp(cucim::Bool src, rv_policy, cleanup_list*) noexcept
    {
        return src.value ? Py_NewRef(Py_True) : Py_NewRef(Py_False);
    }
};

// Replaces <nanobind/stl/string.h> (never include both): like pybind11, std::string parameters also accept bytes and
// bytearray.
template <>
struct type_caster<std::string>
{
    NB_TYPE_CASTER(std::string, const_name("str"))

    bool from_python(handle src, uint32_t, cleanup_list*) noexcept
    {
        Py_ssize_t size = 0;
        const char* data = nullptr;
        if (PyUnicode_Check(src.ptr()))
        {
            data = PyUnicode_AsUTF8AndSize(src.ptr(), &size);
        }
        else if (PyBytes_Check(src.ptr()))
        {
            char* bytes = nullptr;
            data = PyBytes_AsStringAndSize(src.ptr(), &bytes, &size) == 0 ? bytes : nullptr;
        }
        else if (PyByteArray_Check(src.ptr()))
        {
            data = PyByteArray_AsString(src.ptr());
            size = PyByteArray_Size(src.ptr());
        }
        else
        {
            return false;
        }
        if (data == nullptr)
        {
            PyErr_Clear();
            return false;
        }
        value.assign(data, static_cast<size_t>(size));
        return true;
    }

    static handle from_cpp(const std::string& value, rv_policy, cleanup_list*) noexcept
    {
        return PyUnicode_FromStringAndSize(value.data(), static_cast<Py_ssize_t>(value.size()));
    }
};

NAMESPACE_END(detail)
NAMESPACE_END(NB_NAMESPACE)

namespace cucim
{

/**
 * nanobind enums are `enum.Enum` subclasses without `int()` support and with `Name.Member` as repr.
 * Callers rely on `int(member)`, so every bound enum also gets `__int__`, `__index__`, and
 * pybind11's `<Name.Member: value>` repr. As in pybind11, members of an unscoped enum also compare equal to ints.
 */
template <typename T>
nb::enum_<T> bind_enum(nb::handle scope, const char* name)
{
    constexpr bool is_unscoped = std::is_convertible_v<T, std::underlying_type_t<T>>;
    nb::enum_<T> e = is_unscoped ? nb::enum_<T>(scope, name, nb::is_arithmetic()) : nb::enum_<T>(scope, name);
    e.def("__int__", [](T value) { return static_cast<int64_t>(value); });
    e.def("__index__", [](T value) { return static_cast<int64_t>(value); });
    e.def("__repr__", [name](nb::handle self) {
        return nb::str("<{}.{}: {}>").format(name, self.attr("name"), self.attr("value"));
    });
    return e;
}

template <typename T>
T cast_or_runtime_error(nb::handle value)
{
    T result;
    if (!nb::try_cast<T>(value, result))
    {
        throw std::runtime_error(
            fmt::format("Unable to cast Python instance of type {} to C++ type", nb::type_name(value.type()).c_str()));
    }
    return result;
}

} // namespace cucim

#endif // PYCUCIM_COMPAT_H
