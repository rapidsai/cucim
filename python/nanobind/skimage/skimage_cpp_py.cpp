/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "contour_assembly.h"
#include "segmented_sort.h"

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

namespace nb = nanobind;
using namespace nanobind::literals;

namespace
{

template <typename T>
void def_segmented_sort(nb::module_& m, const char* temp_storage_bytes_name, const char* sort_name, const char* ranges_name)
{
    m.def(temp_storage_bytes_name, &cucim::skimage::segmented_sort_keys_temp_storage_bytes<T>, "num_items"_a,
          "num_segments"_a,
          "Return the size in bytes of the temporary device storage required by the matching "
          "segmented_sort_keys functions.");
    m.def(sort_name, &cucim::skimage::segmented_sort_keys<T>, "keys_in"_a, "keys_out"_a, "segment_offsets"_a,
          "num_items"_a, "num_segments"_a, "temp_storage"_a, "temp_storage_bytes"_a, "stream_ptr"_a = 0,
          "Sort keys independently within each segment using cub::DeviceSegmentedSort.\n\n"
          "The sort is enqueued asynchronously on ``stream_ptr``. ``temp_storage`` must provide at least the "
          "number of bytes reported by the matching temp_storage_bytes function.");
    m.def(ranges_name, &cucim::skimage::segmented_sort_keys_ranges<T>, "keys_in"_a, "keys_out"_a, "begin_offsets"_a,
          "end_offsets"_a, "num_items"_a, "num_segments"_a, "temp_storage"_a, "temp_storage_bytes"_a, "stream_ptr"_a = 0,
          "Sort selected key ranges using cub::DeviceSegmentedSort.\n\n"
          "The sort is enqueued asynchronously on ``stream_ptr``. ``temp_storage`` must provide at least the "
          "number of bytes reported by the matching temp_storage_bytes function.");
}

using SegmentsArray = nb::ndarray<const double, nb::c_contig, nb::device::cpu>;
using SegmentKeysArray = nb::ndarray<const long long, nb::c_contig, nb::device::cpu>;

// Check the shapes of the inputs of the contour assembly functions.
void check_segments(const SegmentsArray& segments, const SegmentKeysArray& segment_keys)
{
    if (segments.ndim() != 3 || segments.shape(1) != 2 || segments.shape(2) != 2)
    {
        throw nb::value_error("segments must have shape (n_segments, 2, 2)");
    }
    if (segment_keys.ndim() != 2 || segment_keys.shape(0) != segments.shape(0) || segment_keys.shape(1) != 2)
    {
        throw nb::value_error("segment_keys must have shape (n_segments, 2)");
    }
}

// Allocate an uninitialized C-contiguous array whose memory is owned by the
// returned NumPy array. ``data`` receives a pointer to the new memory.
template <typename T>
nb::ndarray<nb::numpy, T> new_array(std::initializer_list<std::size_t> shape, T*& data)
{
    std::size_t size = 1;
    for (std::size_t extent : shape)
    {
        size *= extent;
    }
    data = new T[size];
    nb::capsule owner(data, [](void* p) noexcept { delete[] static_cast<T*>(p); });
    return nb::ndarray<nb::numpy, T>(data, shape, owner);
}

// Assemble marching-squares output segments into ordered contours.
//
// ``segments`` has shape (n_segments, 2, 2): segments[i, 0, :] and
// segments[i, 1, :] are the from and to points in row/column coordinates.
// ``segment_keys`` has shape (n_segments, 2): the topological ID of the source
// grid edge of each endpoint. Returns a list of arrays of shape (n_points, 2),
// one per open or closed contour, in row/column order.
nb::list assemble_contours(SegmentsArray segments, SegmentKeysArray segment_keys)
{
    check_segments(segments, segment_keys);
    const cucim::skimage::ContourAssembly assembly =
        cucim::skimage::build_contour_assembly(segments.data(), segment_keys.data(), segments.shape(0));

    nb::list output;
    for (const cucim::skimage::Contour& contour : assembly.contours)
    {
        if (!contour.active)
        {
            continue;
        }
        double* contour_data;
        auto contour_array = new_array<double>({ contour.size, 2 }, contour_data);
        std::size_t j = 0;
        for (std::size_t node_index = contour.head; node_index != cucim::skimage::invalid_index;
             node_index = assembly.nodes[node_index].next)
        {
            const cucim::skimage::Point& point = assembly.nodes[node_index].point;
            contour_data[2 * j] = point.row;
            contour_data[2 * j + 1] = point.col;
            ++j;
        }
        output.append(nb::cast(std::move(contour_array)));
    }
    return output;
}

// Return contours in packed form: (points, offsets).
//
//     points:  float64 array, shape (total_points, 2)
//     offsets: int64 array, shape (n_contours + 1,)
//
// Contour i is points[offsets[i]:offsets[i + 1]]. This avoids allocating one
// NumPy array per contour and gives callers a single contiguous coordinate
// array that can be transferred to another backend in one operation.
nb::tuple assemble_contours_packed(SegmentsArray segments, SegmentKeysArray segment_keys)
{
    check_segments(segments, segment_keys);
    const cucim::skimage::ContourAssembly assembly =
        cucim::skimage::build_contour_assembly(segments.data(), segment_keys.data(), segments.shape(0));
    const auto [active_contours, total_points] = cucim::skimage::count_active_contours(assembly);

    double* points_data;
    long long* offsets_data;
    auto points = new_array<double>({ total_points, 2 }, points_data);
    auto offsets = new_array<long long>({ active_contours + 1 }, offsets_data);

    std::size_t output_index = 0;
    std::size_t point_index = 0;
    offsets_data[0] = 0;
    for (const cucim::skimage::Contour& contour : assembly.contours)
    {
        if (!contour.active)
        {
            continue;
        }
        for (std::size_t node_index = contour.head; node_index != cucim::skimage::invalid_index;
             node_index = assembly.nodes[node_index].next)
        {
            const cucim::skimage::Point& point = assembly.nodes[node_index].point;
            points_data[2 * point_index] = point.row;
            points_data[2 * point_index + 1] = point.col;
            ++point_index;
        }
        ++output_index;
        offsets_data[output_index] = static_cast<long long>(point_index);
    }
    return nb::make_tuple(std::move(points), std::move(offsets));
}

// Return the contours of segments that were linked on the device, in packed
// form (points, offsets) as ``assemble_contours_packed``.
//
// ``next`` has shape (n_segments,): ``next[i]`` is the segment that starts at
// the end of segment ``i``, or -1. See ``trace_linked_contours``.
nb::tuple assemble_linked_contours(SegmentsArray segments, nb::ndarray<const std::int32_t, nb::c_contig, nb::device::cpu> next)
{
    if (segments.ndim() != 3 || segments.shape(1) != 2 || segments.shape(2) != 2)
    {
        throw nb::value_error("segments must have shape (n_segments, 2, 2)");
    }
    if (next.ndim() != 1 || next.shape(0) != segments.shape(0))
    {
        throw nb::value_error("next must have shape (n_segments,)");
    }
    std::vector<double> point_values;
    std::vector<std::int64_t> offset_values;
    if (!cucim::skimage::trace_linked_contours(segments.data(), next.data(), segments.shape(0), point_values,
                                               offset_values))
    {
        throw nb::value_error("next must link every segment to at most one other, and vice versa");
    }

    // Hand the vectors' memory to NumPy
    auto* points_owner = new std::vector<double>(std::move(point_values));
    auto* offsets_owner = new std::vector<std::int64_t>(std::move(offset_values));
    nb::capsule points_capsule(points_owner, [](void* p) noexcept { delete static_cast<std::vector<double>*>(p); });
    nb::capsule offsets_capsule(offsets_owner,
                                [](void* p) noexcept { delete static_cast<std::vector<std::int64_t>*>(p); });
    nb::ndarray<nb::numpy, double> points(points_owner->data(), { points_owner->size() / 2, 2 }, points_capsule);
    nb::ndarray<nb::numpy, std::int64_t> offsets(offsets_owner->data(), { offsets_owner->size() }, offsets_capsule);
    return nb::make_tuple(std::move(points), std::move(offsets));
}

} // namespace

NB_MODULE(_cucim_skimage_cpp_ext, m)
{
    m.doc() = "Optional compiled helpers for cucim.skimage.";
    m.def("assemble_contours", &assemble_contours, "segments"_a, "segment_keys"_a,
          "Assemble marching-squares line segments into contours.");
    m.def("assemble_contours_packed", &assemble_contours_packed, "segments"_a, "segment_keys"_a,
          "Assemble marching-squares line segments into a packed points/offsets representation.");
    m.def("assemble_linked_contours", &assemble_linked_contours, "segments"_a, "next"_a,
          "Assemble line segments whose successors are known into a packed points/offsets representation.");
    def_segmented_sort<std::uint8_t>(m, "segmented_sort_keys_temp_storage_bytes_uint8", "segmented_sort_keys_uint8",
                                     "segmented_sort_keys_ranges_uint8");
    def_segmented_sort<std::uint16_t>(m, "segmented_sort_keys_temp_storage_bytes_uint16", "segmented_sort_keys_uint16",
                                      "segmented_sort_keys_ranges_uint16");
    def_segmented_sort<float>(m, "segmented_sort_keys_temp_storage_bytes_float32", "segmented_sort_keys_float32",
                              "segmented_sort_keys_ranges_float32");
    def_segmented_sort<double>(m, "segmented_sort_keys_temp_storage_bytes_float64", "segmented_sort_keys_float64",
                               "segmented_sort_keys_ranges_float64");
}
