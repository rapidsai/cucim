/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstddef>
#include <limits>
#include <unordered_map>
#include <utility>
#include <vector>

namespace cucim::skimage
{

struct Point
{
    double row;
    double col;

    bool operator==(const Point& other) const
    {
        return row == other.row && col == other.col;
    }
};

constexpr std::size_t invalid_index = std::numeric_limits<std::size_t>::max();

struct Node
{
    Point point;
    long long key = 0;
    std::size_t next = invalid_index;
};

struct Contour
{
    std::size_t head = invalid_index;
    std::size_t tail = invalid_index;
    std::size_t size = 0;
    bool active = true;
};

struct ContourAssembly
{
    std::vector<Node> nodes;
    std::vector<Contour> contours;
};

// Link marching-squares output segments into contours.
//
// ``segments`` points to ``n_segments`` consecutive (from_row, from_col,
// to_row, to_col) records. ``segment_keys`` points to ``n_segments`` pairs of
// topological IDs (of the grid edges containing the from and to points). The
// segments are linked by these integer keys, which preserves the traversal
// order of scikit-image's Python implementation.
inline ContourAssembly build_contour_assembly(const double* data, const long long* key_data, std::size_t n_segments)
{
    ContourAssembly assembly;
    std::vector<Node>& nodes = assembly.nodes;
    std::vector<Contour>& contours = assembly.contours;
    std::unordered_map<long long, std::size_t> starts;
    std::unordered_map<long long, std::size_t> ends;
    nodes.reserve(2 * n_segments);
    contours.reserve(n_segments);
    starts.max_load_factor(0.7F);
    ends.max_load_factor(0.7F);
    starts.reserve(n_segments);
    ends.reserve(n_segments);

    auto add_node = [&nodes](const Point& point, long long key) {
        nodes.push_back(Node{ point, key });
        return nodes.size() - 1;
    };

    auto append_node = [&nodes, &add_node](Contour& contour, const Point& point, long long key) {
        const std::size_t node_index = add_node(point, key);
        nodes[contour.tail].next = node_index;
        contour.tail = node_index;
        ++contour.size;
    };

    auto prepend_node = [&nodes, &add_node](Contour& contour, const Point& point, long long key) {
        const std::size_t node_index = add_node(point, key);
        nodes[node_index].next = contour.head;
        contour.head = node_index;
        ++contour.size;
    };

    std::size_t current_index = 0;
    for (std::size_t i = 0; i < n_segments; ++i)
    {
        const double* segment = data + 4 * i;
        Point from_point{ segment[0], segment[1] };
        Point to_point{ segment[2], segment[3] };
        const long long from_key = key_data[2 * i];
        const long long to_key = key_data[2 * i + 1];

        // Ignore degenerate segments. This matches scikit-image's behavior for
        // vertices exactly on the requested level.
        if (from_point == to_point)
        {
            continue;
        }

        bool has_tail = false;
        bool has_head = false;
        std::size_t tail_index = 0;
        std::size_t head_index = 0;

        auto tail_iter = starts.find(to_key);
        if (tail_iter != starts.end())
        {
            has_tail = true;
            tail_index = tail_iter->second;
            starts.erase(tail_iter);
        }

        auto head_iter = ends.find(from_key);
        if (head_iter != ends.end())
        {
            has_head = true;
            head_index = head_iter->second;
            ends.erase(head_iter);
        }

        if (has_tail && has_head)
        {
            Contour& tail = contours[tail_index];
            Contour& head = contours[head_index];

            if (tail_index == head_index)
            {
                append_node(head, to_point, to_key);
            }
            else if (tail_index > head_index)
            {
                nodes[head.tail].next = tail.head;
                head.tail = tail.tail;
                head.size += tail.size;
                tail.active = false;
                starts[nodes[head.head].key] = head_index;
                ends[nodes[head.tail].key] = head_index;
            }
            else
            {
                starts.erase(nodes[head.head].key);
                nodes[head.tail].next = tail.head;
                tail.head = head.head;
                tail.size += head.size;
                head.active = false;
                starts[nodes[tail.head].key] = tail_index;
                ends[nodes[tail.tail].key] = tail_index;
            }
        }
        else if (!has_tail && !has_head)
        {
            const std::size_t from_node = add_node(from_point, from_key);
            const std::size_t to_node = add_node(to_point, to_key);
            nodes[from_node].next = to_node;
            contours.push_back(Contour{ from_node, to_node, 2, true });
            starts[from_key] = current_index;
            ends[to_key] = current_index;
            ++current_index;
        }
        else if (!has_head)
        {
            Contour& tail = contours[tail_index];
            prepend_node(tail, from_point, from_key);
            starts[from_key] = tail_index;
        }
        else
        {
            Contour& head = contours[head_index];
            append_node(head, to_point, to_key);
            ends[to_key] = head_index;
        }
    }

    return assembly;
}

// Return the number of contours that were not merged into another one and the
// total number of points in them.
inline std::pair<std::size_t, std::size_t> count_active_contours(const ContourAssembly& assembly)
{
    std::size_t active_contours = 0;
    std::size_t total_points = 0;
    for (const Contour& contour : assembly.contours)
    {
        if (contour.active)
        {
            ++active_contours;
            total_points += contour.size;
        }
    }
    return { active_contours, total_points };
}

} // namespace cucim::skimage
