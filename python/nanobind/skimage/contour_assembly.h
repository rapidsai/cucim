/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstddef>
#include <limits>
#include <cstdint>
#include <stdexcept>
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

// Open-addressing hash table from non-negative 64-bit keys to contour indices.
//
// Removing a key only clears its value (the slot keeps its key), so the table
// never needs tombstones: a later insertion of the same key reuses the slot.
// ``max_keys`` is an upper bound on the number of distinct keys that are ever
// inserted; the load factor stays at or below one half.
class ContourIndexTable
{
public:
    explicit ContourIndexTable(std::size_t max_keys)
    {
        std::size_t capacity = 16;
        while (capacity < 2 * max_keys)
        {
            capacity *= 2;
        }
        mask_ = capacity - 1;
        keys_.assign(capacity, -1);
        values_.assign(capacity, absent);
    }

    // Return the contour index stored for ``key`` or ``invalid_index``.
    std::size_t find(long long key) const
    {
        std::size_t slot = hash(key);
        while (keys_[slot] != -1)
        {
            if (keys_[slot] == key)
            {
                return values_[slot] == absent ? invalid_index : values_[slot];
            }
            slot = (slot + 1) & mask_;
        }
        return invalid_index;
    }

    void set(long long key, std::size_t value)
    {
        std::size_t slot = hash(key);
        while (keys_[slot] != -1 && keys_[slot] != key)
        {
            slot = (slot + 1) & mask_;
        }
        keys_[slot] = key;
        values_[slot] = static_cast<std::uint32_t>(value);
    }

    void erase(long long key)
    {
        std::size_t slot = hash(key);
        while (keys_[slot] != -1)
        {
            if (keys_[slot] == key)
            {
                values_[slot] = absent;
                return;
            }
            slot = (slot + 1) & mask_;
        }
    }

private:
    static constexpr std::uint32_t absent = std::numeric_limits<std::uint32_t>::max();

    std::size_t hash(long long key) const
    {
        // Keys of neighboring grid edges are close, and so are the keys that
        // consecutive segments look up, so keeping the key order in the slot
        // order makes these lookups hit the same cache lines.
        return static_cast<std::size_t>(key) & mask_;
    }

    std::size_t mask_;
    std::vector<long long> keys_;
    std::vector<std::uint32_t> values_;
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
    if (n_segments >= std::numeric_limits<std::uint32_t>::max())
    {
        throw std::length_error("too many contour segments");
    }
    ContourIndexTable starts(n_segments);
    ContourIndexTable ends(n_segments);
    nodes.reserve(2 * n_segments);
    contours.reserve(n_segments);

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

        const std::size_t tail_index = starts.find(to_key);
        const bool has_tail = tail_index != invalid_index;
        if (has_tail)
        {
            starts.erase(to_key);
        }

        const std::size_t head_index = ends.find(from_key);
        const bool has_head = head_index != invalid_index;
        if (has_head)
        {
            ends.erase(from_key);
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
                starts.set(nodes[head.head].key, head_index);
                ends.set(nodes[head.tail].key, head_index);
            }
            else
            {
                starts.erase(nodes[head.head].key);
                nodes[head.tail].next = tail.head;
                tail.head = head.head;
                tail.size += head.size;
                head.active = false;
                starts.set(nodes[tail.head].key, tail_index);
                ends.set(nodes[tail.tail].key, tail_index);
            }
        }
        else if (!has_tail && !has_head)
        {
            const std::size_t from_node = add_node(from_point, from_key);
            const std::size_t to_node = add_node(to_point, to_key);
            nodes[from_node].next = to_node;
            contours.push_back(Contour{ from_node, to_node, 2, true });
            starts.set(from_key, current_index);
            ends.set(to_key, current_index);
            ++current_index;
        }
        else if (!has_head)
        {
            Contour& tail = contours[tail_index];
            prepend_node(tail, from_point, from_key);
            starts.set(from_key, tail_index);
        }
        else
        {
            Contour& head = contours[head_index];
            append_node(head, to_point, to_key);
            ends.set(to_key, head_index);
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

// Contours of segments that are already linked.
//
// ``next[i]`` is the segment that starts where segment ``i`` ends (-1 if there
// is none). Every segment has at most one successor and one predecessor, so
// the segments form open paths and closed loops. The contours (and the order
// of their points) are the ones that ``build_contour_assembly`` makes:
//
// - contours are ordered by their lowest segment index;
// - an open path starts at the ``from`` point of its first segment and ends at
//   the ``to`` point of its last one;
// - at the junction of two segments the point of the segment that comes first
//   in the input is used (the ``to`` point of ``a`` if ``a`` precedes ``b``,
//   else the ``from`` point of ``b``), because that is the node that already
//   existed when the other segment was attached;
// - a loop is closed by its highest numbered segment ``c``. It starts at the
//   ``from`` point of the successor of ``c`` and ends with the ``to`` point of
//   ``c``, so it has one more point than it has segments.
//
// ``segments`` points to ``n_segments`` (from_row, from_col, to_row, to_col)
// records. ``points`` receives the points of all contours and ``offsets`` the
// start of each contour in it (plus the total number of points at the end).
// Returns false if ``next`` is not a valid set of links.
inline bool trace_linked_contours(const double* segments,
                                  const std::int32_t* next,
                                  std::size_t n_segments,
                                  std::vector<double>& points,
                                  std::vector<std::int64_t>& offsets)
{
    std::vector<std::int32_t> prev(n_segments, -1);
    for (std::size_t i = 0; i < n_segments; ++i)
    {
        const std::int32_t j = next[i];
        if (j < 0)
        {
            continue;
        }
        if (static_cast<std::size_t>(j) >= n_segments || prev[j] >= 0)
        {
            return false;
        }
        prev[j] = static_cast<std::int32_t>(i);
    }

    // Find the first segment and the number of points of each contour. Segment
    // ``i`` is the lowest numbered segment of its contour when it is reached
    // here unseen, because all the lower numbered ones have been visited.
    std::vector<std::uint8_t> seen(n_segments, 0);
    std::vector<std::int32_t> first;
    std::vector<std::uint8_t> closed;
    offsets.assign(1, 0);
    for (std::size_t i = 0; i < n_segments; ++i)
    {
        if (seen[i])
        {
            continue;
        }
        std::size_t start = i;
        bool is_closed = false;
        while (prev[start] >= 0)
        {
            start = static_cast<std::size_t>(prev[start]);
            if (start == i)
            {
                is_closed = true;
                break;
            }
        }
        std::size_t count = 0;
        if (is_closed)
        {
            std::size_t last = i;
            std::size_t current = i;
            do
            {
                seen[current] = 1;
                ++count;
                last = current > last ? current : last;
                current = static_cast<std::size_t>(next[current]);
            } while (current != i);
            start = static_cast<std::size_t>(next[last]);
        }
        else
        {
            for (std::int64_t current = static_cast<std::int64_t>(start); current >= 0; current = next[current])
            {
                seen[current] = 1;
                ++count;
            }
        }
        first.push_back(static_cast<std::int32_t>(start));
        closed.push_back(is_closed);
        offsets.push_back(offsets.back() + static_cast<std::int64_t>(count + 1));
    }

    points.resize(2 * static_cast<std::size_t>(offsets.back()));
    double* out = points.data();
    for (std::size_t k = 0; k < first.size(); ++k)
    {
        std::size_t current = static_cast<std::size_t>(first[k]);
        const std::size_t start = current;
        out[0] = segments[4 * current];
        out[1] = segments[4 * current + 1];
        out += 2;
        while (true)
        {
            const std::int32_t following = next[current];
            if (following < 0 || (closed[k] && static_cast<std::size_t>(following) == start))
            {
                break;
            }
            const std::size_t other = static_cast<std::size_t>(following);
            const double* point = current < other ? segments + 4 * current + 2 : segments + 4 * other;
            out[0] = point[0];
            out[1] = point[1];
            out += 2;
            current = other;
        }
        out[0] = segments[4 * current + 2];
        out[1] = segments[4 * current + 3];
        out += 2;
    }
    return true;
}

} // namespace cucim::skimage
