/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <algorithm>
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

// Where the points of linked contours go, from the sublists of a list ranking.
//
// Segments are linked when ``next[i]`` is the segment that starts where
// segment ``i`` ends (-1 if there is none). Every segment has at most one
// successor and one predecessor, so the segments form open paths and closed
// loops. On the device they are cut into sublists (paths of segments that end
// where another sublist starts, or at the end of a contour). This function
// orders the sublists into the contours that ``build_contour_assembly`` makes
// from the segments, from a few numbers per sublist:
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
// The sublists are described by:
//
//     first[s]      first segment of sublist s
//     length[s]     number of segments in it
//     next_sub[s]   sublist that follows it, or -1
//     min_id[s]     lowest segment index in it
//     max_id[s]     highest segment index in it
//     max_pos[s]    position of that segment in the sublist
//
// The output point of the segment that is at position ``j`` of sublist ``s``
// (the junction with the following segment, or the last point of the contour)
// is stored at the index
//
//     j > cut[s] ? start[s] + (j - cut[s] - 1) : wrap[s] + j
//
// of the points array. ``cut[s]`` is -1 except in the sublist that holds the
// highest numbered segment of a closed contour, where the contour is cut. The
// first point of contour ``k`` is the ``from`` point of segment ``anchor[k]``
// (an open contour) or of its successor (a closed one, ``closed[k]``, whose
// anchor is the segment that closes it).
struct ContourPlan
{
    std::vector<std::int64_t> offsets;
    std::vector<std::int32_t> anchor;
    std::vector<std::uint8_t> closed;
    std::vector<std::int64_t> start;
    std::vector<std::int32_t> cut;
    std::vector<std::int64_t> wrap;
};

// Returns false if the sublists are not linked as paths and cycles.
inline bool plan_linked_contours(const std::int32_t* first,
                                 const std::int32_t* length,
                                 const std::int32_t* next_sub,
                                 const std::int32_t* min_id,
                                 const std::int32_t* max_id,
                                 const std::int32_t* max_pos,
                                 std::size_t m,
                                 ContourPlan& plan)
{
    std::vector<std::int32_t> pred(m, -1);
    for (std::size_t s = 0; s < m; ++s)
    {
        const std::int32_t t = next_sub[s];
        if (t < 0)
        {
            continue;
        }
        if (static_cast<std::size_t>(t) >= m || pred[t] >= 0)
        {
            return false;
        }
        pred[t] = static_cast<std::int32_t>(s);
    }

    struct Component
    {
        std::int32_t min_id;
        std::int32_t head;      // first sublist of an open contour, any sublist of a closed one
        std::int32_t cut;       // sublist with the highest segment of a closed contour
        std::int64_t count;     // number of segments
    };
    std::vector<Component> components;
    std::vector<std::uint8_t> seen(m, 0);
    for (std::size_t s = 0; s < m; ++s)
    {
        if (seen[s])
        {
            continue;
        }
        std::size_t head = s;
        bool closed = false;
        while (pred[head] >= 0)
        {
            head = static_cast<std::size_t>(pred[head]);
            if (head == s)
            {
                closed = true;
                break;
            }
        }
        Component component{ min_id[head], static_cast<std::int32_t>(head), -1, 0 };
        std::size_t cut = head;
        std::size_t current = head;
        do
        {
            seen[current] = 1;
            component.count += length[current];
            component.min_id = std::min(component.min_id, min_id[current]);
            if (max_id[current] > max_id[cut])
            {
                cut = current;
            }
            const std::int32_t following = next_sub[current];
            if (following < 0)
            {
                break;
            }
            current = static_cast<std::size_t>(following);
        } while (current != head);
        if (closed)
        {
            component.cut = static_cast<std::int32_t>(cut);
        }
        components.push_back(component);
    }
    std::sort(components.begin(), components.end(),
              [](const Component& a, const Component& b) { return a.min_id < b.min_id; });

    plan.offsets.assign(1, 0);
    plan.anchor.clear();
    plan.closed.clear();
    plan.start.assign(m, 0);
    plan.cut.assign(m, -1);
    plan.wrap.assign(m, 0);
    for (const Component& component : components)
    {
        const std::int64_t base = plan.offsets.back();
        plan.offsets.push_back(base + component.count + 1);
        std::int64_t position = base + 1;
        if (component.cut < 0)
        {
            for (std::int32_t s = component.head; s >= 0; s = next_sub[s])
            {
                plan.start[s] = position;
                position += length[s];
            }
            plan.anchor.push_back(first[component.head]);
            plan.closed.push_back(0);
        }
        else
        {
            const std::int32_t cut = component.cut;
            // The segments after the highest one come first, then the other
            // sublists around the loop, then the segments up to the highest one.
            plan.cut[cut] = max_pos[cut];
            plan.start[cut] = position;
            position += length[cut] - 1 - max_pos[cut];
            for (std::int32_t s = next_sub[cut]; s != cut; s = next_sub[s])
            {
                plan.start[s] = position;
                position += length[s];
            }
            plan.wrap[cut] = position;
            plan.anchor.push_back(max_id[cut]);
            plan.closed.push_back(1);
        }
    }
    return true;
}

} // namespace cucim::skimage
