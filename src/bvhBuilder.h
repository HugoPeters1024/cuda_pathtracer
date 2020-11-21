#ifndef H_BVH_BUILDER
#define H_BVH_BUILDER

#include <utility>
#include <tuple>
#include "types.h"
#include "constants.h"
#include "vec.h"

Box buildTriangleBox(const Triangle* triangles, uint count)
{
    float min_x, min_y, min_z, max_x, max_y, max_z;
    min_x = min_y = min_z = 100000;
    max_x = max_y = max_z = -100000;
    for(int i=0; i<count; i++)
    {
        const Triangle& t = triangles[i];
        min_x = std::min(min_x, t.min_x());
        min_y = std::min(min_y, t.min_y());
        min_z = std::min(min_z, t.min_z());

        max_x = std::max(max_x, t.max_x());
        max_y = std::max(max_y, t.max_y());
        max_z = std::max(max_z, t.max_z());
    }

    return Box {
        make_float3(min_x, min_y, min_z),
        make_float3(max_x, max_y, max_z)
    };
}

BVHNode* createBVH(std::vector<Triangle>& triangles, uint* bvh_size)
{
    // bvh size is bounded by 2*triangle_count-1
    BVHNode* ret = (BVHNode*)malloc((2 * triangles.size() - 1) * sizeof(BVHNode));
    std::stack<std::tuple<uint, uint, uint>> work;
    uint node_count = 0;

    work.push(std::make_tuple(node_count++, 0, triangles.size()));

    while(!work.empty())
    {
        auto workItem = work.top();
        work.pop();
        uint index = std::get<0>(workItem);
        uint start = std::get<1>(workItem);
        uint count = std::get<2>(workItem);

        Box boundingBox = buildTriangleBox(&triangles[start], count);
        float parentSurface = boundingBox.getSurfaceArea();
        int min_level = -1;
        int min_split_pos = -1;
        float min_cost = count;
        for(int level = 0; level<3; level++)
        {
            // Sort the triangles on the dimension we want to check
            switch (level) {
                case 0: std::sort(triangles.begin()+start, triangles.begin()+start+count, __compare_triangles_x); break;
                case 1: std::sort(triangles.begin()+start, triangles.begin()+start+count, __compare_triangles_y); break;
                case 2: std::sort(triangles.begin()+start, triangles.begin()+start+count, __compare_triangles_z); break;
            }

            float leftCosts[count];
            Box leftBox = triangles[start].getBoundingBox();

            float rightCosts[count];
            Box rightBox = triangles[start+count-1].getBoundingBox();

            // first sweep from the left to get the left costs
            // we sweep so we can incrementally update the bounding box for efficiency
            for (int i=0; i<count; i++)
            {
                const Triangle& tl = triangles[start+i];
                leftBox.consumePoint(tl.v0);
                leftBox.consumePoint(tl.v1);
                leftBox.consumePoint(tl.v2);
                leftCosts[i] = leftBox.getSurfaceArea() / parentSurface;

                const Triangle& tr = triangles[start + count - i - 1];
                rightBox.consumePoint(tr.v0);
                rightBox.consumePoint(tr.v1);
                rightBox.consumePoint(tr.v2);
                rightCosts[i] = rightBox.getSurfaceArea() / parentSurface;
            }

            // Find the optimal combined costs index
            for(int i=0; i<count; i++)
            {
                // 0.5 is the cost of traversal
                float thisCost = leftCosts[i] * i + rightCosts[count - i - 1] * (count - i) + 2.5;
                if (thisCost < min_cost)
                {
                    min_cost = thisCost;
                    min_split_pos = i;
                    min_level = level;
                }
            }
        }

        // Splitting was not worth it become a child and push no new work.
        if (min_level == -1)
        {
            ret[index] = BVHNode::MakeChild(boundingBox, start, count);
            continue;
        }

        // splitting should never be better than terminating with 1
        // triangle.
        assert (count >= 2);

        // Sort the triangles one last time based on the level the heuristic gave us
        // to ensure that they are in the expected order
        switch (min_level) {
            case 0: std::sort(triangles.begin()+start, triangles.begin()+start+count, __compare_triangles_x); break;
            case 1: std::sort(triangles.begin()+start, triangles.begin()+start+count, __compare_triangles_y); break;
            case 2: std::sort(triangles.begin()+start, triangles.begin()+start+count, __compare_triangles_z); break;
        }

        // Create items for the children, ensures children are next to each other
        // in memory
        uint child1_index = node_count++;
        uint child2_index = node_count++;

        uint child1_start = start;
        uint child1_count = min_split_pos;

        uint child2_start = start + min_split_pos;
        uint child2_count = count - min_split_pos;

        assert (child2_start == child1_start+child1_count);
        assert (child1_count + child2_count == count);

        // push the work on the stack
        work.push(std::make_tuple(child2_index, child2_start, child2_count));
        work.push(std::make_tuple(child1_index, child1_start, child1_count));

        // Create a node
        ret[index] = BVHNode::MakeNode(boundingBox, child1_index, min_level);
    }

    // Reduce the memory allocation to match the final size
    ret = (BVHNode*)realloc(ret, node_count * sizeof(BVHNode));
    *bvh_size = node_count;
    return ret;
}

#endif
