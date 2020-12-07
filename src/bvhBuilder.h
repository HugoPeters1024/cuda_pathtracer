#ifndef H_BVH_BUILDER
#define H_BVH_BUILDER

#include <utility>
#include <tuple>
#include "types.h"
#include "constants.h"
#include "vec.h"

inline BVHNode* createBVH(std::vector<TriangleV>& trianglesV, std::vector<TriangleD>& trianglesD, uint* bvh_size)
{
    // bvh size is bounded by 2*triangle_count-1
    BVHNode* ret = (BVHNode*)malloc((2 * trianglesV.size() - 1) * sizeof(BVHNode));
    uint* indices = (uint*)malloc(trianglesV.size() * sizeof(uint));
    for(uint i=0; i<trianglesV.size(); i++) indices[i] = i;
    float* leftCosts = (float*)malloc(trianglesV.size() * sizeof(float));
    float* rightCosts = (float*)malloc(trianglesV.size() * sizeof(float));
    std::stack<std::tuple<uint, uint, uint>> work;
    uint node_count = 0;

    SORTING_SOURCE = trianglesV.data();

    work.push(std::make_tuple(node_count++, 0, trianglesV.size()));

    while(!work.empty())
    {
        auto workItem = work.top();
        work.pop();
        uint index = std::get<0>(workItem);
        uint start = std::get<1>(workItem);
        uint count = std::get<2>(workItem);

        int min_level = -1;
        int min_split_pos = -1;
        float min_cost = count;
        Box boundingBox;
        float invParentSurface;
        for(int level = 0; level<3; level++)
        {
            // Sort the triangles on the dimension we want to check
            switch (level) {
                case 0: std::sort(indices+start, indices+start+count, __compare_triangles_x); break;
                case 1: std::sort(indices+start, indices+start+count, __compare_triangles_y); break;
                case 2: std::sort(indices+start, indices+start+count, __compare_triangles_z); break;
            }

            Box leftBox = Box::fromPoint(trianglesV[indices[start]].v0);
            Box rightBox = Box::fromPoint(trianglesV[indices[start+count-1]].v0);

            // first sweep from the left to get the left costs
            // we sweep so we can incrementally update the bounding box for efficiency
            for (int i=0; i<count; i++)
            {
                // left is exclusive
                const TriangleV& tl = trianglesV[indices[start+i]];
                leftCosts[i] = leftBox.getSurfaceArea();
                leftBox.consumePoint(tl.v0);
                leftBox.consumePoint(tl.v1);
                leftBox.consumePoint(tl.v2);

                // right is inclusive
                const TriangleV& tr = trianglesV[indices[start + count - i - 1]];
                rightBox.consumePoint(tr.v0);
                rightBox.consumePoint(tr.v1);
                rightBox.consumePoint(tr.v2);
                rightCosts[i] = rightBox.getSurfaceArea();
            }

            // left box and right box now have the full set of triangles, so we know the parent surface
            boundingBox = rightBox;
            invParentSurface = 1.0f / boundingBox.getSurfaceArea();

            // Find the optimal combined costs index
            for(int i=0; i<count; i++)
            {
                // 0.5 is the cost of traversal
                float thisCost = leftCosts[i] * i * invParentSurface + rightCosts[count - i - 1] * (count - i) * invParentSurface + EPS;
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
            case 0: std::sort(indices+start, indices+start+count, __compare_triangles_x); break;
            case 1: std::sort(indices+start, indices+start+count, __compare_triangles_y); break;
                // case 2 is already satisfied
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
        assert (child1_count > 0);
        assert (child2_count > 0);

        // push the work on the stack
        work.push(std::make_tuple(child2_index, child2_start, child2_count));
        work.push(std::make_tuple(child1_index, child1_start, child1_count));

        // Create a node
        ret[index] = BVHNode::MakeNode(boundingBox, child1_index, min_level);
    }

    // permute the triangles given the indices.
    auto tmp_trianglesV = std::vector<TriangleV>(trianglesV.begin(), trianglesV.end());
    auto tmp_trianglesD = std::vector<TriangleD>(trianglesD.begin(), trianglesD.end());
    for(int i=0; i<trianglesV.size(); i++)
    {
        trianglesV[i] = tmp_trianglesV[indices[i]];
        trianglesD[i] = tmp_trianglesD[indices[i]];
    }

    free(leftCosts);
    free(rightCosts);
    free(indices);

    // Reduce the memory allocation to match the final size
    ret = (BVHNode*)realloc(ret, node_count * sizeof(BVHNode));
    *bvh_size = node_count;
    return ret;
}

#endif
