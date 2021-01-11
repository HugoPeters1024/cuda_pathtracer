#ifndef H_BVH_BUILDER
#define H_BVH_BUILDER

#include <utility>
#include <tuple>
#include "types.h"
#include "constants.h"
#include "vec.h"

struct WorkItem
{
    uint index;
    uint start;
    uint count;
};

inline void permuteTriangles(uint* indices, TriangleV* trianglesV, TriangleD* trianglesD, uint n)
{
    // permute the triangles given the indices. 
    auto tmp_trianglesV = std::vector<TriangleV>(trianglesV, trianglesV + n);
    auto tmp_trianglesD = std::vector<TriangleD>(trianglesD, trianglesD + n);
    for(int i=0; i<n; i++)
    {
        trianglesV[i] = tmp_trianglesV[indices[i]];
        trianglesD[i] = tmp_trianglesD[indices[i]];
    }
}

inline uint getBinId(uint K, uint axis, const float3& centroid, float bmin, float bmax)
{
    return (uint)(K*(1-EPS)*(at(centroid, axis) - bmin) / (bmax - bmin));
}

// http://www.sci.utah.edu/~wald/Publications/2007/ParallelBVHBuild/fastbuild.pdf
inline BVHNode* createBVHBinned(TriangleV* trianglesV, TriangleD* trianglesD, uint nrTriangles, uint triangleOffset, uint* bvh_size)
{
    float ping = glfwGetTime();
    // find dominant axis
    // uniformly select K spatial intervals
    // use nth_interval to partition the triangles
    const uint K = 16;
    
    BVHNode* ret = (BVHNode*)malloc((2 * nrTriangles - 1) * sizeof(BVHNode));
    uint* indices = (uint*)malloc(nrTriangles * sizeof(uint));
    SSEBox* boundingBoxes = (SSEBox*)malloc(nrTriangles * sizeof(SSEBox));
    char* binIds = (char*)malloc(nrTriangles * sizeof(char));
    __m128* centroids = (__m128*)malloc(nrTriangles * sizeof(__m128));

    SSEBox rootBox = SSEBox::insideOut();
    for(uint i=0; i<nrTriangles; i++) 
    {
        indices[i] = i;
        //centroids[i] = 0.333333 * (trianglesV[i].v0 + trianglesV[i].v1 + trianglesV[i].v2);
        __m128 ssev0 = _mm_setr_ps(trianglesV[i].v0.x, trianglesV[i].v0.y, trianglesV[i].v0.z, 0);
        __m128 ssev1 = _mm_setr_ps(trianglesV[i].v1.x, trianglesV[i].v1.y, trianglesV[i].v1.z, 0);
        __m128 ssev2 = _mm_setr_ps(trianglesV[i].v2.x, trianglesV[i].v2.y, trianglesV[i].v2.z, 0);
        centroids[i] = _mm_mul_ps(_mm_setr_ps(0.333333, 0.3333333, 0.3333333, 0), _mm_add_ps(ssev0, _mm_add_ps(ssev1, ssev2)));
        boundingBoxes[i] = SSEBox::fromTriangle(trianglesV[i]);
        rootBox.consumeBox(boundingBoxes[i]);
    }

    // set the root box
    Box rootBoxNormal = rootBox.toNormalBox();
    ret[0].vmin = make_float4(rootBoxNormal.vmin, 0);
    ret[0].vmax = make_float4(rootBoxNormal.vmax, 0);

    //std::stack<std::tuple<uint, uint, uint>> work;
    uint node_count = 0;
    WorkItem stack[256];
    uint stack_size = 0;
    stack[stack_size++] = WorkItem { node_count++, 0, nrTriangles };
    //work.push(std::make_tuple(node_count++, 0, nrTriangles));

    float leftCosts[K];
    float rightCosts[K];
    uint binCounts[K];
    SSEBox bins[K];

    SSEBox scannedBinsLeft[K];
    SSEBox scannedBinsRight[K];

    while(stack_size > 0)
    {
        assert(stack_size < 128);
        auto workItem = stack[--stack_size];
        uint index = workItem.index;
        uint start = workItem.start;
        uint count = workItem.count;
        // bounding boxes are assigned forwardly
        Box parent = ret[index].getBox();

        // Count lower bound
        if (count <= 4)
        {
            ret[index] = BVHNode::MakeChild(parent, start + triangleOffset, count);
            continue;
        }

        float invParentSurface = 1.0f / parent.getSurfaceArea();

        // use centroids to find the dominant axis.
        //SSEBox parent = SSEBox::insideOut();
        SSEBox parentCentroid = SSEBox::insideOut();
        for(int i=start; i<start+count; i++)
        {
            const __m128& centroid = centroids[indices[i]];
            //const SSEBox& box = boundingBoxes[indices[i]];
           // parent.consumeBox(box);
            parentCentroid.consumePoint(centroid);
        }

        __m128 diff = _mm_sub_ps(parentCentroid.vmax, parentCentroid.vmin);

        int axis;
        if (diff[0] > diff[1] && diff[0] > diff[2])
            axis = 0;
        else if (diff[1] > diff[0] && diff[1] > diff[2])
            axis = 1;
        else 
            axis = 2;

        float bmin = parentCentroid.vmin[axis];
        float bmax = parentCentroid.vmax[axis];

        // dominant axis bound
        if (bmax - bmin < K * EPS)
        {
            ret[index] = BVHNode::MakeChild(parent, start + triangleOffset, count);
            continue;
        }


        for(uint k=0; k<K; k++)
        {
            binCounts[k] = 0;
            bins[k] = SSEBox::insideOut();
        }

        // Populate the bins
        float binFac = K*(1-EPS) / (bmax - bmin);
        for(int i=start; i<start+count; i++)
        {
            const __m128& centroid = centroids[indices[i]];
            const SSEBox& b = boundingBoxes[indices[i]];
            char binId = (char)((centroid[axis] - bmin) * binFac);
            binIds[indices[i]] = binId;

            binCounts[binId]++;
            bins[binId].consumeBox(b);
        }

        float min_sah = count;
        int min_k = -1;

        // Do a sweep to collect the SAH values
        SSEBox leftBox = SSEBox::insideOut();
        SSEBox rightBox = SSEBox::insideOut();
        uint leftCount = 0;
        uint rightCount = 0;


        // first sweep from the left to get the left costs
        // we sweep so we can incrementally update the bounding box for efficiency
        for (int k=0; k<K; k++)
        {
            // left is exclusive
            leftCosts[k] = leftCount * leftBox.getSurfaceArea() * invParentSurface;
            scannedBinsLeft[k] = leftBox;
            leftBox.consumeBox(bins[k]);
            leftCount += binCounts[k];

            // right is inclusive
            rightBox.consumeBox(bins[K-k-1]);
            rightCount += binCounts[K - k - 1];
            rightCosts[K - k - 1] = rightCount * rightBox.getSurfaceArea() * invParentSurface;
            scannedBinsRight[K - k - 1] = rightBox;
        }

        // calculate the sah
        for(int k=0; k<K; k++)
        {
            float sah = leftCosts[k] + rightCosts[k] + EPS;
            if (sah < min_sah)
            {
                min_sah = sah;
                min_k = k;
            }
        }

        // Splitting was not worth it become a child and push no new work.
        if (min_k == -1)
        {
            ret[index] = BVHNode::MakeChild(parent, start + triangleOffset, count);
            continue;
        }

        uint* pLeft = indices+start;
        uint* pRight = indices+start+count-1;
        char state = 0;
        while(pLeft != pRight)
        {
            if (state == 0) {
                if(binIds[*pLeft] < min_k) {
                    pLeft++;
                    continue;
                } else state = 1;
            }

            if (state == 1) {
                if(binIds[*pRight] >= min_k) {
                    pRight--;
                    continue;
                }
            }

            uint tmp = *pLeft;
            *pLeft = *pRight;
            *pRight = tmp;
            state = 0;
        }

        uint minLeftCount = pLeft - (indices+start);

        // Create items for the children, ensures children are next to each other
        // in memory
        uint child1_index = node_count++;
        uint child2_index = node_count++;

        uint child1_start = start;
        uint child1_count = minLeftCount;

        uint child2_start = start + child1_count;
        uint child2_count = count - child1_count;

        /*
        assert (child2_start == child1_start+child1_count);
        assert (child1_count + child2_count == count);
        assert (child1_count > 0);
        assert (child2_count > 0);
         */

        // forward assign the bounding boxes
        SSEBox child1Box = scannedBinsLeft[min_k];
        Box child1BoxNormal = child1Box.toNormalBox();
        ret[child1_index].vmin = make_float4(child1BoxNormal.vmin, 0);
        ret[child1_index].vmax = make_float4(child1BoxNormal.vmax, 0);

        SSEBox child2Box = scannedBinsRight[min_k];
        Box child2BoxNormal = child2Box.toNormalBox();
        ret[child2_index].vmin = make_float4(child2BoxNormal.vmin, 0);
        ret[child2_index].vmax = make_float4(child2BoxNormal.vmax, 0);

        // push the work on the stack
        //work.push(std::make_tuple(child2_index, child2_start, child2_count));
        //work.push(std::make_tuple(child1_index, child1_start, child1_count));
        stack[stack_size++] = WorkItem { child2_index, child2_start, child2_count };
        stack[stack_size++] = WorkItem { child1_index, child1_start, child1_count };

        // Create a node
        ret[index] = BVHNode::MakeNode(parent, child1_index);
    }


    printf("BVH build took %f ms\n", (glfwGetTime() - ping)*1000);
    permuteTriangles(indices, trianglesV, trianglesD, nrTriangles);

    free(indices);
    free(binIds);
    free(centroids);
    free(boundingBoxes);
    ret = (BVHNode*)realloc(ret, node_count * sizeof(BVHNode));
    *bvh_size = node_count;
    return ret;
}

inline BVHNode* createBVH(TriangleV* trianglesV, TriangleD* trianglesD, uint nrTriangles, uint triangleOffset, uint* bvh_size)
{
    float ping = glfwGetTime();
    // bvh size is bounded by 2*triangle_count-1
    BVHNode* ret = (BVHNode*)malloc((2 * nrTriangles - 1) * sizeof(BVHNode));
    uint* indices = (uint*)malloc(nrTriangles * sizeof(uint));
    float3* centroids = (float3*)malloc(nrTriangles * sizeof(float3));
    for(uint i=0; i<nrTriangles; i++)
    {
        indices[i] = i;
        centroids[i] = 0.333333 * (trianglesV[i].v0 + trianglesV[i].v1 + trianglesV[i].v2);
    }

    float* leftCosts = (float*)malloc(nrTriangles * sizeof(float));
    float* rightCosts = (float*)malloc(nrTriangles * sizeof(float));
    std::stack<std::tuple<uint, uint, uint>> work;
    uint node_count = 0;

    SORTING_SOURCE = centroids;

    work.push(std::make_tuple(node_count++, 0, nrTriangles));

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
        if (min_level == -1 || count <= 4)
        {
            ret[index] = BVHNode::MakeChild(boundingBox, start + triangleOffset, count);
            continue;
        }

        // splitting should never be better than terminating with 1
        // triangle.
        assert (count >= 1);

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
        ret[index] = BVHNode::MakeNode(boundingBox, child1_index);
    }

    printf("BVH build took %f ms\n", (glfwGetTime() - ping)*1000);
    permuteTriangles(indices, trianglesV, trianglesD, nrTriangles);
    free(leftCosts);
    free(rightCosts);
    free(centroids);
    free(indices);

    // Reduce the memory allocation to match the final size
    ret = (BVHNode*)realloc(ret, node_count * sizeof(BVHNode));
    *bvh_size = node_count;

    return ret;
}

inline BVHNode* makeQBVH(BVHNode* bvh, uint bvh_size)
{
    BVHNode* ret = (BVHNode*)malloc(bvh_size * sizeof(BVHNode));
    return ret;
}

#endif
