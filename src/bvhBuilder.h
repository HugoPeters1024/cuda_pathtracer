#ifndef H_BVH_BUILDER
#define H_BVH_BUILDER

#include "types.h"
#include "constants.h"
#include "vec.h"

Box buildTriangleBox(const std::vector<Triangle> triangles)
{
    float min_x, min_y, min_z, max_x, max_y, max_z;
    min_x = min_y = min_z = 100000;
    max_x = max_y = max_z = -100000;
    for(const Triangle t: triangles)
    {
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

BVHTree* createBVH(std::vector<Triangle> triangles)
{
    BVHTree* ret = new BVHTree();
    ret->isLeaf = false;
    ret->child1 = nullptr;
    ret->child2 = nullptr;
    ret->triangles = std::vector<Triangle>();
    ret->boundingBox = buildTriangleBox(triangles);
    const float parentSurface = ret->boundingBox.getSurfaceArea();
    ret->used_level = -1;

    int min_level = -1;
    int min_split_pos = -1;
    float min_cost = triangles.size();

    for(int level = 0; level<3; level++)
    {
        // Sort the triangles on the dimension we want to check
        switch (level) {
            case 0: std::sort(triangles.begin(), triangles.end(), __compare_triangles_x); break;
            case 1: std::sort(triangles.begin(), triangles.end(), __compare_triangles_y); break;
            case 2: std::sort(triangles.begin(), triangles.end(), __compare_triangles_z); break;
        }

        std::vector<float> leftCosts;
        leftCosts.reserve(triangles.size());
        Box leftBox = triangles[0].getBoundingBox();

        std::vector<float> rightCosts;
        rightCosts.reserve(triangles.size());
        Box rightBox = triangles.back().getBoundingBox();

        // first sweep from the left to get the left costs
        // we sweep so we can incrementally update the bounding box for efficiency
        for (int i=0; i<triangles.size(); i++)
        {
            const Triangle& tl = triangles[i];
            leftBox.consumePoint(tl.v0);
            leftBox.consumePoint(tl.v1);
            leftBox.consumePoint(tl.v2);
            leftCosts.push_back((leftBox.getSurfaceArea() / parentSurface));

            const Triangle& tr = triangles[triangles.size() - i-1];
            rightBox.consumePoint(tr.v0);
            rightBox.consumePoint(tr.v1);
            rightBox.consumePoint(tr.v2);
            rightCosts.push_back((rightBox.getSurfaceArea() / parentSurface));
        }

        // Find the optimal combined costs index
        for(int i=0; i<triangles.size(); i++)
        {
            // 0.5 is the cost of traversal
            float thisCost = leftCosts[i] * i + rightCosts[triangles.size() - i - 1] * (triangles.size() - i) + 1.0;
            if (thisCost < min_cost)
            {
                min_cost = thisCost;
                min_split_pos = i;
                min_level = level;
            }
        }
    }

    // Splitting was not worth it
    if (min_level == -1)
    {
        ret->isLeaf = true;
        ret->triangles = triangles;
        return ret;
    }

    assert(triangles.size() >= 2);

    // Sort the triangles one last time based on the level the heuristic gave us.
    switch (min_level) {
        case 0: std::sort(triangles.begin(), triangles.end(), __compare_triangles_x); break;
        case 1: std::sort(triangles.begin(), triangles.end(), __compare_triangles_y); break;
        case 2: std::sort(triangles.begin(), triangles.end(), __compare_triangles_z); break;
    }

    int m = min_split_pos;
    std::vector<Triangle> lows(triangles.begin(), triangles.begin() + m);
    std::vector<Triangle> highs(triangles.begin() + m, triangles.end());

    assert (lows.size() + highs.size() == triangles.size());
    assert (lows.size() > 0 && highs.size() > 0);

    // child1 must be the near child
    ret->child1 = createBVH(lows);
    ret->child2 = createBVH(highs);
    ret->used_level = min_level;
    return ret;
}

void sequentializeBvh(const BVHTree* root, std::vector<Triangle>& newTriangles, std::vector<BVHNode>& seqBvh)
{
    // Keep track of a parent id and subtree.
    std::stack<std::pair<uint, const BVHTree*>> work;
    work.push(std::pair<uint, const BVHTree*>(0, root));

    while(!work.empty())
    {
        std::pair<uint, const BVHTree*> tmp = work.top();
        work.pop();

        const BVHTree* currentNode = tmp.second;
        uint discoveredBy = tmp.first;

        Box boundingBox = currentNode->boundingBox;
        uint split_plane = currentNode->used_level;

        // The node adds it's segment
        uint t_start = newTriangles.size();
        uint t_count = currentNode->triangles.size();

        // Add the triangles
        for (const Triangle& t : currentNode->triangles) newTriangles.push_back(t);


        // Calculate the indices of the children before hand
        uint myId = seqBvh.size();
        if (t_count == 0) // aka not a leaf
        {
            uint child2 = myId + 1 + currentNode->child1->treeSize();
            seqBvh.push_back(BVHNode::MakeNode(boundingBox, child2, split_plane));

            // child 1 should be on the top of the stack so we push it last
            work.push(std::make_pair(myId, currentNode->child2));
            work.push(std::make_pair(myId, currentNode->child1));
        }
        else
        {
            seqBvh.push_back(BVHNode::MakeChild(boundingBox, t_start, t_count));
        }
    }
}

class Scene
{
public:
    std::vector<Triangle> triangles;
    void addModel(std::string filename, float scale, float3 rotation, float3 offset, Material material)
    {
        printf("Loading model %s\n", filename.c_str());
        tinyobj::ObjReaderConfig objConfig;
        objConfig.vertex_color = false;
        tinyobj::ObjReader objReader;
        objReader.ParseFromFile(filename, objConfig);

        Matrix4 transform = Matrix4::FromTranslation(offset.x, offset.y, offset.z) * Matrix4::FromScale(scale) * Matrix4::FromAxisRotations(rotation.x, rotation.y, rotation.z);

        for(int i=0; i<objReader.GetShapes()[0].mesh.indices.size(); i+=3)
        {
            auto it0 = objReader.GetShapes()[0].mesh.indices[i+0];
            auto it1 = objReader.GetShapes()[0].mesh.indices[i+1];
            auto it2 = objReader.GetShapes()[0].mesh.indices[i+2];
            auto vertices = objReader.GetAttrib().vertices;
            float3 v0 = make_float3(vertices[it0.vertex_index * 3 + 0], vertices[it0.vertex_index * 3 + 1], vertices[it0.vertex_index * 3 + 2]);
            float3 v1 = make_float3(vertices[it1.vertex_index * 3 + 0], vertices[it1.vertex_index * 3 + 1], vertices[it1.vertex_index * 3 + 2]);
            float3 v2 = make_float3(vertices[it2.vertex_index * 3 + 0], vertices[it2.vertex_index * 3 + 1], vertices[it2.vertex_index * 3 + 2]);

            Vector4 v0_tmp = transform * Vector4(v0.x, v0.y, v0.z, 1);
            Vector4 v1_tmp = transform * Vector4(v1.x, v1.y, v1.z, 1);
            Vector4 v2_tmp = transform * Vector4(v2.x, v2.y, v2.z, 1);

            v0 = make_float3(v0_tmp.x, v0_tmp.y, v0_tmp.z);
            v1 = make_float3(v1_tmp.x, v1_tmp.y, v1_tmp.z);
            v2 = make_float3(v2_tmp.x, v2_tmp.y, v2_tmp.z);

            float3 n0, n1, n2;

            if (it0.normal_index == -1 || it1.normal_index == -1 | it2.normal_index == -1)
            {
                float3 edge1 = v1 - v0;
                float3 edge2 = v2 - v0;
                n2 = n1 = n0 = normalize(cross(edge1, edge2));
            }
            else {
                auto normals = objReader.GetAttrib().normals;
                n0 = make_float3(normals[it0.normal_index * 3 + 0], normals[it0.normal_index * 3 + 1], normals[it0.normal_index * 3 + 2]);
                n1 = make_float3(normals[it1.normal_index * 3 + 0], normals[it1.normal_index * 3 + 1], normals[it1.normal_index * 3 + 2]);
                n2 = make_float3(normals[it2.normal_index * 3 + 0], normals[it2.normal_index * 3 + 1], normals[it2.normal_index * 3 + 2]);
            }

            triangles.push_back(Triangle { v0, v1, v2, n0, n1, n2, material});
        }
    }

    BVHTree* finalize() const { return createBVH(triangles); }
};

#endif
