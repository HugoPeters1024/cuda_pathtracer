#ifndef H_BVH_BUILDER
#define H_BVH_BUILDER

#include "types.h"
#include "constants.h"

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
        make_float3(min_x - EPS, min_y - EPS, min_z - EPS), 
        make_float3(max_x + EPS, max_y + EPS, max_z + EPS)
    };
}

float approximateBVHCost(std::vector<Triangle> triangles, int credit, int* min_level)
{
    // Maximum depth reached, use approximation
    if (credit == 0) return buildTriangleBox(triangles).volume() * triangles.size();

    float min_cost = std::numeric_limits<float>::max();

    // iterate over the 3 dimensions to find the most profitable splitting plane.
    for(int level=0; level<3; level++)
    {
        switch(level) {
            case 0: std::sort(triangles.begin(), triangles.end(), __compare_triangles_x); break;
            case 1: std::sort(triangles.begin(), triangles.end(), __compare_triangles_y); break;
            case 2: std::sort(triangles.begin(), triangles.end(), __compare_triangles_z); break;
        }

        int m = triangles.size() / 2;

        std::vector<Triangle> lows(triangles.begin(), triangles.begin() + m);
        std::vector<Triangle> highs(triangles.begin() + m, triangles.end());

        assert( lows.size() + highs.size() == triangles.size() );

        int tmp;
        float cost = approximateBVHCost(lows, credit-1, &tmp) + approximateBVHCost(highs, credit-1, &tmp);
        if (cost < min_cost)
        {
            *min_level = level;
            min_cost = cost;
        }
    }

    // Calculate the cost of not splitting
    float currentCost = buildTriangleBox(triangles).volume() * triangles.size();
    if (currentCost <= min_cost)
    {
        min_cost = currentCost;
        *min_level = -1;
    }
    return min_cost;
}


BVHTree* createBVH(std::vector<Triangle> triangles)
{
    BVHTree* ret = new BVHTree();
    ret->isLeaf = false;
    ret->child1 = nullptr;
    ret->child2 = nullptr;
    ret->triangles = std::vector<Triangle>();
    ret->boundingBox = buildTriangleBox(triangles);

    int min_level;
    float min_cost = approximateBVHCost(triangles, 3, &min_level);
    ret->used_level = min_level;

    // Splitting with less than 8 triangles causes too much overhead
    if (min_level == -1 || triangles.size() < 8)
    {
        ret->triangles = triangles;
        ret->isLeaf = true;
        return ret;
    }


    // Sort the triangles one last time based on the level the heuristic gave us.
    switch (min_level) {
        case 0: std::sort(triangles.begin(), triangles.end(), __compare_triangles_x); break;
        case 1: std::sort(triangles.begin(), triangles.end(), __compare_triangles_y); break;
        case 2: std::sort(triangles.begin(), triangles.end(), __compare_triangles_z); break;
    }

    int m = triangles.size() / 2;
    std::vector<Triangle> lows(triangles.begin(), triangles.begin() + m);
    std::vector<Triangle> highs(triangles.begin() + m, triangles.end());

    assert (lows.size() > 0 && highs.size() > 0);

    // child1 must be the near child
    ret->child1 = createBVH(lows);
    ret->child2 = createBVH(highs);
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

        BVHNode node;
        node.child1 = 0;
        node.child2 = 0;
        node.boundingBox = currentNode->boundingBox;
        node.parent = discoveredBy;
        node.split_plane = currentNode->used_level;

        // The node adds it's segment
        node.t_start = newTriangles.size();
        node.t_count = currentNode->triangles.size();

        // Add the triangles
        for (const Triangle& t : currentNode->triangles) newTriangles.push_back(t);


        // Calculate the indices of the children before hand
        uint myId = seqBvh.size();
        if (node.t_count == 0) // aka not a leaf
        {
            node.child1 = myId + 1;
            node.child2 = myId + 1 + currentNode->child1->treeSize();

            // child 1 should be on the top of the stack so we push it last
            work.push(std::make_pair(myId, currentNode->child2));
            work.push(std::make_pair(myId, currentNode->child1));
        }
        seqBvh.push_back(node);
    }
}

// Sanity checks
bool verifyBVHTree(const BVHTree* root)
{
    std::stack<const BVHTree*> work;
    work.push(root);

    while (!work.empty())
    {
        const BVHTree* current = work.top();
        work.pop();

        if ((current->child1 == nullptr) ^ (current->child2 == nullptr))
        {
            printf("Partial nodes detected in tree, only one child\n");
            return false;
        }
    }

    return true;
}

class Scene
{
public:
    std::vector<Triangle> triangles;
    void addModel(std::string filename, float3 color, float scale, float3 offset, float reflect)
    {
        printf("Loading model %s\n", filename.c_str());
        tinyobj::ObjReaderConfig objConfig;
        objConfig.vertex_color = false;
        tinyobj::ObjReader objReader;
        objReader.ParseFromFile(filename, objConfig);

        for(int i=0; i<objReader.GetShapes()[0].mesh.indices.size(); i+=3)
        {
            auto it0 = objReader.GetShapes()[0].mesh.indices[i+0];
            auto it1 = objReader.GetShapes()[0].mesh.indices[i+1];
            auto it2 = objReader.GetShapes()[0].mesh.indices[i+2];
            auto vertices = objReader.GetAttrib().vertices;
            float3 v0 = offset + scale * make_float3(vertices[it0.vertex_index * 3 + 0], vertices[it0.vertex_index * 3 + 1], vertices[it0.vertex_index * 3 + 2]);
            float3 v1 = offset + scale * make_float3(vertices[it1.vertex_index * 3 + 0], vertices[it1.vertex_index * 3 + 1], vertices[it1.vertex_index * 3 + 2]);
            float3 v2 = offset + scale * make_float3(vertices[it2.vertex_index * 3 + 0], vertices[it2.vertex_index * 3 + 1], vertices[it2.vertex_index * 3 + 2]);

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
            triangles.push_back(Triangle { v0, v1, v2, n0, n1, n2, color, reflect});
        }
    }

    BVHTree* finalize() const { return createBVH(triangles); }
};

#endif
