#ifndef H_SCENE
#define H_SCENE

#include "types.h"
#include "bvhBuilder.h"


inline Instance ConvertToInstance(const GameObject& obj)
{
    glm::mat4x4 transform = glm::mat4x4(1.0f);
    transform = glm::translate(transform, glm::vec3(obj.position.x, obj.position.y, obj.position.z));
    transform = glm::rotate(transform, obj.rotation.x, glm::vec3(1.0f, 0.0f, 0.0f));
    transform = glm::rotate(transform, obj.rotation.y, glm::vec3(0.0f, 1.0f, 0.0f));
    transform = glm::rotate(transform, obj.rotation.z, glm::vec3(0.0f, 0.0f, 1.0f));
    transform = glm::scale(transform, glm::vec3(obj.scale.x, obj.scale.y, obj.scale.z));
    return Instance
    {
        obj.model_id,
        transform,
        glm::inverse(transform)
    };
}

inline Box transformBox(const Box& box, const glm::mat4x4& transform)
{
    float3 points[8];
    float3 minToMax = box.vmax - box.vmin;
    points[0] = box.vmin;
    points[1] = box.vmin + make_float3(minToMax.x, 0, 0);
    points[2] = box.vmin + make_float3(0, minToMax.y, 0);
    points[3] = box.vmin + make_float3(0, 0, minToMax.z);

    points[4] = box.vmax;
    points[5] = box.vmax - make_float3(minToMax.x, 0, 0);
    points[6] = box.vmax - make_float3(0, minToMax.y, 0);
    points[7] = box.vmax - make_float3(0, 0, minToMax.z);

    Box ret = Box::insideOut();
    for(const float3& p : points) {
        glm::vec4 wp = transform * glm::vec4(p.x, p.y, p.z, 1);
        ret.consumePoint(make_float3(wp.x, wp.y, wp.z));
    }
    return ret;
}

struct Indexed
{
    uint index;
    TopLevelBVH item;
};

inline Indexed FindBestMatch(const uint node, const std::map<uint, Indexed>& list)
{
    float minSurface = std::numeric_limits<float>::max();
    bool found = false;
    Indexed match;
    Box nodeBox = list.at(node).item.box;

    auto it = list.begin();
    for(uint i=0; i<list.size(); i++, it = std::next(it))
    {
        const Indexed& item = it->second;
        if (item.index == node) continue;
        Box newBox = nodeBox;
        newBox.consumeBox(item.item.box);
        if (newBox.getSurfaceArea() < minSurface)
        {
            minSurface = newBox.getSurfaceArea();
            match = item;
            found = true;
        }
    }

    return found ? match : list.at(node);
}


inline void BuildTopLevelBVH(TopLevelBVH* dest, const Instance* instances, const Model* models, uint num_instances)
{
    // The toplevel bvh leaves should have the same bounding box as the bvh but transformed.
    std::map<uint, Indexed> list;
    uint node_count = 2 * num_instances - 1;
    for(uint i=0; i<num_instances; i++) {
        node_count--;
        Box box = transformBox(models[instances[i].model_id].bvh[0].getBox(), instances[i].transform);
        TopLevelBVH node = TopLevelBVH::CreateLeaf(i, box);
        dest[node_count] = node;
        list[node_count] = Indexed { node_count, node };
    }

    Indexed A = list.begin()->second;
    Indexed B = FindBestMatch(A.index, list);

    while(list.size() > 1)
    {
        Indexed C = FindBestMatch(B.index, list);
        if (A.index == C.index)
        {
            list.erase(A.index);
            list.erase(B.index);
            TopLevelBVH parent = TopLevelBVH::CreateNode(A.index, B.index, Box::merged(A.item.box, B.item.box));
            A = Indexed { --node_count, parent };
            dest[A.index] = A.item;
            list[A.index] = A;
            B = FindBestMatch(A.index, list);
        }
        else
        {
            A = B;
            B = C;
        }
    }

    assert(node_count == 0);
}

class Scene
{
    // derivatives of gameobjects
public:
    std::vector<Model> models = std::vector<Model>(0);
    std::vector<GameObject> objects = std::vector<GameObject>(0);
    Instance* instances;
    std::vector<Material> materials = std::vector<Material>(0);
    std::vector<Sphere> spheres = std::vector<Sphere>(0);
    std::vector<Plane> planes = std::vector<Plane>(0);
    std::vector<PointLight> pointLights = std::vector<PointLight>(0);
    std::vector<SphereLight> sphereLights = std::vector<SphereLight>(0);
    std::vector<TopLevelBVH> topLevelBVH;
    std::vector<std::function<void(std::vector<GameObject>&, float)>> handlers;

    MATERIAL_ID addMaterial(Material material)
    {
        materials.push_back(material);
        return materials.size() - 1;
    }

    void addSphere(Sphere sphere) { spheres.push_back(sphere); }
    void addPlane(Plane plane) { planes.push_back(plane); }
    void addPointLight(PointLight light) { pointLights.push_back(light); }
    void addSphereLight(SphereLight light) { sphereLights.push_back(light); }
    void addObject(GameObject object) { objects.push_back(object); }
    void addHandler(std::function<void(std::vector<GameObject>&, float)> handler) { handlers.push_back(handler); }

    uint addModel(std::string filename, float scale, float3 rotation, float3 offset, MATERIAL_ID material, bool useMtl = false)
    {
        printf("Loading model %s\n", filename.c_str());
        tinyobj::ObjReaderConfig objConfig;
        objConfig.vertex_color = false;
        objConfig.triangulate = true;
        tinyobj::ObjReader objReader;
        objReader.ParseFromFile(filename, objConfig);
        if (!objReader.Valid())
        {
            printf("Tinyobj could not load the model...\n");
            exit(1);
        }


        std::map<std::string, cudaTextureObject_t> textureItems;

        Matrix4 transform = Matrix4::FromTranslation(offset.x, offset.y, offset.z) * Matrix4::FromScale(scale) * Matrix4::FromAxisRotations(rotation.x, rotation.y, rotation.z);

        MATERIAL_ID material_ids[objReader.GetMaterials().size()];

        if (useMtl)
        {
            printf("Loading %lu materials\n", objReader.GetMaterials().size());
            for(int m=0; m<objReader.GetMaterials().size(); m++)
            {
                auto mat = objReader.GetMaterials()[m];
                Material material = Material::DIFFUSE(make_float3(1));
                material.diffuse_color = make_float3(mat.diffuse[0], mat.diffuse[1], mat.diffuse[2]);
                material.specular_color = make_float3(mat.specular[0], mat.specular[1], mat.specular[2]);
                material.transmit = 1-mat.dissolve;
                material.reflect = (mat.specular[0]+mat.specular[1]+mat.specular[2])/3.0f;
                material.glossy = mat.shininess / 4000.0f;

                // Ensure that we don't get crazy values by normalizing the
                // different components
                float sum = material.transmit + material.reflect;
                if (sum > 1)
                {
                    float scaleFactor = 1.0f / sum;
                    material.transmit *= scaleFactor;
                    material.reflect *= scaleFactor;
                }

                assert(material.transmit + material.reflect <= 1);
                material.refractive_index = mat.ior;
                if (mat.diffuse_texname != "")
                {
                    if (textureItems.find(mat.diffuse_texname) == textureItems.end())
                    {
                        // not found, load and add to map
                        cudaTextureObject_t texture = loadTexture(mat.diffuse_texname.c_str());
                        material.texture = texture;
                        textureItems[mat.diffuse_texname] = texture;
                    }
                    else
                    {
                        // already loaded
                        material.texture = textureItems.at(mat.diffuse_texname);
                    }
                    material.hasTexture = true;
                }

                if (mat.normal_texname != "")
                {
                    if (textureItems.find(mat.normal_texname) == textureItems.end())
                    {
                        // not found, load and add to map
                        cudaTextureObject_t normal_texture = loadTexture(mat.normal_texname.c_str());
                        material.normal_texture = normal_texture;
                        textureItems[mat.normal_texname] = normal_texture;
                    }
                    else
                    {
                        // already loaded
                        material.normal_texture = textureItems.at(mat.normal_texname);
                    }
                    material.hasNormalMap = true;
                }


                material_ids[m] = addMaterial(material);
            }
        }

        const auto& vertices = objReader.GetAttrib().vertices;
        const auto& normals = objReader.GetAttrib().normals;
        const auto& uvs = objReader.GetAttrib().texcoords;
        bool hasUvs = uvs.size() > 0;

        Model model;
        model.nrTriangles = 0;
        for(int s=0; s<objReader.GetShapes().size(); s++)
        {
            const auto& shape = objReader.GetShapes()[s];
            model.nrTriangles += shape.mesh.indices.size() / 3;
        }

        model.trianglesV = (TriangleV*)malloc(model.nrTriangles * sizeof(TriangleV));
        model.trianglesD = (TriangleD*)malloc(model.nrTriangles * sizeof(TriangleD));

        uint triangle_index = 0;
        for(int s=0; s<objReader.GetShapes().size(); s++)
        {
            const auto& shape = objReader.GetShapes()[s];
            for(int i=0; i<shape.mesh.indices.size(); i+=3, triangle_index++)
            {
                const auto& it0 = shape.mesh.indices[i+0];
                const auto& it1 = shape.mesh.indices[i+1];
                const auto& it2 = shape.mesh.indices[i+2];
                auto mit = shape.mesh.material_ids[i/3];
                float3 v0 = make_float3(vertices[it0.vertex_index*3+0], vertices[it0.vertex_index*3+1], vertices[it0.vertex_index*3+2]);
                float3 v1 = make_float3(vertices[it1.vertex_index*3+0], vertices[it1.vertex_index*3+1], vertices[it1.vertex_index*3+2]);
                float3 v2 = make_float3(vertices[it2.vertex_index*3+0], vertices[it2.vertex_index*3+1], vertices[it2.vertex_index*3+2]);

                float2 uv0 = hasUvs ? make_float2(uvs[it0.texcoord_index*2+0], uvs[it0.texcoord_index*2+1]) : make_float2(0);
                float2 uv1 = hasUvs ? make_float2(uvs[it1.texcoord_index*2+0], uvs[it1.texcoord_index*2+1]) : make_float2(0);
                float2 uv2 = hasUvs ? make_float2(uvs[it2.texcoord_index*2+0], uvs[it2.texcoord_index*2+1]) : make_float2(0);
                if (useMtl)
                {
                    // MTL files suck!!!!
                    auto mat = objReader.GetMaterials()[mit];
                    float2 offset = make_float2(mat.diffuse_texopt.origin_offset[0], mat.diffuse_texopt.origin_offset[1]);
                    uv0 = uv0 + offset;
                    uv1 = uv1 + offset;
                    uv2 = uv2 + offset;
                }

                Vector4 v0_tmp = transform * Vector4(v0.x, v0.y, v0.z, 1);
                Vector4 v1_tmp = transform * Vector4(v1.x, v1.y, v1.z, 1);
                Vector4 v2_tmp = transform * Vector4(v2.x, v2.y, v2.z, 1);

                v0 = make_float3(v0_tmp.x, v0_tmp.y, v0_tmp.z);
                v1 = make_float3(v1_tmp.x, v1_tmp.y, v1_tmp.z);
                v2 = make_float3(v2_tmp.x, v2_tmp.y, v2_tmp.z);

                float3 normal, tangent, bitangent;

                if (it0.normal_index == -1 || it1.normal_index == -1 || it2.normal_index == -1)
                {
                    float3 edge1 = v1 - v0;
                    float3 edge2 = v2 - v0;
                    normal = normalize(cross(edge1, edge2));
                }
                else {
                    normal = make_float3(normals[it0.normal_index*3+0], normals[it0.normal_index*3+1], normals[it0.normal_index*3+2]);
                    //n1 = make_float3(normals[it1.normal_index*3+0], normals[it1.normal_index*3+1], normals[it1.normal_index*3+2]);
                    //n2 = make_float3(normals[it2.normal_index*3+0], normals[it2.normal_index*3+1], normals[it2.normal_index*3+2]);
                }


                if (useMtl && materials[material_ids[mit]].hasNormalMap)
                {
                    float3 edge1 = v1 - v0;
                    float3 edge2 = v2 - v0;
                    float2 deltaUV1 = uv1 - uv0;
                    float2 deltaUV2 = uv2 - uv0;

                    float f = 1.0f / (deltaUV1.x * deltaUV2.y - deltaUV2.x * deltaUV1.y);
                    tangent = f * (deltaUV2.y * edge1 - deltaUV1.y * edge2);
                    bitangent = f * (deltaUV1.x * edge2 - deltaUV2.x * edge1);
                }

                model.trianglesV[triangle_index] = TriangleV(v0, v1, v2);
                model.trianglesD[triangle_index] = TriangleD(normal, tangent, bitangent, uv0, uv1, uv2, useMtl ? material_ids[mit] : material);
            }
        }
        printf("Building a BVH over %u triangles\n", model.nrTriangles);
        float ping = glfwGetTime();
        model.bvh = createBVHBinned(model.trianglesV, model.trianglesD, model.nrTriangles, &model.nrBvhNodes);
        printf("Build took %fms\n", (glfwGetTime() - ping) *1000);
        printf("BVH Size: %u\n", model.nrBvhNodes);

        models.push_back(model);
        return models.size() - 1;
    }

    void validate()
    {
        // I can't deal with that lol
        assert(sphereLights.size() > 0);
    }

    void finalize()
    {
        validate();
        // Allocate the upperbound of the toplevel bvh size
        topLevelBVH = std::vector<TopLevelBVH>(2 * objects.size() - 1);

        // Allocate the instance buffer
        instances = (Instance*)malloc(objects.size() * sizeof(Instance));

        // derive instances
        for(int i=0; i<objects.size(); i++) instances[i] = ConvertToInstance(objects[i]);
    }

    void update(float currentTime)
    {
        // update gameobjects
        for(auto& handler : handlers) handler(objects, currentTime);

        // derive instances
        for(int i=0; i<objects.size(); i++) instances[i] = ConvertToInstance(objects[i]);

        // build top level bvh
        BuildTopLevelBVH(topLevelBVH.data(), instances, models.data(), objects.size());
    }
};
#endif
