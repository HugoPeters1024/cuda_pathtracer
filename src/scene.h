#ifndef H_SCENE
#define H_SCENE

#include "types.h"
#include "bvhBuilder.h"

// Final product of the scene
struct SceneData
{
    TriangleV* h_vertex_buffer;
    TriangleD* h_data_buffer;
    BVHNode* h_bvh_buffer;
    Material* h_material_buffer;
    Sphere* h_sphere_buffer;
    Plane* h_plane_buffer;
    PointLight* h_point_light_buffer;
    SphereLight* h_sphere_light_buffer;
    uint num_triangles;
    uint num_bvh_nodes;
    uint num_materials;
    uint num_spheres;
    uint num_planes;
    uint num_point_lights;
    uint num_sphere_lights;
};

class Scene
{
public:
    std::vector<TriangleV> trianglesV;
    std::vector<TriangleD> trianglesD;
    std::vector<Material> materials;
    std::vector<Sphere> spheres;
    std::vector<Plane> planes;
    std::vector<PointLight> pointLights;
    std::vector<SphereLight> sphereLights;

    MATERIAL_ID addMaterial(Material material)
    {
        materials.push_back(material);
        return materials.size() - 1;
    }

    void addSphere(Sphere sphere) { spheres.push_back(sphere); }
    void addPlane(Plane plane) { planes.push_back(plane); }
    void addPointLight(PointLight light) { pointLights.push_back(light); }
    void addSphereLight(SphereLight light) { sphereLights.push_back(light); }

    void addModel(std::string filename, float scale, float3 rotation, float3 offset, MATERIAL_ID material, bool useMtl = false)
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

        for(int s=0; s<objReader.GetShapes().size(); s++)
        {
            const auto& shape = objReader.GetShapes()[s];
            trianglesV.reserve(trianglesV.size() + shape.mesh.indices.size() / 3);
            trianglesD.reserve(trianglesD.size() + shape.mesh.indices.size() / 3);
            for(int i=0; i<shape.mesh.indices.size(); i+=3)
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

                float3 n0, n1, n2;

                if (it0.normal_index == -1 || it1.normal_index == -1 || it2.normal_index == -1)
                {
                    float3 edge1 = v1 - v0;
                    float3 edge2 = v2 - v0;
                    n2 = n1 = n0 = normalize(cross(edge1, edge2));
                }
                else {
                    n0 = make_float3(normals[it0.normal_index*3+0], normals[it0.normal_index*3+1], normals[it0.normal_index*3+2]);
                    n1 = make_float3(normals[it1.normal_index*3+0], normals[it1.normal_index*3+1], normals[it1.normal_index*3+2]);
                    n2 = make_float3(normals[it2.normal_index*3+0], normals[it2.normal_index*3+1], normals[it2.normal_index*3+2]);
                }


                Matrix4 TBN;
                if (useMtl && materials[material_ids[mit]].hasNormalMap)
                {
                    float3 edge1 = v1 - v0;
                    float3 edge2 = v2 - v0;
                    float2 deltaUV1 = uv1 - uv0;
                    float2 deltaUV2 = uv2 - uv0;

                    float f = 1.0f / (deltaUV1.x * deltaUV2.y - deltaUV2.x * deltaUV1.y);
                    float3 tangent = f * (deltaUV2.y * edge1 - deltaUV1.y * edge2);
                    float3 bitangent = f * (deltaUV1.x * edge2 - deltaUV2.x * edge1);

                    TBN = Matrix4::FromColumnVectors(
                            Vector3(tangent.x, tangent.y, tangent.z),
                            Vector3(bitangent.x, bitangent.y, bitangent.z),
                            Vector3(n0.x, n0.y, n0.z));
                }

                trianglesV.push_back(TriangleV(v0, v1, v2));
                trianglesD.push_back(TriangleD(n0, n1, n2, uv0, uv1, uv2, useMtl ? material_ids[mit] : material, TBN));
            }
        }
    }

    SceneData finalize()
    {
        assert(sphereLights.size() > 0);
        SceneData ret;

        printf("Finalizing scene, total triangles: %u\n", trianglesV.size());

        printf("Building a BVH...\n");
        uint bvhSize;
        ret.h_bvh_buffer = createBVHBinned(trianglesV, trianglesD, &bvhSize);
        printf("BVH Size: %u\n", bvhSize);

        ret.h_vertex_buffer = trianglesV.data();
        ret.h_data_buffer = trianglesD.data();

        ret.h_material_buffer = materials.data();
        ret.h_sphere_buffer = spheres.data();
        ret.h_plane_buffer = planes.data();

        // copy over the point lights
        ret.h_point_light_buffer = pointLights.data();

        // copy over the sphere lights
        ret.h_sphere_light_buffer = sphereLights.data();

        assert(trianglesV.size() == trianglesD.size());
        ret.num_triangles = trianglesV.size();
        ret.num_bvh_nodes = bvhSize;
        ret.num_materials = materials.size();
        ret.num_spheres = spheres.size();
        ret.num_planes = planes.size();
        ret.num_point_lights = pointLights.size();
        ret.num_sphere_lights = sphereLights.size();
        return ret;
    }
};
#endif
