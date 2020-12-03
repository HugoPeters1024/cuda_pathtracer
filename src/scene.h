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
    std::vector<Triangle> triangles;
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


                material_ids[m] = addMaterial(material);
            }
        }

        for(int s=0; s<objReader.GetShapes().size(); s++)
        {
            for(int i=0; i<objReader.GetShapes()[s].mesh.indices.size(); i+=3)
            {
                auto it0 = objReader.GetShapes()[s].mesh.indices[i+0];
                auto it1 = objReader.GetShapes()[s].mesh.indices[i+1];
                auto it2 = objReader.GetShapes()[s].mesh.indices[i+2];
                auto mit = objReader.GetShapes()[s].mesh.material_ids[i/3];
                auto vertices = objReader.GetAttrib().vertices;
                auto uvs = objReader.GetAttrib().texcoords;
                bool hasUvs = uvs.size() > 0;
                float3 v0 = make_float3(vertices[it0.vertex_index*3+0], vertices[it0.vertex_index*3+1], vertices[it0.vertex_index*3+2]);
                float3 v1 = make_float3(vertices[it1.vertex_index*3+0], vertices[it1.vertex_index*3+1], vertices[it1.vertex_index*3+2]);
                float3 v2 = make_float3(vertices[it2.vertex_index*3+0], vertices[it2.vertex_index*3+1], vertices[it2.vertex_index*3+2]);

                float2 uv0 = hasUvs ? make_float2(uvs[it0.texcoord_index*2+0], uvs[it0.texcoord_index*2+1]) : make_float2(0);
                float2 uv1 = hasUvs ? make_float2(uvs[it1.texcoord_index*2+0], uvs[it1.texcoord_index*2+1]) : make_float2(0);
                float2 uv2 = hasUvs ? make_float2(uvs[it2.texcoord_index*2+0], uvs[it2.texcoord_index*2+1]) : make_float2(0);

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
                    auto normals = objReader.GetAttrib().normals;
                    n0 = make_float3(normals[it0.normal_index*3+0], normals[it0.normal_index*3+1], normals[it0.normal_index*3+2]);
                    n1 = make_float3(normals[it1.normal_index*3+0], normals[it1.normal_index*3+1], normals[it1.normal_index*3+2]);
                    n2 = make_float3(normals[it2.normal_index*3+0], normals[it2.normal_index*3+1], normals[it2.normal_index*3+2]);
                }

                if (useMtl)
                {
                    // MTL files suck!!!!
                    auto mat = objReader.GetMaterials()[mit];
                    float2 offset = make_float2(mat.diffuse_texopt.origin_offset[0], mat.diffuse_texopt.origin_offset[1]);
                    uv0 = uv0 + offset;
                    uv1 = uv1 + offset;
                    uv2 = uv2 + offset;
                }

                triangles.push_back(Triangle { v0, v1, v2, n0, n1, n2, uv0, uv1, uv2, useMtl ? material_ids[mit] : material});
            }
        }
    }

    SceneData finalize()
    { 
        assert(sphereLights.size() > 0);
        SceneData ret;

        uint bvhSize;
        ret.h_bvh_buffer = createBVH(triangles, &bvhSize);
        printf("BVH Size: %u\n", bvhSize);

        // Split the vertices and other data for better caching
        ret.h_vertex_buffer = (TriangleV*)malloc(triangles.size() * sizeof(TriangleV));
        ret.h_data_buffer = (TriangleD*)malloc(triangles.size() * sizeof(TriangleD));
        for(int i=0; i<triangles.size(); i++)
        {
            const Triangle& t = triangles[i];
            ret.h_vertex_buffer[i] = TriangleV(t.v0, t.v1, t.v2);
            ret.h_data_buffer[i] = TriangleD(t.n0, t.n1, t.n2, t.uv0, t.uv1, t.uv2, t.material);
        }

        // copy over the materials
        ret.h_material_buffer = (Material*)malloc(materials.size() * sizeof(Material));
        memcpy(ret.h_material_buffer, materials.data(), materials.size() * sizeof(Material));

        // copy over the spheres
        ret.h_sphere_buffer = (Sphere*)malloc(spheres.size() * sizeof(Sphere));
        memcpy(ret.h_sphere_buffer, spheres.data(), spheres.size() * sizeof(Sphere));

        // copy over the planes
        ret.h_plane_buffer = (Plane*)malloc(planes.size() * sizeof(Plane));
        memcpy(ret.h_plane_buffer, planes.data(), planes.size() * sizeof(Plane));

        // copy over the point lights
        ret.h_point_light_buffer = (PointLight*)malloc(pointLights.size() * sizeof(PointLight));
        memcpy(ret.h_point_light_buffer, pointLights.data(), pointLights.size() * sizeof(PointLight));

        // copy over the sphere lights
        ret.h_sphere_light_buffer = (SphereLight*)malloc(sphereLights.size() * sizeof(SphereLight));
        memcpy(ret.h_sphere_light_buffer, sphereLights.data(), sphereLights.size() * sizeof(SphereLight));

        ret.num_triangles = triangles.size();
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
