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
    uint num_triangles;
    uint num_bvh_nodes;
    uint num_materials;
    uint num_spheres;
    uint num_planes;
    uint num_point_lights;
};

class Scene
{
public:
    std::vector<Triangle> triangles;
    std::vector<Material> materials;
    std::vector<Sphere> spheres;
    std::vector<Plane> planes;
    std::vector<PointLight> pointLights;

    MATERIAL_ID addMaterial(Material material)
    {
        materials.push_back(material);
        return materials.size() - 1;
    }

    void addSphere(Sphere sphere) { spheres.push_back(sphere); }
    void addPlane(Plane plane) { planes.push_back(plane); }
    void addPointLight(PointLight light) { pointLights.push_back(light); }

    void addModel(std::string filename, float scale, float3 rotation, float3 offset, MATERIAL_ID material)
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

            if (it0.normal_index == -1 || it1.normal_index == -1 || it2.normal_index == -1)
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

    SceneData finalize()
    { 
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
            ret.h_data_buffer[i] = TriangleD(t.n0, t.n1, t.n2, t.material);
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

        ret.num_triangles = triangles.size();
        ret.num_bvh_nodes = bvhSize;
        ret.num_materials = materials.size();
        ret.num_spheres = spheres.size();
        ret.num_planes = planes.size();
        ret.num_point_lights = pointLights.size();
        return ret;
    }
};
#endif
