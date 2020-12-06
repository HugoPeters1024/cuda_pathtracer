#ifndef VEC_H
#define VEC_H

#include <cmath>
#include <exception>
#include <limits>
#include <array>
#include "linmath.h"
#include "use_cuda.h"

class Vector2 {
public:
  float x, y;
  Vector2() : x(0), y(0) {}
  Vector2(float a) : x(a), y(a) {}
  Vector2(float x, float y) : x(x), y(y) {}

  Vector2& normalize() { float l = length(); x/=l; y/=l; return *this; }
  float length() const { return sqrt(x*x + y*y); }
  Vector2 operator * (const Vector2 &o) const { return Vector2(x*o.x, y*o.y); } 
  Vector2 operator * (float s) const    { return Vector2(x*s, y*s);     }
  Vector2 operator / (const Vector2 &o) const { return Vector2(x/o.x, y/o.y); } 
  Vector2 operator + (const Vector2 &o) const { return Vector2(x+o.x, y+o.y); } 
  Vector2& operator += (const Vector2 &o) { x += o.x; y += o.y; return *this; } 
  Vector2 operator - (const Vector2 &o) const { return Vector2(x-o.x, y-o.y); } 
  Vector2& operator -= (const Vector2 &o) { x -= o.x; y -= o.y; return *this; }
  Vector2 operator - () const { return Vector2(-x, -y); }
  void print() { printf("(%f, %f)\n", x, y); }
  static float dot(const Vector2 &a, const Vector2& b) { return a.x * b.x + a.y * b.y; }
};


class Vector3 {
public:
  float x, y, z;
  Vector3() : x(0), y(0), z(0) {}
  Vector3(float a) : x(a), y(a), z(a) {}
  Vector3(float x, float y, float z) : x(x), y(y), z(z) {}

  Vector3& normalize() { 
    float l = length(); 
    x /= l == 0 ? 1 : l; 
    y /= l == 0 ? 1 : l; 
    z /= l == 0 ? 1 : l; 
    return *this; }
  Vector3 normalized() const { 
    float l = length(); 
    return l == 0 ? Vector3(0) : Vector3(x/l, y/l, z/l); 
  }
  float length() const { return sqrt(x*x + y*y + z*z); }
  static Vector3 cross(const Vector3 &a, const Vector3 &b) {
    vec3 v;
    vec3 va = { a.x, a.y, a.z };
    vec3 vb = { b.x, b.y, b.z };
    vec3_mul_cross(v, va, vb);
    return Vector3(v[0], v[1], v[2]);
  }
  static float dot(const Vector3 &a, const Vector3 &b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
  float sq_length() const { return dot(*this, *this); }
  float largestComponent() const { 
    if (x >= y && x >= z) return x;
    if (y >= x && y >= z) return y;
    if (z >= x && z >= y) return z; 
  }
  Vector3 operator * (const Vector3 &o) const { return Vector3(x*o.x, y*o.y, z*o.z); } 
  Vector3 operator * (float s) const    { return Vector3(x*s, y*s, z*s);       }
  Vector3& operator *= (float s) { x*=s; y*=s; z*=s; return *this; }
  Vector3 operator / (const Vector3 &o) const { 
    float nx = o.x == 0 ? 0 : x/o.x;
    float ny = o.y == 0 ? 0 : y/o.y;
    float nz = o.z == 0 ? 0 : z/o.z;
    return Vector3(nx, ny, nz); }
  Vector3 operator + (const Vector3 &o) const { return Vector3(x+o.x, y+o.y, z+o.z); } 
  Vector3& operator += (const Vector3 &o) { x += o.x; y += o.y; z += o.z; return *this; }
  Vector3 operator - (const Vector3 &o) const { return Vector3(x-o.x, y-o.y, z-o.z); }
  Vector3& operator -= (const Vector3 &o) { x -= o.x; y -= o.y; z -= o.z; return *this; }
  Vector3 operator - () const { return Vector3(-x, -y, -z); }
  Vector2 xz() const { return Vector2(x, z); }
  void unpack(vec3 &v) const { v[0] = x, v[1] = y, v[2] = z; }
  void print() const { printf("(%f, %f, %f)\n", x, y, z); }

  static Vector3 reflect(const Vector3 &a, const Vector3 &n) {
    vec3 _a = { a.x, a.y, a.z };
    vec3 _n = { n.x, n.y, n.z };
    vec3 r;
    vec3_reflect(r, _a, _n);
    return Vector3(r[0], r[1], r[2]);
  }

  // Pointwise min
  static Vector3 min(const Vector3 &a, const Vector3 &b)
  {
    return Vector3(
           std::min(a.x, b.x),
           std::min(a.y, b.y),
           std::min(a.z, b.z));
  }

  // Pointwise max
  static Vector3 max(const Vector3 &a, const Vector3 &b)
  {
    return Vector3(
            std::max(a.x, b.x),
            std::max(a.y, b.y),
            std::max(a.z, b.z));
  }
};

//Commutative mapping
Vector3 operator * (float s, const Vector3 &o) { return o * s; }

class Vector4 {
public:
  float x, y, z, w;
  Vector4() : x(0), y(0), z(0), w(0) {};
  Vector4(float a) : x(a), y(a), z(a), w(a) {}
  Vector4(float x, float y, float z, float w) : x(x), y(y), z(z), w(w) {}
  Vector4(Vector3 a, float w) : Vector4(a.x, a.y, a.z, w) {}
  Vector4 operator * (float s) const { return Vector4(x*s, y*s, z*s, w*s); }
  Vector4 operator - () const { return Vector4(-x, -y, -z, -w); }
  float length() const { return sqrt(x * x + y * y + z * z + w * w); }
  Vector4 normalized() const { 
    float l = length();
    return Vector4(x/l, y/l, z/l, w/l);
  }

  Vector3 xyz() const { return Vector3(x, y, z); }
  void print() const { printf("(%f, %f, %f, %f)\n", x, y, z, w); }
};

Vector4 operator * (float s, const Vector4 &o) { return o * s; }

class Matrix4 {
private:
  mat4x4 data;
public:
  Matrix4() {}
  Matrix4(const mat4x4 data) { mat4x4_dup(this->data, data); }
  Matrix4(const Matrix4 &m) : Matrix4(m.data) {}
  static Matrix4 FromColumnVectors(Vector3 v0, Vector3 v1, Vector3 v2) {
    mat4x4 m;
    vec4 vv0 = { v0.x, v0.y, v0.z, 0 };
    vec4 vv1 = { v1.x, v1.y, v1.z, 0 };
    vec4 vv2 = { v2.x, v2.y, v2.z, 0 };
    vec4 vv3 = { 0, 0, 0, 1};
    mat4x4_set_col(m, vv0, 0);
    mat4x4_set_col(m, vv1, 1);
    mat4x4_set_col(m, vv2, 2);
    mat4x4_set_col(m, vv3, 3);
    return Matrix4(m);
  }
  void unpack(mat4x4 f) const { mat4x4_dup(f, data); }
  float* operator[] (int column) { return data[column]; }
  Matrix4 inverted() const { mat4x4 r; mat4x4_invert(r, data); return Matrix4(r); }
  Vector4 operator* (const Vector4 &o) const {
    vec4 r;
    vec4 i = { o.x, o.y, o.z, o.w };
    mat4x4_mul_vec4(r, data, i);
    return Vector4(r[0], r[1], r[2], r[3]);
  }
  HYBRID float4 operator* (const float4& o) const {
      vec4 r;
      vec4 i = { o.x, o.y, o.z, o.w };
      mat4x4_mul_vec4(r, data, i);
      return make_float4(r[0], r[1], r[2], r[3]);
  }
  Matrix4 operator * (const Matrix4 &o) const {
    mat4x4 r;
    mat4x4_mul(r, data, o.data);
    return Matrix4(r);
  }
  static Matrix4 Identity() { mat4x4 r; mat4x4_identity(r); return Matrix4(r); } 
  static Matrix4 FromTranslation(float x, float y, float z) {
    mat4x4 r;
    mat4x4_translate(r, x, y, z);
    return Matrix4(r);
  }
  static Matrix4 FromTranslation(Vector3 v) { return FromTranslation(v.x, v.y, v.z); }
  static Matrix4 FromScale(float x, float y, float z) {
    mat4x4 r;
    mat4x4_identity(r);
    mat4x4_scale_aniso(r, r, x, y, z);
    return Matrix4(r);
  }
  static Matrix4 FromScale(float s) { return FromScale(s,s,s); }
  static Matrix4 FromScale(Vector3 v) { return FromScale(v.x, v.y, v.z); }
  static Matrix4 FromAxisRotations(float xr, float yr, float zr) {
    mat4x4 rx, ry, rz, i;
    mat4x4_identity(i);
    mat4x4_rotate_X(rx, i, xr);
    mat4x4_rotate_Y(ry, i, yr);
    mat4x4_rotate_Z(rz, i, zr);
    Matrix4 Rx = Matrix4(rx);
    Matrix4 Ry = Matrix4(ry);
    Matrix4 Rz = Matrix4(rz);
    return Rx * Ry * Rz;
  }
  static Matrix4 FromAxisRotations(Vector3 v) { return FromAxisRotations(v.x, v.y, v.z); }
  static Matrix4 FromNormal(Vector3 normal) {
    Vector3 from = Vector3(0, 0, 1);
    Vector3 a = Vector3::cross(from, normal).normalized();
    float alpha = acos(Vector3::dot(from, normal));
    float s = sin(alpha);
    float c = cos(alpha);

    Vector3 c1 = Vector3(a.x * a.x * (1-c) + c, a.x * a.y * (1-c) + s * a.z, a.x * a.z * (1-c) - s * a.y);
    Vector3 c2 = Vector3(a.x * a.y * (1-c) - s * a.z, a.y * a.y * (1-c) + c, a.y * a.z * (1-c) + s * a.x);
    Vector3 c3 = Vector3(a.x * a.z * (1-c) + s * a.y, a.y * a.z * (1-c) - s * a.x, a.z * a.z * (1-c) + c); 
    return Matrix4::FromColumnVectors(c1, c2, c3);
  }
  static Matrix4 FromPerspective(float fov, float ratio, float znear, float zfar) {
    mat4x4 r;
    mat4x4_perspective(r, fov, ratio, znear, zfar);
    return Matrix4(r);
  }
  static Matrix4 FromArray(const float* data)
  {
    mat4x4 r;
    r[0][0] = data[0];
    r[0][1] = data[1];
    r[0][2] = data[2];
    r[0][3] = data[3];

    r[1][0] = data[4];
    r[1][1] = data[5];
    r[1][2] = data[6];
    r[1][3] = data[7];

    r[2][0] = data[8];
    r[2][1] = data[9];
    r[2][2] = data[10];
    r[2][3] = data[11];

    r[3][0] = data[12];
    r[3][1] = data[13];
    r[3][2] = data[14];
    r[3][3] = data[15];
    return Matrix4(r);
  }
  void print() const {
    for(int i=0; i<4; i++) {
      printf("%f, %f, %f, %f\n", data[0][i], data[1][i], data[2][i], data[3][i]);
    }
    printf("---------\n");
  }
};

struct BoundingBox {
    Vector3 min;
    Vector3 max;

    BoundingBox() {}

    BoundingBox(Vector3 min, Vector3 max)
      : min(min), max(max) {}
};

BoundingBox operator * (BoundingBox box, const Vector3 &scale) { return {box.min * scale, box.max * scale}; }

#endif
