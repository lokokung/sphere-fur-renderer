#pragma once

#include <string>

#include "cereal/archives/json.hpp"
#include "Eigen/Dense"

#define USE_DOUBLE_PRECISION 1

#define SPHERE_MAX_HAIR_LAYERS 5
#define SCENE_MAX_LIGHTS 5

// Toggle to use single or double precision
#if USE_DOUBLE_PRECISION
typedef Eigen::Vector3d Vector3;
typedef double decimal;
#endif
#if !USE_DOUBLE_PRECISION
typedef Eigen::Vector3f Vector3;
typedef float decimal;
#endif

// Position class to signify a position
class Position : public Vector3 {
public:
    Position(void) : Vector3() {}

    template<typename OtherDerived>
    Position(const Eigen::MatrixBase<OtherDerived>& other)
        : Vector3(other) {}

    template<typename OtherDerived>
    Position& operator=(const Eigen::MatrixBase<OtherDerived>& other) {
        this->Vector3::operator=(other);
        return *this;
    }

    // Serialization function
    template<class Archive> void serialize(Archive& archive) {
        archive(cereal::make_nvp("x", this->operator()(0)),
                cereal::make_nvp("y", this->operator()(1)),
                cereal::make_nvp("z", this->operator()(2)));
    }
};

// Color class to signify a position
class Color : public Vector3 {
public:
    Color(void) : Vector3() {}

    template<typename OtherDerived>
    Color(const Eigen::MatrixBase<OtherDerived>& other)
        : Vector3(other) {}

    template<typename OtherDerived>
    Color& operator=(const Eigen::MatrixBase<OtherDerived>& other) {
        this->Vector3::operator=(other);
        return *this;
    }

    // Serialization function
    template<class Archive> void serialize(Archive& archive) {
        archive(cereal::make_nvp("r", this->operator()(0)),
                cereal::make_nvp("g", this->operator()(1)),
                cereal::make_nvp("b", this->operator()(2)));
    }
};

// Camera structure to hold information about a camera
typedef struct camera_t {
    Position pos;
    decimal near, far, fov, aspect;

    // Serialization function
    template<class Archive> void serialize(Archive& archive) {
        archive(cereal::make_nvp("position", pos),
                CEREAL_NVP(near),
                CEREAL_NVP(far),
                CEREAL_NVP(fov),
                CEREAL_NVP(aspect));
    }
} Camera;

// Light structure to hold information about a light
typedef struct light_t {
    Position pos;
    Color color;
    decimal atten;

    // Serialization function
    template<class Archive> void serialize(Archive& archive) {
        archive(cereal::make_nvp("position", pos),
                cereal::make_nvp("color", color),
                CEREAL_NVP(atten));
    }
} Light;

// Skin structure to hold information about the skin
typedef struct skin_t {
    Color diffuse, specular, ambient;
    decimal phong;

    // Serialization function
    template<class Archive> void serialize(Archive& archive) {
        archive(CEREAL_NVP(diffuse),
                CEREAL_NVP(specular),
                CEREAL_NVP(ambient),
                CEREAL_NVP(phong));
    }
} Skin;

// Hair layer structure to hold information about a layer of hair
typedef struct hair_t {
    Color diffuse, specular, ambient;
    decimal phong, density, radius, length, atten;

    // Serialization function
    template<class Archive> void serialize(Archive& archive) {
        archive(CEREAL_NVP(diffuse),
                CEREAL_NVP(specular),
                CEREAL_NVP(ambient),
                CEREAL_NVP(phong),
                CEREAL_NVP(density),
                CEREAL_NVP(radius),
                CEREAL_NVP(length),
                CEREAL_NVP(atten));
    }
} Hair;

// Sphere structure to hold information about a sphere
typedef struct sphere_t {
    // Spacial data
    Position pos;
    decimal radius;
    // Skin
    Skin skin_data;
    // Layers of hairs
    int n_hair_layers;
    Hair hair_layers[SPHERE_MAX_HAIR_LAYERS];

    // Serialization function
    template<class Archive> void serialize(Archive& archive) {
        archive(cereal::make_nvp("position", pos),
                CEREAL_NVP(radius),
                cereal::make_nvp("skin", skin_data),
                CEREAL_NVP(n_hair_layers));
        for (int i = 0; i < SPHERE_MAX_HAIR_LAYERS; i++) {
            archive(cereal::make_nvp("hair_layer" + std::to_string(i),
                                     hair_layers[i]));
        }
    }
} Sphere;

// Scene structure to hold information about a scene
typedef struct scene_t {
    Camera cam;
    Sphere s;
    int n_lights;
    Light lights[SCENE_MAX_LIGHTS];

    // Serialization function
    template<class Archive> void serialize(Archive& archive) {
        archive(cereal::make_nvp("camera", cam),
                cereal::make_nvp("sphere", s),
                CEREAL_NVP(n_lights));
        for (int i = 0; i < SCENE_MAX_LIGHTS; i++) {
            archive(cereal::make_nvp("light" + std::to_string(i),
                                     lights[i]));
        }
    }
} Scene;
