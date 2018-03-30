#pragma once

#include <string>

#include "cereal/archives/json.hpp"
#include "cereal/types/eigen.hpp"
#include "Eigen/Dense"

#define SPHERE_MAX_HAIR_LAYERS 5
#define SCENE_MAX_LIGHTS 5

// Camera structure to hold information about a camera
typedef struct camera_t {
    Eigen::Vector3d pos;
    double near, far, fov, aspect;

    // Serialization function
    template<class Archive> void serialize(Archive& archive) {
        archive(cereal::make_nvp("position", pos),
                CEREAL_NVP(near),
                CEREAL_NVP(far),
                CEREAL_NVP(fov),
                CEREAL_NVP(aspect));
    }
} camera;

// Light structure to hold information about a light
typedef struct light_t {
    Eigen::Vector3d pos, color;
    double atten;

    // Serialization function
    template<class Archive> void serialize(Archive& archive) {
        archive(cereal::make_nvp("position", pos),
                cereal::make_nvp("color", color),
                CEREAL_NVP(atten));
    }
} light;

// Skin structure to hold information about the skin
typedef struct skin_t {
    Eigen::Vector3d diffuse, specular, ambient;
    double phong;

    // Serialization function
    template<class Archive> void serialize(Archive& archive) {
        archive(CEREAL_NVP(diffuse),
                CEREAL_NVP(specular),
                CEREAL_NVP(ambient),
                CEREAL_NVP(phong));
    }
} skin;

// Hair layer structure to hold information about a layer of hair
typedef struct hair_t {
    Eigen::Vector3d diffuse, specular, ambient;
    double phong, density, radius, length, atten;

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
} hair;

// Sphere structure to hold information about a sphere
typedef struct sphere_t {
    // Spacial data
    Eigen::Vector3d pos;
    double radius;
    // Skin
    skin skin_data;
    // Layers of hairs
    int n_hair_layers;
    hair hair_layers[SPHERE_MAX_HAIR_LAYERS];

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
} sphere;

// Scene structure to hold information about a scene
typedef struct scene_t {
    camera cam;
    sphere s;
    int n_lights;
    light lights[SCENE_MAX_LIGHTS];

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
} scene;
