#pragma once
#include <vector>
#include <string>

struct Sample {
    std::vector<double> features;
    std::vector<double> label;
};

std::vector<Sample> loadIris(const std::string& filename);