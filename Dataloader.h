#ifndef DATALOADER_H
#define DATALOADER_H

#include <vector>
#include <string>

// Each sample = input features + one-hot label
struct Sample {
    std::vector<double> features;
    std::vector<double> label;
};

// Function to load Iris dataset
std::vector<Sample> loadIris(const std::string& filename);

#endif
