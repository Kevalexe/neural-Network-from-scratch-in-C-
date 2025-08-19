#include "Dataloader.h"
#include <fstream>
#include <sstream>
#include <iostream>

std::vector<Sample> loadIris(const std::string& filename) {
    std::vector<Sample> dataset;
    std::ifstream file(filename);

    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return dataset;
    }

    std::string line;
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string item;
        Sample sample;
        sample.features.reserve(4);

        // 4 numeric features
        for (int i = 0; i < 4; i++) {
            std::getline(ss, item, ',');
            sample.features.push_back(std::stod(item));
        }

        // class label → one-hot
        std::getline(ss, item, ',');
        if (item == "Iris-setosa")
            sample.label = {1,0,0};
        else if (item == "Iris-versicolor")
            sample.label = {0,1,0};
        else if (item == "Iris-virginica")
            sample.label = {0,0,1};

        dataset.push_back(sample);
    }

    return dataset;
}
