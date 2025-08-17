#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cstdlib>
#include <ctime>
#include "Dataloader.h"
using namespace std;

// Structure for one sample
struct Sample {
    vector<double> features; // 4 inputs
    vector<double> label;    // one-hot (3 outputs)
};

// Load CSV dataset
vector<Sample> loadIris(string filename) {
    vector<Sample> data;
    ifstream file(filename);
    string line;
    while (getline(file, line)) {
        stringstream ss(line);
        string value;
        vector<double> features;
        vector<double> label(3, 0);

        // Read 4 features
        for (int i = 0; i < 4; i++) {
            getline(ss, value, ',');
            features.push_back(stod(value));
        }
        // Read label
        getline(ss, value, ',');
        int cls = stoi(value);
        label[cls] = 1.0;

        data.push_back({features, label});
    }
    return data;
}