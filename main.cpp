#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <algorithm>
#include <fstream>
#include <sstream>
using namespace std;

// ---------- NN Utils ----------
double sigmoid(double x) { return 1.0 / (1.0 + exp(-x)); }
double sigmoidDerivative(double x) { double s = sigmoid(x); return s * (1 - s); }
double relu(double x) { return x > 0 ? x : 0; }
double reluDerivative(double x) { return x > 0 ? 1 : 0; }

void init_weights(double *weights, int size) {
    for (int i = 0; i < size; i++)
        weights[i] = ((double) rand() / RAND_MAX) * 2 - 1;
}

double cross_entropy(const vector<double>& y, const vector<double>& a) {
    double loss = 0.0;
    for (size_t i = 0; i < y.size(); i++)
        loss += -(y[i] * log(a[i] + 1e-9) + (1 - y[i]) * log(1 - a[i] + 1e-9));
    return loss;
}

bool correct_prediction(const vector<double>& y, const vector<double>& a) {
    int target = max_element(y.begin(), y.end()) - y.begin();
    int pred   = max_element(a.begin(), a.end()) - a.begin();
    return target == pred;
}

// ---------- Dataset Struct ----------
struct Sample {
    vector<double> features;
    vector<double> label; // one-hot
};

// ---------- Load Iris CSV ----------
vector<Sample> loadIris(const string& filename) {
    vector<Sample> dataset;
    ifstream file(filename);
    if (!file.is_open()) {
        cerr << "Error: could not open " << filename << endl;
        return dataset;
    }

    string line;
    while (getline(file, line)) {
        if (line.empty()) continue; // skip blanks
        stringstream ss(line);
        string item;
        Sample sample;
        sample.features.reserve(4);

        // try parse first feature (skip header if fails)
        getline(ss, item, ',');
        try {
            sample.features.push_back(stod(item));
        } catch (...) {
            continue;
        }

        // 3 more features
        for (int i = 1; i < 4; i++) {
            getline(ss, item, ',');
            sample.features.push_back(stod(item));
        }

        // class → one-hot
        getline(ss, item, ',');
        if (item == "Iris-setosa")
            sample.label = {1,0,0};
        else if (item == "Iris-versicolor")
            sample.label = {0,1,0};
        else if (item == "Iris-virginica")
            sample.label = {0,0,1};
        else
            continue;

        dataset.push_back(sample);
    }
    return dataset;
}

// ---------- MAIN ----------
int main() {
    srand(time(NULL));

    const int n_in = 4;
    const int n_hidden = 8;
    const int n_out = 3;
    const double eta = 0.05;
    const int epochs = 500;
    const int batch_size = 16;

    // load iris
    vector<Sample> dataset = loadIris("iris.csv");
    if (dataset.empty()) {
        cerr << "Dataset not loaded!" << endl;
        return 1;
    }
    random_shuffle(dataset.begin(), dataset.end());

    // train/test split (80/20)
    int train_size = dataset.size() * 0.8;
    vector<Sample> train(dataset.begin(), dataset.begin() + train_size);
    vector<Sample> test(dataset.begin() + train_size, dataset.end());

    cout << "Train size: " << train.size() << ", Test size: " << test.size() << endl;

    // weights
    double w_input_hidden[n_hidden][n_in];
    double b_hidden[n_hidden];
    double w_hidden_output[n_out][n_hidden];
    double b_output[n_out];

    init_weights(&w_input_hidden[0][0], n_hidden * n_in);
    init_weights(b_hidden, n_hidden);
    init_weights(&w_hidden_output[0][0], n_out * n_hidden);
    init_weights(b_output, n_out);

    // ---------- Training ----------
    for (int epoch = 0; epoch < epochs; epoch++) {
        double total_loss = 0.0;
        int correct = 0;

        // mini-batches
        for (size_t batch_start = 0; batch_start < train.size(); batch_start += batch_size) {
            size_t batch_end = min(batch_start + batch_size, train.size());

            // gradients
            double dW_hidden_output[n_out][n_hidden] = {0};
            double dB_output[n_out] = {0};
            double dW_input_hidden[n_hidden][n_in] = {0};
            double dB_hidden[n_hidden] = {0};

            for (size_t s = batch_start; s < batch_end; s++) {
                auto x = train[s].features;
                auto y = train[s].label;

                // forward
                double z_hidden[n_hidden], a_hidden[n_hidden];
                for (int i = 0; i < n_hidden; i++) {
                    z_hidden[i] = b_hidden[i];
                    for (int j = 0; j < n_in; j++)
                        z_hidden[i] += w_input_hidden[i][j] * x[j];
                    a_hidden[i] = relu(z_hidden[i]);
                }

                double z_output[n_out], a_output[n_out];
                for (int k = 0; k < n_out; k++) {
                    z_output[k] = b_output[k];
                    for (int j = 0; j < n_hidden; j++)
                        z_output[k] += w_hidden_output[k][j] * a_hidden[j];
                    a_output[k] = sigmoid(z_output[k]);
                }

                total_loss += cross_entropy(y, vector<double>(a_output, a_output+n_out));
                if (correct_prediction(y, vector<double>(a_output, a_output+n_out))) correct++;

                // backprop
                double delta_output[n_out];
                for (int k = 0; k < n_out; k++)
                    delta_output[k] = (a_output[k] - y[k]) * sigmoidDerivative(z_output[k]);

                double delta_hidden[n_hidden];
                for (int i = 0; i < n_hidden; i++) {
                    double sum_err = 0.0;
                    for (int k = 0; k < n_out; k++)
                        sum_err += delta_output[k] * w_hidden_output[k][i];
                    delta_hidden[i] = sum_err * reluDerivative(z_hidden[i]);
                }

                // accumulate grads
                for (int k = 0; k < n_out; k++) {
                    for (int j = 0; j < n_hidden; j++)
                        dW_hidden_output[k][j] += delta_output[k] * a_hidden[j];
                    dB_output[k] += delta_output[k];
                }
                for (int j = 0; j < n_hidden; j++) {
                    for (int i = 0; i < n_in; i++)
                        dW_input_hidden[j][i] += delta_hidden[j] * x[i];
                    dB_hidden[j] += delta_hidden[j];
                }
            }

            // update weights
            size_t batch_size_real = batch_end - batch_start;
            for (int k = 0; k < n_out; k++) {
                for (int j = 0; j < n_hidden; j++)
                    w_hidden_output[k][j] -= eta * (dW_hidden_output[k][j] / batch_size_real);
                b_output[k] -= eta * (dB_output[k] / batch_size_real);
            }
            for (int j = 0; j < n_hidden; j++) {
                for (int i = 0; i < n_in; i++)
                    w_input_hidden[j][i] -= eta * (dW_input_hidden[j][i] / batch_size_real);
                b_hidden[j] -= eta * (dB_hidden[j] / batch_size_real);
            }
        }

        if (epoch % 50 == 0)
            cout << "Epoch " << epoch
                 << " | Loss: " << total_loss / train.size()
                 << " | Train Acc: " << (double)correct / train.size() << endl;
    }

    // ---------- Testing ----------
    int correct_test = 0;
    for (auto &s : test) {
        auto x = s.features;
        auto y = s.label;

        double z_hidden[n_hidden], a_hidden[n_hidden];
        for (int i = 0; i < n_hidden; i++) {
            z_hidden[i] = b_hidden[i];
            for (int j = 0; j < n_in; j++)
                z_hidden[i] += w_input_hidden[i][j] * x[j];
            a_hidden[i] = relu(z_hidden[i]);
        }

        double z_output[n_out], a_output[n_out];
        for (int k = 0; k < n_out; k++) {
            z_output[k] = b_output[k];
            for (int j = 0; j < n_hidden; j++)
                z_output[k] += w_hidden_output[k][j] * a_hidden[j];
            a_output[k] = sigmoid(z_output[k]);
        }

        if (correct_prediction(y, vector<double>(a_output, a_output+n_out)))
            correct_test++;
    }
    cout << "Test Accuracy: " << (double)correct_test / test.size() << endl;

    return 0;
}
