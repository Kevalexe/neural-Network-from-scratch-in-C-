#include <algorithm>
#include <iostream>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <vector>

using namespace std;

double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

double relu(double x) {
    return x > 0 ? x : 0;
}

double reluDerivative(double x) {
    return x > 0 ? 1 : 0;
}

double sigmoidDerivative(double x) {
    double s = sigmoid(x);
    return s * (1 - s);
}

void init_weights(double *weights, int size) {
    for (int i = 0; i < size; i++) {
        weights[i] = ((double) rand() / RAND_MAX) * 2 - 1;
    }
}

// cross-entropy loss
double cross_entropy(const vector<double>& y, const vector<double>& a) {
    double loss = 0.0;
    for (size_t i = 0; i < y.size(); i++) {
        loss += -(y[i] * log(a[i] + 1e-9) + (1 - y[i]) * log(1 - a[i] + 1e-9));
    }
    return loss;
}

// accuracy (argmax comparison)
bool correct_prediction(const vector<double>& y, const vector<double>& a) {
    int target = max_element(y.begin(), y.end()) - y.begin();
    int pred   = max_element(a.begin(), a.end()) - a.begin();
    return target == pred;
}

int main() {
    srand(time(NULL));

    const int n_int = 3;
    const int n_hidden = 5;
    const int n_outputs = 2;
    const double eta = 0.1;
    const int epochs = 1000;

    double w_input_hidden[n_hidden][n_int];
    double b_hidden[n_hidden];

    double w_output_hidden[n_outputs][n_hidden];
    double b_output[n_outputs];

    init_weights(&w_input_hidden[0][0], n_hidden * n_int);
    init_weights(b_hidden, n_hidden);
    init_weights(&w_output_hidden[0][0], n_outputs * n_hidden);
    init_weights(&b_output[0], n_outputs);

    // Example dataset: XOR style (just 1 sample here)
    double x[n_int] = {0.5, 0.1, 0.9};
    vector<double> y = {1.0, 0.0}; // one-hot target

    for (int epoch = 0; epoch < epochs; epoch++) {
        // --- Forward pass ---
        double z_hidden[n_hidden];
        double a_hidden[n_hidden];
        for (int i = 0; i < n_hidden; i++) {
            z_hidden[i] = b_hidden[i];
            for (int j = 0; j < n_int; j++) {
                z_hidden[i] += w_input_hidden[i][j] * x[j];
            }
            a_hidden[i] = relu(z_hidden[i]);
        }

        vector<double> z_output(n_outputs);
        vector<double> a_output(n_outputs);
        for (int k = 0; k < n_outputs; k++) {
            z_output[k] = b_output[k];
            for (int s = 0; s < n_hidden; s++) {
                z_output[k] += w_output_hidden[k][s] * a_hidden[s];
            }
            a_output[k] = sigmoid(z_output[k]);
        }

        // --- Loss + Accuracy ---
        double loss = cross_entropy(y, a_output);
        bool correct = correct_prediction(y, a_output);

        // --- Backpropagation ---
        double delta_output[n_outputs];
        for (int m = 0; m < n_outputs; m++) {
            double error = a_output[m] - y[m];
            delta_output[m] = error * sigmoidDerivative(z_output[m]);
        }

        double delta_hidden[n_hidden];
        for (int i = 0; i < n_hidden; i++) {
            double sum_errors = 0.0;
            for (int j = 0; j < n_outputs; j++) {
                sum_errors += delta_output[j] * w_output_hidden[j][i];
            }
            delta_hidden[i] = sum_errors * reluDerivative(z_hidden[i]);
        }

        // Update weights hidden → output
        for (int k = 0; k < n_outputs; k++) {
            for (int j = 0; j < n_hidden; j++) {
                w_output_hidden[k][j] -= eta * delta_output[k] * a_hidden[j];
            }
            b_output[k] -= eta * delta_output[k];
        }

        // Update weights input → hidden
        for (int j = 0; j < n_hidden; j++) {
            for (int i = 0; i < n_int; i++) {
                w_input_hidden[j][i] -= eta * delta_hidden[j] * x[i];
            }
            b_hidden[j] -= eta * delta_hidden[j];
        }

        // Print progress every 100 epochs
        if (epoch % 100 == 0) {
            cout << "Epoch " << epoch
                 << " | Loss: " << loss
                 << " | Accuracy: " << (correct ? 1 : 0)
                 << " | Outputs: ";
            for (double val : a_output) cout << val << " ";
            cout << endl;
        }
    }

    return 0;
}
