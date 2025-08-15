#include <iostream>
#include <cmath>
#include <cstdlib>
#include <ctime>
using namespace std;

double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

double relu (double x) {
    return x > 0 ? x : 0;
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

    double x[n_int] = {0.5, 0.1, 0.9};
    double y[n_outputs] = {1.0, 0.0}; // target

    for (int epoch = 0; epoch < epochs; epoch++) {
        // --- Forward pass ---
        double z_hidden[n_hidden];
        double a_hidden[n_hidden];
        for (int i = 0; i < n_hidden; i++) {
            z_hidden[i] = b_hidden[i];
            for (int j = 0; j < n_int; j++) {
                z_hidden[i] += w_input_hidden[i][j] * x[j];
            }
            a_hidden[i] = sigmoid(z_hidden[i]);
        }

        double z_output[n_outputs];
        double a_output[n_outputs];
        for (int k = 0; k < n_outputs; k++) {
            z_output[k] = b_output[k];
            for (int s = 0; s < n_hidden; s++) {
                z_output[k] += w_output_hidden[k][s] * a_hidden[s];
            }
            a_output[k] = sigmoid(z_output[k]);
        }

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
            delta_hidden[i] = sum_errors * sigmoidDerivative(z_hidden[i]);
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
            cout << "Epoch " << epoch << ": ";
            for (int l = 0; l < n_outputs; l++) {
                cout << a_output[l] << " ";
            }
            cout << endl;
        }
    }

    return 0;
}
