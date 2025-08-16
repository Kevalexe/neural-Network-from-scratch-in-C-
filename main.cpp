#include <iostream>
#include <cmath>
#include <cstdlib>
#include <ctime>
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

int main() {
    srand(time(NULL));

    // XOR dataset
    const int n_samples = 4;
    const int n_int = 2;      // two inputs for XOR
    const int n_hidden = 5;
    const int n_outputs = 1;  // one output (0 or 1)
    const double eta = 0.1;
    const int epochs = 5000;

    double X[n_samples][n_int] = {
        {0, 0},
        {0, 1},
        {1, 0},
        {1, 1}
    };

    double Y[n_samples][n_outputs] = {
        {0},
        {1},
        {1},
        {0}
    };

    // Weights and biases
    double w_input_hidden[n_hidden][n_int];
    double b_hidden[n_hidden];

    double w_output_hidden[n_outputs][n_hidden];
    double b_output[n_outputs];

    init_weights(&w_input_hidden[0][0], n_hidden * n_int);
    init_weights(b_hidden, n_hidden);
    init_weights(&w_output_hidden[0][0], n_outputs * n_hidden);
    init_weights(&b_output[0], n_outputs);

    for (int epoch = 0; epoch < epochs; epoch++) {
        double total_loss = 0.0;

        for (int sample = 0; sample < n_samples; sample++) {
            double* x = X[sample];
            double* y = Y[sample];

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

            double z_output[n_outputs];
            double a_output[n_outputs];
            for (int k = 0; k < n_outputs; k++) {
                z_output[k] = b_output[k];
                for (int s = 0; s < n_hidden; s++) {
                    z_output[k] += w_output_hidden[k][s] * a_hidden[s];
                }
                a_output[k] = sigmoid(z_output[k]);
            }

            // --- Loss ---
            double error = a_output[0] - y[0];
            total_loss += error * error;

            // --- Backpropagation ---
            double delta_output[n_outputs];
            delta_output[0] = error * sigmoidDerivative(z_output[0]);

            double delta_hidden[n_hidden];
            for (int i = 0; i < n_hidden; i++) {
                double sum_errors = delta_output[0] * w_output_hidden[0][i];
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
        }

        if (epoch % 500 == 0) {
            cout << "Epoch " << epoch
                 << " | Loss: " << total_loss << endl;
        }
    }

    // Test after training
    cout << "\nTesting trained network:" << endl;
    for (int sample = 0; sample < n_samples; sample++) {
        double* x = X[sample];

        double z_hidden[n_hidden];
        double a_hidden[n_hidden];
        for (int i = 0; i < n_hidden; i++) {
            z_hidden[i] = b_hidden[i];
            for (int j = 0; j < n_int; j++) {
                z_hidden[i] += w_input_hidden[i][j] * x[j];
            }
            a_hidden[i] = relu(z_hidden[i]);
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

        cout << "Input (" << x[0] << ", " << x[1] << ") -> "
             << a_output[0] << endl;
    }

    return 0;
}
