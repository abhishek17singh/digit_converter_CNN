#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <sycl/sycl.hpp>

// Function to load MNIST dataset
void load_mnist_images(const std::string &image_file, const std::string &label_file, 
    std::vector<std::vector<float>> &images, std::vector<int> &labels) {
    // Load images
    std::ifstream img_file(image_file, std::ios::binary);
    if (!img_file.is_open()) {
        std::cerr << "Failed to open image file!" << std::endl;
        exit(1);
    }

    int magic_number, num_images, num_rows, num_cols;
    img_file.read(reinterpret_cast<char*>(&magic_number), 4);
    img_file.read(reinterpret_cast<char*>(&num_images), 4);
    img_file.read(reinterpret_cast<char*>(&num_rows), 4);
    img_file.read(reinterpret_cast<char*>(&num_cols), 4);

    for (int i = 0; i < num_images; ++i) {
        std::vector<float> image(num_rows * num_cols);
        for (int j = 0; j < num_rows * num_cols; ++j) {
            unsigned char pixel;
            img_file.read(reinterpret_cast<char*>(&pixel), 1);
            image[j] = static_cast<float>(pixel) / 255.0f;
        }
        images.push_back(image);
    }
    img_file.close();

    // Load labels
    std::ifstream lbl_file(label_file, std::ios::binary);
    if (!lbl_file.is_open()) {
        std::cerr << "Failed to open label file!" << std::endl;
        exit(1);
    }

    lbl_file.read(reinterpret_cast<char*>(&magic_number), 4);
    lbl_file.read(reinterpret_cast<char*>(&num_images), 4);

    for (int i = 0; i < num_images; ++i) {
        unsigned char label;
        lbl_file.read(reinterpret_cast<char*>(&label), 1);
        labels.push_back(static_cast<int>(label));
    }
    lbl_file.close();
}

// Simple fully connected neural network for digit classification
int classify_digit(const std::vector<float>& image) {
    // Simple feedforward network with one hidden layer
    // Input layer (784 neurons for 28x28 pixels), hidden layer (128 neurons), output layer (10 neurons for digits 0-9)
    std::vector<float> hidden_layer(128, 0.0f);
    std::vector<float> output_layer(10, 0.0f);
    
    // Random weights and biases (in practice, use trained weights)
    std::vector<float> hidden_weights(784 * 128, 0.01f);
    std::vector<float> output_weights(128 * 10, 0.01f);
    std::vector<float> hidden_biases(128, 0.0f);
    std::vector<float> output_biases(10, 0.0f);

    // Calculate hidden layer values (ReLU activation)
    for (int i = 0; i < 128; ++i) {
        for (int j = 0; j < 784; ++j) {
            hidden_layer[i] += image[j] * hidden_weights[i * 784 + j];
        }
        hidden_layer[i] += hidden_biases[i];
        hidden_layer[i] = std::max(0.0f, hidden_layer[i]);  // ReLU activation
    }

    // Calculate output layer values (Softmax activation)
    for (int i = 0; i < 10; ++i) {
        for (int j = 0; j < 128; ++j) {
            output_layer[i] += hidden_layer[j] * output_weights[i * 128 + j];
        }
        output_layer[i] += output_biases[i];
    }

    // Find the digit with the highest probability
    int predicted_digit = 0;
    float max_prob = output_layer[0];
    for (int i = 1; i < 10; ++i) {
        if (output_layer[i] > max_prob) {
            max_prob = output_layer[i];
            predicted_digit = i;
        }
    }

    return predicted_digit;
}

int main() {
    // Load MNIST data (images and labels)
    std::vector<std::vector<float>> images;
    std::vector<int> labels;
    load_mnist_images("train-images.idx3-ubyte", "train-labels.idx1-ubyte", images, labels);

    // Select a test image (first image in dataset for simplicity)
    std::vector<float> test_image = images[0];

    // Measure execution time using chrono
    auto start = std::chrono::high_resolution_clock::now();
    int predicted_digit = classify_digit(test_image);  // Perform inference
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> duration = end - start;
    
    // Output the results
    std::cout << "Predicted Digit: " << predicted_digit << std::endl;
    std::cout << "Execution Time (CPU): " << duration.count() << " seconds" << std::endl;

    return 0;
}