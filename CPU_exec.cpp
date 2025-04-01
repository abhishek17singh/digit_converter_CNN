#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <sycl/sycl.hpp>
#include <limits>

// Function to reverse integer bytes
int reverse_int(int n) {
    return ((n & 0xFF) << 24) | 
           ((n & 0xFF00) << 8) | 
           ((n & 0xFF0000) >> 8) | 
           ((n & 0xFF000000) >> 24);
}

// Function to load MNIST dataset (images and labels)
void load_mnist_images(const std::string &image_file, const std::string &label_file, 
    std::vector<std::vector<float>> &images, std::vector<int> &labels) {
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

    magic_number = reverse_int(magic_number);
    num_images = reverse_int(num_images);
    num_rows = reverse_int(num_rows);
    num_cols = reverse_int(num_cols);

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

    std::ifstream lbl_file(label_file, std::ios::binary);
    if (!lbl_file.is_open()) {
        std::cerr << "Failed to open label file!" << std::endl;
        exit(1);
    }

    lbl_file.read(reinterpret_cast<char*>(&magic_number), 4);
    lbl_file.read(reinterpret_cast<char*>(&num_images), 4);

    magic_number = reverse_int(magic_number);
    num_images = reverse_int(num_images);

    for (int i = 0; i < num_images; ++i) {
        unsigned char label;
        lbl_file.read(reinterpret_cast<char*>(&label), 1);
        labels.push_back(static_cast<int>(label));
    }
    lbl_file.close();
}

// SYCL kernel for matrix multiplication
void matrix_multiply(sycl::queue &q, const std::vector<float> &input, const std::vector<float> &weights, 
                     std::vector<float> &output, size_t input_size, size_t output_size) {
    sycl::buffer<float> input_buf(input.data(), sycl::range<1>(input_size));
    sycl::buffer<float> weights_buf(weights.data(), sycl::range<1>(input_size * output_size));
    sycl::buffer<float> output_buf(output.data(), sycl::range<1>(output_size));

    q.submit([&](sycl::handler &cgh) {
        auto input_acc = input_buf.get_access<sycl::access::mode::read>(cgh);
        auto weights_acc = weights_buf.get_access<sycl::access::mode::read>(cgh);
        auto output_acc = output_buf.get_access<sycl::access::mode::write>(cgh);

        cgh.parallel_for(sycl::range<1>(output_size), [=](sycl::id<1> idx) {
            float result = 0.0f;
            for (size_t i = 0; i < input_size; ++i) {
                result += input_acc[i] * weights_acc[idx[0] * input_size + i];
            }
            output_acc[idx] = result;
        });
    }).wait();
}

// ReLU activation function on GPU
void relu_activation(sycl::queue &q, std::vector<float> &layer, size_t layer_size) {
    sycl::buffer<float> layer_buf(layer.data(), sycl::range<1>(layer_size));

    q.submit([&](sycl::handler &cgh) {
        auto layer_acc = layer_buf.get_access<sycl::access::mode::read_write>(cgh);

        cgh.parallel_for(sycl::range<1>(layer_size), [=](sycl::id<1> idx) {
            layer_acc[idx] = sycl::fmax(layer_acc[idx], 0.0f);  // ReLU
        });
    }).wait();
}

// Softmax activation to normalize output layer values
void softmax_activation(sycl::queue &q, std::vector<float> &output_layer, size_t output_size) {
    sycl::buffer<float> output_buf(output_layer.data(), sycl::range<1>(output_size));

    // Find max value in output to prevent overflow during exp calculation
    float max_val = -std::numeric_limits<float>::infinity();
    for (size_t i = 0; i < output_size; ++i) {
        max_val = std::max(max_val, output_layer[i]);
    }

    q.submit([&](sycl::handler &cgh) {
        auto output_acc = output_buf.get_access<sycl::access::mode::read_write>(cgh);

        cgh.parallel_for(sycl::range<1>(output_size), [=](sycl::id<1> idx) {
            float exp_val = sycl::exp(output_acc[idx] - max_val);
            output_acc[idx] = exp_val;
        });
    }).wait();

    // Now perform the manual sum of the exponentiated values for softmax
    float sum = 0.0f;
    for (size_t i = 0; i < output_size; ++i) {
        sum += output_layer[i];
    }

    // Normalize the output values by dividing each by the sum
    for (size_t i = 0; i < output_size; ++i) {
        output_layer[i] /= sum;
    }
}

// Function to train the neural network
void train_network(sycl::queue &q, std::vector<std::vector<float>>& images, std::vector<int>& labels,
    std::vector<float>& hidden_weights, std::vector<float>& hidden_biases,
    std::vector<float>& output_weights, std::vector<float>& output_biases) {
    const int epochs = 10;
    const float learning_rate = 0.01f;

    for (int epoch = 0; epoch < epochs; ++epoch) {
        for (int i = 0; i < images.size(); ++i) {
            // Forward pass
            std::vector<float> hidden_layer(128, 0.0f);
            std::vector<float> output_layer(10, 0.0f);

            matrix_multiply(q, images[i], hidden_weights, hidden_layer, 784, 128);
            relu_activation(q, hidden_layer, 128);
            matrix_multiply(q, hidden_layer, output_weights, output_layer, 128, 10);
            softmax_activation(q, output_layer, 10);

            // Backward pass (simplified for demonstration)
            // Calculate output error gradients
            std::vector<float> output_errors(10, 0.0f);
            for (int j = 0; j < 10; ++j) {
                output_errors[j] = output_layer[j] - (labels[i] == j ? 1.0f : 0.0f);
            }

            // Update weights and biases
            for (int j = 0; j < 128; ++j) {
                for (int k = 0; k < 10; ++k) {
                    output_weights[j * 10 + k] -= learning_rate * hidden_layer[j] * output_errors[k];
                }
            }

            for (int j = 0; j < 10; ++j) {
                output_biases[j] -= learning_rate * output_errors[j];
            }

            for (int j = 0; j < 784; ++j) {
                for (int k = 0; k < 128; ++k) {
                    hidden_weights[j * 128 + k] -= learning_rate * images[i][j] * hidden_layer[k];
                }
            }

            for (int j = 0; j < 128; ++j) {
                hidden_biases[j] -= learning_rate * hidden_layer[j];
            }
        }
    }
}

// Function to classify the digit using the neural network (GPU version)
int classify_digit(sycl::queue &q, const std::vector<float>& image, 
    const std::vector<float>& hidden_weights, const std::vector<float>& hidden_biases,
    const std::vector<float>& output_weights, const std::vector<float>& output_biases) {
    const size_t input_size = 784;  // 28x28 pixels
    const size_t hidden_size = 128;
    const size_t output_size = 10;

    std::vector<float> hidden_layer(hidden_size, 0.0f);
    std::vector<float> output_layer(output_size, 0.0f);

    // Matrix multiplication for hidden layer
    matrix_multiply(q, image, hidden_weights, hidden_layer, input_size, hidden_size);
    
    // ReLU activation for hidden layer
    relu_activation(q, hidden_layer, hidden_size);

    // Matrix multiplication for output layer
    matrix_multiply(q, hidden_layer, output_weights, output_layer, hidden_size, output_size);

    // Softmax activation for output layer
    softmax_activation(q, output_layer, output_size);

    // Find the predicted digit (max output value)
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

    // Initialize weights and biases
    const size_t input_size = 784;
    const size_t hidden_size = 128;
    const size_t output_size = 10;

    std::vector<float> hidden_weights(input_size * hidden_size, 0.01f);
    std::vector<float> hidden_biases(hidden_size, 0.1f);
    std::vector<float> output_weights(hidden_size * output_size, 0.01f);
    std::vector<float> output_biases(output_size, 0.1f);

    // Create a SYCL queue using the default GPU device
    auto platforms = sycl::platform::get_platforms();
    auto gpu_devices = platforms[0].get_devices(sycl::info::device_type::gpu);
    if (!gpu_devices.empty()) {
        sycl::queue q(gpu_devices[0]);
        
        // Train the network
        train_network(q, images, labels, hidden_weights, hidden_biases, output_weights, output_biases);

        // Select a test image (first image in dataset for simplicity)
        std::vector<float> test_image = images[0];

        // Measure execution time using chrono
        auto start = std::chrono::high_resolution_clock::now();
        int predicted_digit = classify_digit(q, test_image, hidden_weights, hidden_biases, output_weights, output_biases);  
        auto end = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double> duration = end - start;
        
        // Output the results
        std::cout << "Predicted Digit: " << predicted_digit << std::endl;
        std::cout << "Execution Time (GPU): " << duration.count() << " seconds" << std::endl;
    } else {
        std::cerr << "No GPU device found!" << std::endl;
    }

    return 0;
}
