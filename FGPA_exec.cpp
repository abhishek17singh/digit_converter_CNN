#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>  // FPGA-specific extensions

// Function to load MNIST dataset (same as before)
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

    for (int i = 0; i < num_images; ++i) {
        unsigned char label;
        lbl_file.read(reinterpret_cast<char*>(&label), 1);
        labels.push_back(static_cast<int>(label));
    }
    lbl_file.close();
}

// SYCL kernel for matrix multiplication (to compute the dot product for hidden layers and output layers)
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

// ReLU activation function on FPGA (to process hidden layer values)
void relu_activation(sycl::queue &q, std::vector<float> &layer, size_t layer_size) {
    sycl::buffer<float> layer_buf(layer.data(), sycl::range<1>(layer_size));

    q.submit([&](sycl::handler &cgh) {
        auto layer_acc = layer_buf.get_access<sycl::access::mode::read_write>(cgh);

        cgh.parallel_for(sycl::range<1>(layer_size), [=](sycl::id<1> idx) {
            layer_acc[idx] = sycl::fmax(layer_acc[idx], 0.0f);  // ReLU
        });
    }).wait();
}

// Function to classify the digit using the neural network (FPGA version)
int classify_digit(sycl::queue &q, const std::vector<float>& image) {
    // Define layer sizes and random weights (in practice, use trained weights)
    const size_t input_size = 784;  // 28x28 pixels
    const size_t hidden_size = 128;
    const size_t output_size = 10;

    std::vector<float> hidden_layer(hidden_size, 0.0f);
    std::vector<float> output_layer(output_size, 0.0f);

    std::vector<float> hidden_weights(input_size * hidden_size, 0.01f);
    std::vector<float> output_weights(hidden_size * output_size, 0.01f);

    // Matrix multiplication for hidden layer
    matrix_multiply(q, image, hidden_weights, hidden_layer, input_size, hidden_size);
    
    // ReLU activation for hidden layer
    relu_activation(q, hidden_layer, hidden_size);

    // Matrix multiplication for output layer
    matrix_multiply(q, hidden_layer, output_weights, output_layer, hidden_size, output_size);

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

    // Select a test image (first image in dataset for simplicity)
    std::vector<float> test_image = images[0];

    // Create a SYCL queue to run on the FPGA (or CPU, depending on the device available)
    sycl::queue q(sycl::ext::intel::fpga_selector{});

    // Measure execution time using chrono
    auto start = std::chrono::high_resolution_clock::now();
    int predicted_digit = classify_digit(q, test_image);  // Perform inference on FPGA
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> duration = end - start;
    
    // Output the results
    std::cout << "Predicted Digit: " << predicted_digit << std::endl;
    std::cout << "Execution Time (FPGA): " << duration.count() << " seconds" << std::endl;

    return 0;
}