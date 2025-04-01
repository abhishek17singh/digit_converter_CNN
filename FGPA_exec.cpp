#include <CL/sycl.hpp>
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <chrono>

using namespace sycl;

// Function to reverse integer bytes (for MNIST format)
int reverse_int(int n) {
    return ((n & 0xFF) << 24) | 
           ((n & 0xFF00) << 8) | 
           ((n & 0xFF0000) >> 8) | 
           ((n & 0xFF000000) >> 24);
}

// Function to load MNIST dataset
bool load_mnist_images(const std::string &image_file, const std::string &label_file, 
    std::vector<std::vector<float>> &images, std::vector<int> &labels) {
    
    std::ifstream img_file(image_file, std::ios::binary);
    if (!img_file.is_open()) return false;

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
    if (!lbl_file.is_open()) return false;

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

    return true;
}

// Leaky ReLU activation function
float leaky_relu(float x) {
    return x > 0.0f ? x : 0.01f * x;
}

// Classify a single digit using FPGA
int classify_digit(const std::vector<float>& image, 
    std::vector<float>& hidden_weights, std::vector<float>& hidden_biases,
    std::vector<float>& output_weights, std::vector<float>& output_biases) {
    
    const int input_size = 784;
    const int hidden_size = 256;
    const int output_size = 10;

    std::vector<float> hidden_layer(hidden_size, 0.0f);
    std::vector<float> output_layer(output_size, 0.0f);

    queue q(sycl::ext::intel::fpga_selector_v);

    buffer<float, 1> img_buf(image.data(), range<1>(input_size));
    buffer<float, 1> hidden_w_buf(hidden_weights.data(), range<1>(input_size * hidden_size));
    buffer<float, 1> hidden_b_buf(hidden_biases.data(), range<1>(hidden_size));
    buffer<float, 1> output_w_buf(output_weights.data(), range<1>(hidden_size * output_size));
    buffer<float, 1> output_b_buf(output_biases.data(), range<1>(output_size));
    buffer<float, 1> hidden_buf(hidden_layer.data(), range<1>(hidden_size));
    buffer<float, 1> output_buf(output_layer.data(), range<1>(output_size));

    q.submit([&](handler& h) {
        auto img_acc = img_buf.get_access<access::mode::read>(h);
        auto hidden_w_acc = hidden_w_buf.get_access<access::mode::read>(h);
        auto hidden_b_acc = hidden_b_buf.get_access<access::mode::read>(h);
        auto hidden_acc = hidden_buf.get_access<access::mode::write>(h);

        h.parallel_for(range<1>(hidden_size), [=](id<1> i) {
            float sum = hidden_b_acc[i];
            for (int j = 0; j < input_size; ++j) {
                sum += img_acc[j] * hidden_w_acc[j * hidden_size + i];
            }
            hidden_acc[i] = leaky_relu(sum);
        });
    }).wait();

    q.submit([&](handler& h) {
        auto hidden_acc = hidden_buf.get_access<access::mode::read>(h);
        auto output_w_acc = output_w_buf.get_access<access::mode::read>(h);
        auto output_b_acc = output_b_buf.get_access<access::mode::read>(h);
        auto output_acc = output_buf.get_access<access::mode::write>(h);

        h.parallel_for(range<1>(output_size), [=](id<1> i) {
            float sum = output_b_acc[i];
            for (int j = 0; j < hidden_size; ++j) {
                sum += hidden_acc[j] * output_w_acc[j * output_size + i];
            }
            output_acc[i] = sum;
        });
    }).wait();

    auto output_host = output_buf.get_access<access::mode::read>();
    return std::distance(output_host.begin(), std::max_element(output_host.begin(), output_host.end()));
}

int main() {
    std::vector<std::vector<float>> images;
    std::vector<int> labels;

    if (!load_mnist_images("train-images.idx3-ubyte", "train-labels.idx1-ubyte", images, labels)) {
        std::cerr << "Error loading MNIST data." << std::endl;
        return 1;
    }

    std::vector<float> hidden_weights(784 * 256, 0.01f);
    std::vector<float> hidden_biases(256, 0.1f);
    std::vector<float> output_weights(256 * 10, 0.01f);
    std::vector<float> output_biases(10, 0.1f);

    const int test_index = 100;
    auto start = std::chrono::high_resolution_clock::now();
    int predicted = classify_digit(images[test_index], hidden_weights, hidden_biases, output_weights, output_biases);
    auto end = std::chrono::high_resolution_clock::now();

    std::cout << "\nResults:" 
              << "\nActual Label: " << labels[test_index] 
              << "\nPredicted Digit: " << predicted 
              << "\nExecution Time: " 
              << std::chrono::duration<double>(end - start).count() 
              << " seconds\n";

    return 0;
}
