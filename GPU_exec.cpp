#include <CL/sycl.hpp>
#include <iostream>
#include <vector>
#include <fstream>
#include <cmath>
#include <chrono>

using namespace sycl;

// Function to reverse integer bytes
int reverse_int(int n) {
    return ((n & 0xFF) << 24) | ((n & 0xFF00) << 8) | ((n & 0xFF0000) >> 8) | ((n & 0xFF000000) >> 24);
}

// Load MNIST dataset
bool load_mnist_images(const std::string &image_file, const std::string &label_file,
    std::vector<std::vector<float>> &images, std::vector<int> &labels) {

    std::ifstream img_file(image_file, std::ios::binary);
    if (!img_file.is_open()) return false;

    int magic_number, num_images, num_rows, num_cols;
    img_file.read(reinterpret_cast<char*>(&magic_number), 4);
    img_file.read(reinterpret_cast<char*>(&num_images), 4);
    img_file.read(reinterpret_cast<char*>(&num_rows), 4);
    img_file.read(reinterpret_cast<char*>(&num_cols), 4);

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

    num_images = reverse_int(num_images);

    for (int i = 0; i < num_images; ++i) {
        unsigned char label;
        lbl_file.read(reinterpret_cast<char*>(&label), 1);
        labels.push_back(static_cast<int>(label));
    }
    lbl_file.close();
    return true;
}

// GPU-accelerated function to classify an image
int classify_digit(const std::vector<float>& image, const std::vector<float>& hidden_weights,
                   const std::vector<float>& hidden_biases, const std::vector<float>& output_weights,
                   const std::vector<float>& output_biases) {
    const int input_size = 784, hidden_size = 256, output_size = 10;
    std::vector<float> hidden_layer(hidden_size, 0.0f);
    std::vector<float> output_layer(output_size, 0.0f);

    queue q(gpu_selector_v);
    {
        buffer<float, 1> img_buf(image.data(), range<1>(input_size));
        buffer<float, 1> hidden_w_buf(hidden_weights.data(), range<1>(input_size * hidden_size));
        buffer<float, 1> hidden_b_buf(hidden_biases.data(), range<1>(hidden_size));
        buffer<float, 1> output_w_buf(output_weights.data(), range<1>(hidden_size * output_size));
        buffer<float, 1> output_b_buf(output_biases.data(), range<1>(output_size));
        buffer<float, 1> hidden_layer_buf(hidden_layer.data(), range<1>(hidden_size));
        buffer<float, 1> output_layer_buf(output_layer.data(), range<1>(output_size));

        q.submit([&](handler &h) {
            accessor img_acc(img_buf, h, read_only);
            accessor hidden_w_acc(hidden_w_buf, h, read_only);
            accessor hidden_b_acc(hidden_b_buf, h, read_only);
            accessor hidden_layer_acc(hidden_layer_buf, h, write_only);

            h.parallel_for(range<1>(hidden_size), [=](id<1> i) {
                float sum = hidden_b_acc[i];
                for (int j = 0; j < input_size; ++j) {
                    sum += img_acc[j] * hidden_w_acc[j * hidden_size + i];
                }
                hidden_layer_acc[i] = sum > 0 ? sum : 0.01f * sum;  // Leaky ReLU
            });
        });

        q.submit([&](handler &h) {
            accessor hidden_layer_acc(hidden_layer_buf, h, read_only);
            accessor output_w_acc(output_w_buf, h, read_only);
            accessor output_b_acc(output_b_buf, h, read_only);
            accessor output_layer_acc(output_layer_buf, h, write_only);

            h.parallel_for(range<1>(output_size), [=](id<1> i) {
                float sum = output_b_acc[i];
                for (int j = 0; j < hidden_size; ++j) {
                    sum += hidden_layer_acc[j] * output_w_acc[j * output_size + i];
                }
                output_layer_acc[i] = sum;
            });
        });
    }

    float max_val = *std::max_element(output_layer.begin(), output_layer.end());
    return std::distance(output_layer.begin(), std::max_element(output_layer.begin(), output_layer.end()));
}

int main() {
    std::vector<std::vector<float>> images;
    std::vector<int> labels;

    if (!load_mnist_images("train-images.idx3-ubyte", "train-labels.idx1-ubyte", images, labels)) {
        std::cerr << "Error loading MNIST data." << std::endl;
        return 1;
    }

    const int input_size = 784, hidden_size = 256, output_size = 10;
    std::vector<float> hidden_weights(input_size * hidden_size, 0.01f);
    std::vector<float> hidden_biases(hidden_size, 0.1f);
    std::vector<float> output_weights(hidden_size * output_size, 0.01f);
    std::vector<float> output_biases(output_size, 0.1f);

    int test_index = 100;
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
