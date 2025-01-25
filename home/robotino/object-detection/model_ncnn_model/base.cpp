#include <iostream>
#include "net.h"
#include <cmath>

float sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}


int main() {
    ncnn::Net yolov8;
    yolov8.load_param("model.ncnn.param");
    yolov8.load_model("model.ncnn.bin");

    ncnn::Mat in(640, 640, 3); // Create a simple 640x640x3 input Mat (all zeros)

    ncnn::Extractor ex = yolov8.create_extractor();
    ex.input("in0", in); // Use the correct input name (verify with Netron)

    ncnn::Mat out;
    ex.extract("out0", out); // Use the correct output name (verify with Netron)
    std::cout << out << std::endl;
    // std::cout << "Output shape: " << out.w << " " << out.h << " " << out.c << std::endl;

    for (int i = 0; i < out.h; i++) {
        float* values = out.row(i);
        std::cout << "Output Row " << i << ": ";
        for (int j = 0; j < out.w; j++) {
            std::cout << values[j] << " ";
        }
        std::cout << std::endl;
    }

    return 0;
}