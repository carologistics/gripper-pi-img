#include <iostream>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/dnn.hpp> // Important: Include dnn.hpp
#include "net.h"

struct Object {
    cv::Rect rect;
    int label;
    float prob;
};

int main() {
    ncnn::Net yolov8;
    yolov8.load_param("model.ncnn.param"); 
    yolov8.load_model("model.ncnn.bin");

    cv::Mat image = cv::imread("test.jpg");
    if (image.empty()) {
        std::cerr << "Could not open image!" << std::endl;
        return -1;
    }

    int img_w = image.cols;
    int img_h = image.rows;

    cv::Mat resized_image;
    cv::resize(image, resized_image, cv::Size(640, 640)); // Resize to yolov8 input size

    cv::Mat input_data;
    resized_image.convertTo(input_data, CV_32FC3);

    input_data = input_data / 255.f; // Normalizecp

    ncnn::Mat in = ncnn::Mat::from_pixels(input_data.data, ncnn::Mat::PIXEL_BGR, 640, 640);

    ncnn::Extractor ex = yolov8.create_extractor();
    ex.input("in0", in);

    ncnn::Mat out;
    ex.extract("out0", out);
    // for (int i = 0; i < out.h; i++) {
    // float* values = out.row(i);
    // for(int j = 0; j<out.w; j++){
    //     std::cout << values[j] << " ";
    // }
    // std::cout << std::endl;
    // }


    // 6. Postprocess and get the results
    std::vector<Object> objects;
    for (int i = 0; i < out.h; i++) {
        float* values = out.row(i);
        Object object;
        object.rect.x = values[0] * img_w;
        object.rect.y = values[1] * img_h;
        object.rect.width = values[2] * img_w - object.rect.x;
        object.rect.height = values[3] * img_h - object.rect.y;
        object.prob = values[4];
        object.label = (int)values[5];

        objects.push_back(object);
    }

    // 7. Correct NMS implementation
    std::vector<cv::Rect> boxes;
    std::vector<float> scores;
    std::vector<int> classIds; // Add class IDs

    for (const auto& obj : objects) {
        boxes.push_back(obj.rect);
        scores.push_back(obj.prob);
        classIds.push_back(obj.label); // Store class IDs
    }


    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, scores, 0.25f, 0.45f, indices); 

    // 8. Draw bounding boxes on the image (using NMS results)
    for (int index : indices) {
        const auto& obj = objects[index]; // Use the original objects vector
        std::cout << "Bounding box: x=" << obj.rect.x << ", y=" << obj.rect.y << ", width=" << obj.rect.width << ", height=" << obj.rect.height << std::endl;
        cv::rectangle(resized_image, obj.rect, cv::Scalar(0, 255, 0), 2);
        std::string label_text = std::to_string(obj.label) + ": " + std::to_string(obj.prob);
        cv::putText(resized_image, label_text, cv::Point(obj.rect.x, obj.rect.y - 10), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 2);
    }

    cv::imshow("YOLOv8 Detection", resized_image);
    cv::waitKey(0);

    return 0;
}