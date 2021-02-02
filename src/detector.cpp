//
// Created by jiaopan on 7/10/20.
//


#include <opencv2/opencv.hpp>
#include "cJSON.h"
#include "utils.h"
#include "onnx.h"
#include "detector.h"

int Detector::init(const char* model_path,int num_threads){
    try {
        this->unload();
        this->session = this->onnx.init(model_path, num_threads);
        return 0;
    }
    catch (const std::exception &e) {
        std::cout << "init error:" << e.what() << std::endl;
        return 1;
    }
}

char * Detector::detect(cv::Mat image, float min_score){
    int width = image.cols,height = image.rows;
    std::vector<float> input_image;
    const std::map<std::string, int> params = detectorConfig::map.at(this->onnx.model_name);
    utils::createInputImage(input_image, image, params.at("width"), params.at("height"), params.at("channels"));
    std::vector<std::vector<float>> images;
    images.emplace_back(input_image);
    std::vector<Ort::Value> output_tensor = this->onnx.inference(this->session, images);
    return this->yolov5Analysis(output_tensor, width, height, params, this->onnx.label_name, min_score);
}

char *Detector::yolov5Analysis(std::vector<Ort::Value> &output_tensor, int width, int height,
                               const std::map<std::string, int> &output_name_index, std::string label_name,
                               float min_score) {

    float* output = output_tensor[output_name_index.at("output")].GetTensorMutableData<float>();

    size_t size = output_tensor[output_name_index.at("output")].GetTensorTypeAndShapeInfo().GetElementCount();
    int dimensions = output_name_index.at("dimensions");
    int rows = size / dimensions;
    int confidenceIndex = output_name_index.at("confidence_index");
    int labelStartIndex = output_name_index.at("label_start_index");
    float modelWidth = (float)output_name_index.at("width");
    float modelHeight = (float)output_name_index.at("height");
    float xGain = modelWidth / width;
    float yGain = modelHeight / height;

    const std::vector<std::string> classes = labels::map.at(label_name);

    std::vector<cv::Vec4f> locations;
    std::vector<int> labels;
    std::vector<float> confidences;

    std::vector<cv::Rect> src_rects;
    std::vector<cv::Rect> res_rects;
    std::vector<int> res_indexs;

    cv::Rect rect;
    cv::Vec4f location;
    for (int i = 0; i < rows; ++i) {
        int index = i * dimensions;
        if (output[index + confidenceIndex] <= min_score) continue;

        for (int k = labelStartIndex; k < dimensions; ++k) {
            if (output[index + k] * output[index + confidenceIndex]  <= 0.3) continue;
            labels.emplace_back(k - labelStartIndex);
            location[0] = (output[index] - output[index + 2] / 2.0) / xGain;//top left x
            location[1] = (output[index + 1] - output[index + 3] / 2.0) / yGain;//top left y
            location[2] = (output[index] + output[index + 2] / 2.0) / xGain;//bottom right x
            location[3] = (output[index + 1] + output[index + 3] / 2.0) / yGain;//bottom right y

            locations.emplace_back(location);

            rect = cv::Rect(location[0], location[1],location[2] - location[0], location[3] - location[1]);
            src_rects.push_back(rect);
        }
        confidences.emplace_back(output[index + confidenceIndex]);

    }
    utils::nms(src_rects, res_rects, res_indexs);

    cJSON  *result = cJSON_CreateObject(), *items = cJSON_CreateArray();
    for (int i = 0; i < res_indexs.size(); ++i) {
        cJSON  *item = cJSON_CreateObject();
        int index = res_indexs[i];
        cJSON_AddStringToObject(item, "label", classes[labels[index]].c_str());
        cJSON_AddNumberToObject(item, "score", confidences[index]);
        cJSON  *location = cJSON_CreateObject();
        cJSON_AddNumberToObject(location, "x", locations[index][0]);
        cJSON_AddNumberToObject(location, "y", locations[index][1]);
        cJSON_AddNumberToObject(location, "width", locations[index][2] - locations[index][0]);
        cJSON_AddNumberToObject(location, "height", locations[index][3] - locations[index][1]);
        cJSON_AddItemToObject(item, "location", location);
        cJSON_AddItemToArray(items, item);
    }
    cJSON_AddNumberToObject(result, "code", 0);
    cJSON_AddStringToObject(result, "msg", "success");
    cJSON_AddItemToObject(result, "data", items);
    char *resultJson = cJSON_PrintUnformatted(result);
    return resultJson;

}

int Detector::unload(){
    try {
        this->session = Ort::Session(nullptr);
        this->onnx.input_names.clear();
        this->onnx.output_names.clear();
        this->onnx.inputs.clear();
        return 0;
    }
    catch (const std::exception &e) {
        std::cout << "unload error:" << e.what() << std::endl;
        return 1;
    }
}