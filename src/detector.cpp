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
        std::cout << "this is a detector lib by jiaopaner@qq.com" << std::endl;
        return 0;
    }
    catch (const std::exception &e) {
        std::cout << "init error:" << e.what() << std::endl;
        return 1;
    }
}

char * Detector::detect(cv::Mat image, float min_score){
    std::vector<float> input_image;
    const std::map<std::string, int> params = detectorConfig::map.at(this->onnx.model_name);
    utils::createInputImage(input_image, image, params.at("width"), params.at("height"), params.at("channels"), true);
    std::vector<std::vector<float>> images;
    images.emplace_back(input_image);
    std::vector<Ort::Value> output_tensor = this->onnx.inference(this->session, images);
    //return Detector::ssdAnalysis(output_tensor, image.cols, image.rows, params, onnx.label_name, min_score);
    return this->yolov5Analysis(output_tensor, image.cols, image.rows, params, this->onnx.label_name, min_score);
}

char *Detector::ssdAnalysis(std::vector<Ort::Value> &output_tensor,
                            int width, int height,const std::map<std::string,int> &output_name_index,
                            std::string label_name,float min_score) {

    float* bboxs = output_tensor[output_name_index.at("bboxs")].GetTensorMutableData<float>();
    int* labels_index = output_tensor[output_name_index.at("labels")].GetTensorMutableData<int>();
    float* scores = output_tensor[output_name_index.at("scores")].GetTensorMutableData<float>();
    size_t size = output_tensor[output_name_index.at("labels")].GetTensorTypeAndShapeInfo().GetElementCount();
    const std::vector<std::string> classes = labels::map.at(label_name);

    std::vector<cv::Vec4f> locations(size);
    std::vector<int> labels(size);
    std::vector<float> confidences(size);

    std::vector<cv::Rect> src_rects;
    std::vector<cv::Rect> res_rects;
    std::vector<int> res_indexs;

    cv::Rect rect;

    for (int i = 0; i < size; i++) {
        if (scores[i] > min_score) {
            locations[i][0] = bboxs[i * 4]  * width;
            locations[i][1] = bboxs[i * 4 + 1] * height;
            locations[i][2] = bboxs[i * 4 + 2] * width;
            locations[i][3] = bboxs[i * 4 + 3] * height;
            rect = cv::Rect(locations[i][0], locations[i][1],
                            locations[i][2] - locations[i][0], locations[i][3] - locations[i][1]);
            src_rects.push_back(rect);
            labels[i] = labels_index[i * 2];
            confidences[i] = scores[i];
        }
    }
    utils::nms(src_rects,res_rects,res_indexs);

    cJSON  *result = cJSON_CreateObject(), *items = cJSON_CreateArray();
    for (int i = 0; i < res_indexs.size(); ++i) {
        cJSON  *item = cJSON_CreateObject();
        int index = res_indexs[i];
        cJSON_AddStringToObject(item, "label", classes[labels[index]-1].c_str());
        cJSON_AddNumberToObject(item,"score",confidences[index]);
        cJSON  *location = cJSON_CreateObject();
        cJSON_AddNumberToObject(location,"x",locations[index][0]);
        cJSON_AddNumberToObject(location,"y",locations[index][1]);
        cJSON_AddNumberToObject(location,"width",locations[index][2] - locations[index][0]);
        cJSON_AddNumberToObject(location,"height",locations[index][3] - locations[index][1]);
        cJSON_AddItemToObject(item,"location",location);
        cJSON_AddItemToArray(items,item);
    }
    cJSON_AddNumberToObject(result, "code", 0);
    cJSON_AddStringToObject(result, "msg", "success");
    cJSON_AddItemToObject(result, "data", items);
    char *resultJson = cJSON_PrintUnformatted(result);
    return resultJson;
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
        if (output[index + confidenceIndex] <= 0.4f) continue;

        for (int j = labelStartIndex; j < dimensions; ++j) {
            output[index + j] = output[index + j] * output[index + confidenceIndex];
        }

        for (int k = labelStartIndex; k < dimensions; ++k) {
            if (output[index + k] <= min_score) continue;

            location[0] = (output[index] - output[index + 2] / 2) / xGain;//top left x
            location[1] = (output[index + 1] - output[index + 3] / 2) / yGain;//top left y
            location[2] = (output[index] + output[index + 2] / 2) / xGain;//bottom right x
            location[3] = (output[index + 1] + output[index + 3] / 2) / yGain;//bottom right y

            locations.emplace_back(location);

            rect = cv::Rect(location[0], location[1],
                            location[2] - location[0], location[3] - location[1]);
            src_rects.push_back(rect);
            labels.emplace_back(k - labelStartIndex);


            confidences.emplace_back(output[index + k]);
        }

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