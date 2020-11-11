//
// Created by jiaopan on 7/15/20.
//

#include "api.h"
static Detector detector;
/**
 * init
 * @param model_path
 * @param num_threads
 * @return
 */
int init(const char* model_path,int num_threads) {
    std::cout << "loading model:" << model_path << std::endl;
    int status =detector.init(model_path, num_threads);
    std::cout << "this is a detector lib by jiaopaner@qq.com" << std::endl;
    return status;
}
int unload() {
    return detector.unload();
}

/**
 * detect
 * @return
 */
char* detectByBase64(const char* base64_data, float min_score) {
    try {
        std::string data(base64_data);
        cv::Mat image = utils::base64ToMat(data);
        return detector.detect(image, min_score);
    }
    catch (const char* msg) {
        cJSON* result = cJSON_CreateObject(), * data = cJSON_CreateArray();;
        cJSON_AddNumberToObject(result, "code", 1);
        cJSON_AddStringToObject(result, "msg", msg);
        cJSON_AddItemToObject(result, "data", data);
        return cJSON_PrintUnformatted(result);
    }
}

char* detectByFile(const char* file, float min_score) {
    try {
        cv::Mat image = cv::imread(file);
        return detector.detect(image, min_score);
    }
    catch (const char* msg) {
        cJSON* result = cJSON_CreateObject(), * data = cJSON_CreateArray();;
        cJSON_AddNumberToObject(result, "code", 1);
        cJSON_AddStringToObject(result, "msg", msg);
        cJSON_AddItemToObject(result, "data", data);
        return cJSON_PrintUnformatted(result);
    }
}

#include <chrono>
using std::chrono::high_resolution_clock;
using std::chrono::milliseconds;
int main(){
    //test detectByFile()
    const char* model_path = "/home/jiaopan/projects/c++/detector-onnx-linux/model/best.onnx";
    int status = init(model_path,1);
    std::cout << "status" << status << std::endl;
    high_resolution_clock::time_point start = high_resolution_clock::now();
    char* result = detectByFile("/home/jiaopan/Downloads/b2.jpg",0.3);
    high_resolution_clock::time_point end = high_resolution_clock::now();
    milliseconds cost = std::chrono::duration_cast<milliseconds>(end - start);
    std::cout << "The elapsed is:" << cost.count() <<"ms"<< std::endl;
    std::cout << "result:" << result << std::endl;
    //std::cout << "unload:" << unload("detector") << std::endl;

    /**/
    cv::Mat image = cv::imread("/home/jiaopan/Downloads/b2.jpg");
    cv::Scalar colors[20] = { cv::Scalar(255, 0, 0), cv::Scalar(0, 255, 0),cv::Scalar(0,0,255),
                              cv::Scalar(255,255,0),cv::Scalar(255,0,255), cv::Scalar(0,255,255),
                              cv::Scalar(255,255,255), cv::Scalar(127,0,0),cv::Scalar(0,127,0),
                              cv::Scalar(0,0,127),cv::Scalar(127,127,0), cv::Scalar(127,0,127),
                              cv::Scalar(0,127,127), cv::Scalar(127,127,127),cv::Scalar(127,255,0),
                              cv::Scalar(127,0,255),cv::Scalar(127,255,255), cv::Scalar(0,127,255),
                              cv::Scalar(255,127,0), cv::Scalar(0,255,127) };  //ÑÕÉ«

    cJSON *root;
    root = cJSON_Parse(result);
    cJSON *code = cJSON_GetObjectItem(root, "code");

    if (code->valueint == 0) {
        cJSON *data = cJSON_GetObjectItem(root, "data");
        int size = cJSON_GetArraySize(data);
        cv::Rect rect;
        for (int i = 0; i < size; ++i) {
            cJSON *item = cJSON_GetArrayItem(data, i);
            cJSON *label = cJSON_GetObjectItem(item, "label");
            std::cout << label->valuestring << std::endl;
            cJSON *location = cJSON_GetObjectItem(item, "location");
            rect = cv::Rect(cJSON_GetObjectItem(location, "x")->valuedouble, cJSON_GetObjectItem(location, "y")->valuedouble,
                            cJSON_GetObjectItem(location, "width")->valuedouble, cJSON_GetObjectItem(location, "height")->valuedouble);
            rectangle(image, rect, cv::Scalar(colors[i % 4]), 3, 1, 0);
            putText(image, label->valuestring, cv::Point(rect.x + 5, rect.y + 13), cv::FONT_HERSHEY_COMPLEX, 0.5, cv::Scalar(colors[i % 4]), 1, 8);

        }
    }
    imwrite("output.jpg", image);

    // unload();//unload loaded model
    //std::cin.get();
    return 0;
}