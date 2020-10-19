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
    return detector.init(model_path, num_threads);
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

int main(){


    //test detectByFile()
    const char* model_path = "/home/jiaopan/projects/c++/detector-onnx-linux/model/yolov5s-320.onnx";
    int status = init(model_path,1);
    std::cout << "status" << status << std::endl;
    char* result = detectByFile("/home/jiaopan/Downloads/bus.jpg",0.5);
    std::cout << "result:" << result << std::endl;
    //std::cout << "unload:" << unload("detector") << std::endl;

    /**/
    cv::Mat image = cv::imread("/home/jiaopan/Downloads/bus.jpg");
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
    std::cin.get();
    return 0;
}