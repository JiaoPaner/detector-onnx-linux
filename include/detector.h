//
// Created by jiaopan on 7/10/20.
//

#include <onnx/onnxruntime_cxx_api.h>
#include <onnx/onnxruntime_c_api.h>

class Detector {
public:
    int init(const char* model_path,int num_threads);
    char* detect(cv::Mat image, float min_score);
    char* yolov5Analysis(std::vector<Ort::Value> &output_tensor,
                         int width, int height, const std::map<std::string, int> &output_name_index,
                         std::string label_name, float min_score = 0.5f);
    int unload();
    ~Detector(){};
    private:
        OnnxInstance onnx;
        Ort::Session session = Ort::Session(nullptr);
};

