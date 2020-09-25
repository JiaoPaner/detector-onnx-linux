//
// Created by jiaopan on 7/9/20.
//


#include <onnxruntime_cxx_api.h>
#include <onnxruntime_c_api.h>
#include "constants.h"
class OnnxInstance {
    public:

        Ort::Session init(std::string model_path,std::string model_name,int num_threads=1);
        std::vector<Ort::Value> inference(Ort::Session &session,std::vector<std::vector<float>> images);
        std::vector<Input> inputs;
        std::vector<const char*> input_names;
        std::vector<const char*> output_names;
        std::string model_name;
        std::string label_name;
    ~OnnxInstance(){}
    private:
        void createInputsInfo(Ort::Session &session);
        void createOutputsInfo(Ort::Session &session);
};



