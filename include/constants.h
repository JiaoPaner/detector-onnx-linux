//
// Created by jiaopan on 7/9/20.
//

#ifndef OBJECT_DETECT_CONSTANTS_H
#define OBJECT_DETECT_CONSTANTS_H

#include <vector>
#include <map>
#include <stdint.h>
struct Input {
    const char* name = nullptr;
    std::vector<int64_t> dims;
    std::vector<float> values;
};
namespace labels{
    const std::vector<std::string> coco = {
            "person","bicycle","car","motorbike","aeroplane","bus","train","truck",
            "boat","traffic light","fire hydrant","stop sign","parking meter","bench",
            "bird","cat","dog","horse","sheep","cow","elephant","bear","zebra","giraffe",
            "backpack","umbrella","handbag","tie","suitcase","frisbee","skis","snowboard",
            "sports ball","kite","baseball bat","baseball glove","skateboard","surfboard",
            "tennis racket","bottle","wine glass","cup","fork","knife","spoon","bowl","banana",
            "apple","sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake",
            "chair","sofa","pottedplant","bed","diningtable","toilet","tvmonitor","laptop","mouse","remote",
            "keyboard","cell phone","microwave","oven","toaster","sink","refrigerator","book",
            "clock","vase","scissors","teddy bear","hair drier","toothbrush"
    };
    const std::map<std::string,std::vector<std::string>> map = {
            {"coco",coco}
    };
}

namespace modelLabel{
    const std::map<std::string,std::string> map = {
           {"ssd","coco"},{"yolov5","coco"}
    };
}



namespace detectorConfig{
    const std::map<std::string,int> ssd = {
            {"width",1200}, {"height",1200}, {"channels",3},
            {"bboxs",0},{"labels",1},{"scores",2},
            {"min_score",.3f}
    };
    const std::map<std::string, int> yolov5 = {
            { "width",640 },{ "height",640 },{ "channels",3 },
            { "output",0 },{ "confidence_index",4 },{ "label_start_index",5 },
            { "dimensions",85 }
    };
    const std::map<std::string,std::map<std::string,int>> map = {
            {"ssd",ssd},{"yolov5",yolov5}
    };

    const std::string modelName{"yolov5"};


}


#endif //OBJECT_DETECT_CONSTANTS_H
