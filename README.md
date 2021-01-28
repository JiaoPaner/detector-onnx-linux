
**this repo based on yolov5 v3.0**
#### requirements
* opencv 
* onnx runtime linux


### how to build
1.git clone  https://github.com/JiaoPaner/detector-onnx-linux.git <br>
2.add your onnx file to model dir<br>
3.modify the main method in src/api.cpp or the main.py in test dir to test <br>
4.mkdir build <br>
5.cd build <br>
6.make -j8<br>
