# ISS-VSE-ObjectDetection
Singapore roadmarks dataset

## Developer Guide under Ubuntu 18.04 LTS

### Install conda:
1. Follows https://docs.anaconda.com/anaconda/install/linux/ for basic conda installation.
2. conda create -n rtav python=3.6 
3. conda activate rtav
4. conda install opencv=4.1.0 imutils imgaug librosa tensorflow-gpu=1.13.1 cython -c conda-forge
5. conda install spyder pathlib ipython notebook yaml pandas keras-gpu pydot graphviz -c conda-forge
note: conda's notebook package is required for jupyter notebook. If not, jupyter notebook reports "No module named 'tensorflow'" error, due to Ubuntu system python's notebook package was running instead.

### Prepare Train images and annotations
1. conda activate rtav
2. git clone https://github.com/XiaoyanYang2008/ISS-VSE-ObjectDetection.git
3. cd ISS-VSE-ObjectDetection
4. python3 FasterRCNNModel.py -m to_simple_parser

### Train Faster RCNN network model
1. ./train.sh
or  
1. conda activate rtav
2. python3 FasterRCNNModel.py -m train -p training.csv -o simple --network resnet50 --config_filename rmconfig.pickle --input_weight_path rm-model.hdf5 --output_weight_path rm-model.hdf5 &> log-train.txt


pip3 install -r requirements.txt
pip3 install tensorflow-gpu # or pip3 install tensorflow #for CPU version!! Really??
wget http://home.leeseng.tech/training-data-20190925.zip
unzip training-data-20190925.zip
python3 trainCNN.py
python3 testCNN.py

optional, if GPU not working in tensorflow: sudo apt install cuda-10-0 nvidia-driver-430
if cuda-10-0 not found, follows https://www.tensorflow.org/install/gpu#ubuntu_1804_cuda_10



runs pycharm community edition.
open project on folder, ISS-VSE-ObjectDetection.
setup "Project Interpreter with existing conda"
run/debug trainCNN.py
note: uses pycharm 2019.1.x. pycharm 2019.2.x needs to comments out server.py in pydevd_dont_trace_files.py under pycharm program folder. Refers bug report, https://youtrack.jetbrains.com/issue/PY-37609

### To annotate dataset in CVAT
Please follow openCV's CVAT installation guide at https://github.com/opencv/cvat/blob/develop/cvat/apps/documentation/installation.md#ubuntu-1804-x86_64amd64 
