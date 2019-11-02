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

optional, nvidia-smi -l # to check if GPU in used during training.
if GPU not working in tensorflow: sudo apt install cuda-10-0 nvidia-driver-430
if cuda-10-0 not found, follows https://www.tensorflow.org/install/gpu#ubuntu_1804_cuda_10

### Prepare Train images and annotations
note: install git lfs as per https://git-lfs.github.com/
1. conda activate rtav
2. git clone https://github.com/XiaoyanYang2008/ISS-VSE-ObjectDetection.git
3. cd ISS-VSE-ObjectDetection
4. python3 FasterRCNNModel.py -m to_simple_parser

### Train Faster RCNN network model
1. ./train.sh &
2. tail -f log-train-$TIMESTAMP.txt # replace $TIMESTAMP as current timestamp. double tap in bash to list.
or  
1. conda activate rtav
2. python3 FasterRCNNModel.py -m train -p training.csv -o simple --network resnet50 --config_filename rmconfig.pickle --input_weight_path rm-model.hdf5 --output_weight_path rm-model.hdf5 &> log-train.txt
3. tail -f log-train.txt

### Quick Test Faster RCNN network training model, at least 3 hours of training later.
1. ./test.sh    # ctrl C and escape to quit. utilize 2nd GPU as model is 6.5GB in GPU RAM.

### Test Video generation with Faster RCNN network model
1. ./test-video.sh  # check *detected.mp4 under folder, data/test

note: rm-model-loss-0.285.hdf5 and rmconfig-loss-0.285.pickle can be used in test-video.sh


### Debug under pycharm
0. Please take note. Followings are not commands for terminal.
1. open pycharm community edition.
2. open project on folder, ISS-VSE-ObjectDetection.
3. setup "Project Interpreter with existing conda"
4. open FasterRCNNModel.py

note: uses pycharm 2019.1.x. pycharm 2019.2.x needs to comments out server.py in pydevd_dont_trace_files.py under pycharm program folder. Refers bug report, https://youtrack.jetbrains.com/issue/PY-37609

### To annotate dataset in CVAT
Please follow openCV's CVAT installation guide at https://github.com/opencv/cvat/blob/develop/cvat/apps/documentation/installation.md#ubuntu-1804-x86_64amd64 

Please refer User guide at https://github.com/opencv/cvat/blob/develop/cvat/apps/documentation/user_guide.md pay attention to interpolation mode as that allows us to annotate videos with ease.
