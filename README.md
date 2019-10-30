# ISS-VSE-ObjectDetection
Singapore roadmarks dataset

## Developer Guide under Ubuntu 18.04 LTS

### Install conda:
1. Follows https://docs.anaconda.com/anaconda/install/linux/ for basic conda installation.
2. conda create -n rtav python=3.6 
3. conda activate rtav
4. conda install opencv=4.1.0 imutils imgaug librosa tensorflow-gpu=1.13.1 cython -c conda-forge
5. conda install spyder pathlib ipython notebook yaml pandas keras-gpu pydot graphviz -c conda-forge

### To Prepare Train images and annotations
1. conda activate rtav
2. git clone https://github.com/XiaoyanYang2008/ISS-VSE-ObjectDetection.git
3. cd ISS-VSE-ObjectDetection
4. 

### To Train Faster RCNN network model
pip3 install -r requirements.txt
pip3 install tensorflow-gpu # or pip3 install tensorflow #for CPU version!! Really??
wget http://home.leeseng.tech/training-data-20190925.zip
unzip training-data-20190925.zip
python3 trainCNN.py
python3 testCNN.py

optional, if GPU not working in tensorflow: sudo apt install cuda-10-0 nvidia-driver-430



runs pycharm community edition.
open project on folder, ISS-VSE-ObjectDetection.
setup "Project Interpreter with existing conda"
run/debug trainCNN.py
note: uses pycharm 2019.1.x. pycharm 2019.2.x needs to comments out server.py in pydevd_dont_trace_files.py under pycharm program folder. Refers bug report, https://youtrack.jetbrains.com/issue/PY-37609
