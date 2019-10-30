export TF_FORCE_GPU_ALLOW_GROWTH=true
export TF_ENABLE_AUTO_MIXED_PRECISION=1

source ~/anaconda3/etc/profile.d/conda.sh

conda activate rtav
conda list
TIMESTAMP=`date +%Y%m%d-%H%M%S`

# python3 FasterRCNNModel.py -m test -p VOCdevkit/VOC2012/test --network resnet50 --config_filename rmconfig.pickle --input_weight_path rm-model.hdf5 --output_weight_path rm-model.hdf5 
python3 FasterRCNNModel.py -m test -p data/images --network resnet50 --gpu 1 --config_filename rmconfig.pickle --input_weight_path rm-model.hdf5 

