# HTR-ctc
Pytorch implementation of Handwritten Text Recognition using CTC loss on IAM dataset. 

Selected Features:
* Dataset is saved in a '.pt' file after the initial preprocessing for faster loading operations
* Loader can handle both word and line-level segmentation of words (change loader parameters in train_htr.py). <br>
E.g. IAMLoader('train', level='line', fixed_size=(128, None)) or IAMLoader('train', level='word', fixed_size=(128, None))
* Image resize operations are set through the loader and specifically the *fixed_sized* argument. 
If the width variable is None, the the resize operation keeps the aspect ratio and resize the image according to the specified height (e.g. 128). 
This case generates images of different sizes and thus they cannot be collected to a fixed sized batch. 
To this end, we update the network every K single image operations (e.g. we set batch_size = 1 and iter_size = 16 in in train_code/config.py). 
If a fixed size is selected (across all dimensions), e.g. IAMLoader('train', level='line', fixed_size=(128, 1024)), a batch size could be set (e.g. batch_size = 16 and iter_size = 1).
* **Model architecture** can be modified by changing the the cnn_cfg and rnn_cfg variables in train_code/config.py. 
Specifically, CNN is consisted of multiple stacks of ResBlocks and the default setting `cnn_cfg = [(2, 32), 'M', (4, 64), 'M', (6, 128), 'M', (2, 256)]` is interpeted as follows:
the first stack consists of 2 resblocks with output channels of 32 dimensions, the second of 4 resblocks with 64 output channels etc. 
The 'M' denotes a max-pooling operation of kernel size and stride equal to 2.
CNN backbone is topped by an RNN head which finally produces the character predictions. 
The recurrent newtork is a bidirectional LSTM and its basic configuration is given by the variable rnn_cfg. 
The deafult setting `rnn_cfg = (256, 1)` corresponds to a single layerd LSTM with 256 hidden size. 



Example: <br>
`python train_htr.py -lr 1e-3 -gpu 0`


**Note:** Local paths of IAM dataset (https://fki.tic.heia-fr.ch/databases/iam-handwriting-database) are hardcoded in iam_data_loader/iam_config.py

**Developed with Pytorch 0.4.1 and warpctc_pytorch lib (https://github.com/SeanNaren/warp-ctc) <br> A newer version is coming with the build-in CTC loss of Pytorch (>1.0)** 
