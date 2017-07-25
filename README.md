[![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/sindresorhus/awesome)

# Awesome Semantic Segmentation

## Networks by architecture
- U-Net [https://arxiv.org/pdf/1505.04597.pdf]
	+ https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/ [Caffe + Matlab]
	+ https://github.com/jocicmarko/ultrasound-nerve-segmentation [Keras]
	+ https://github.com/EdwardTyantov/ultrasound-nerve-segmentation [Keras]
	+ https://github.com/ZFTurbo/ZF_UNET_224_Pretrained_Model [Keras]
- SegNet [https://arxiv.org/pdf/1511.00561.pdf]
	+ https://github.com/alexgkendall/caffe-segnet [Caffe]
	+ https://github.com/developmentseed/caffe/tree/segnet-multi-gpu [Caffe]
	+ https://github.com/preddy5/segnet [Keras]
	+ https://github.com/imlab-uiip/keras-segnet [Keras]
	+ https://github.com/andreaazzini/segnet [Tensorflow]
	+ https://github.com/fedor-chervinskii/segnet-torch [Torch]
- DeepLab [https://arxiv.org/pdf/1606.00915.pdf]
	+ https://bitbucket.org/deeplab/deeplab-public/ [Caffe]
	+ https://github.com/cdmh/deeplab-public [Caffe]
	+ https://bitbucket.org/aquariusjay/deeplab-public-ver2 [Caffe]
	+ https://github.com/TheLegendAli/DeepLab-Context [Caffe]
	+ https://github.com/msracver/Deformable-ConvNets/tree/master/deeplab [MXNet]
- Fully-Convolutional Network (FCN) [https://arxiv.org/pdf/1605.06211.pdf]
	+ https://github.com/vlfeat/matconvnet-fcn [MatConvNet]
	+ https://github.com/shelhamer/fcn.berkeleyvision.org [Caffe]
	+ https://github.com/MarvinTeichmann/tensorflow-fcn [Tensorflow]
	+ https://github.com/aurora95/Keras-FCN [Keras]
	+ https://github.com/mzaradzki/neuralnets/tree/master/vgg_segmentation_keras [Keras]
	+ https://github.com/k3nt0w/FCN_via_keras [Keras]
	+ https://github.com/shekkizh/FCN.tensorflow [Tensorflow]
- ENet [https://arxiv.org/pdf/1606.02147.pdf]
 	+ https://github.com/TimoSaemann/ENet [Caffe]
	+ https://github.com/e-lab/ENet-training [Torch]
	+ https://github.com/PavlosMelissinos/enet-keras [Keras]
- LinkNet [https://arxiv.org/pdf/1707.03718.pdf]
	+ https://github.com/e-lab/LinkNet [Torch]	
- DenseNet [https://arxiv.org/pdf/1608.06993.pdf]
	+ https://github.com/flyyufelix/DenseNet-Keras [Keras]
- Tiramisu [https://arxiv.org/pdf/1611.09326.pdf]
	+ https://github.com/0bserver07/One-Hundred-Layers-Tiramisu [Keras]
- DilatedNet [https://arxiv.org/pdf/1511.07122.pdf]
	+ https://github.com/nicolov/segmentation_keras [Keras]
- PixelNet [https://arxiv.org/pdf/1609.06694.pdf]
	+ https://github.com/aayushbansal/PixelNet [Caffe]
- ICNet [https://arxiv.org/pdf/1704.08545.pdf]
	+ https://github.com/hszhao/ICNet [Caffe]
- Mask-RCNN [https://arxiv.org/pdf/1703.06870.pdf]
	+ https://github.com/CharlesShang/FastMaskRCNN [Tensorflow]
	+ https://github.com/jasjeetIM/Mask-RCNN [Caffe]


## Networks by framework
- Keras
	+ https://github.com/gakarak/FCN_MSCOCO_Food_Segmentation
	+ https://github.com/nicolov/segmentation_keras
	+ https://github.com/yihui-he/u-net
	+ https://github.com/abbypa/NNProject_DeepMask

- TensorFlow
	+ https://github.com/DrSleep/tensorflow-deeplab-resnet
	+ https://github.com/warmspringwinds/tf-image-segmentation
	
- Caffe
	+ https://github.com/fyu/dilation
	+ https://github.com/xiaolonw/nips14_loc_seg_testonly
	+ https://github.com/naibaf7/caffe_neural_tool
	+ http://cvlab.postech.ac.kr/research/deconvnet/
	
- torch
	+ https://github.com/facebookresearch/deepmask
	+ https://github.com/erogol/seg-torch
	+ https://github.com/phillipi/pix2pix
	
- MatConvNet
	+ https://github.com/guosheng/refinenet

- MXNet
	+ https://github.com/tornadomeet/mxnet/tree/seg/example/fcn-xs
	+ https://github.com/msracver/FCIS
	+ https://github.com/itijyou/ademxapp

## Papers and Code:

- Simultaneous detection and segmentation

  + http://www.eecs.berkeley.edu/Research/Projects/CS/vision/shape/sds/
  + https://github.com/bharath272/sds_eccv2014
  
- Learning Deconvolution Network for Semantic Segmentation

  + https://github.com/HyeonwooNoh/DeconvNet
  
- Decoupled Deep Neural Network for Semi-supervised Semantic Segmentation

  + https://github.com/HyeonwooNoh/DecoupledNet
  
- Learning to Propose Objects

  + http://vladlen.info/publications/learning-to-propose-objects/ 
  + https://github.com/philkr/lpo
  
- Nonparametric Scene Parsing via Label Transfer

  + http://people.csail.mit.edu/celiu/LabelTransfer/code.html
  
- Other
  + https://github.com/cvjena/cn24
  + http://lmb.informatik.uni-freiburg.de/resources/software.php
  + https://github.com/hszhao/PSPNet
  + https://github.com/NVIDIA/DIGITS/tree/master/examples/semantic-segmentation
  + https://github.com/daijifeng001/MNC
  + https://github.com/voidrank/FastMask
  + http://jamie.shotton.org/work/code.html 
  + https://github.com/amueller/textonboost

## Graphical Models (CRF, MRF)
  + https://github.com/cvlab-epfl/densecrf
  + http://vladlen.info/publications/efficient-inference-in-fully-connected-crfs-with-gaussian-edge-potentials/
  + http://www.philkr.net/home/densecrf
  + http://graphics.stanford.edu/projects/densecrf/
  + https://github.com/amiltonwong/segmentation/blob/master/segmentation.ipynb
  + https://github.com/jliemansifry/super-simple-semantic-segmentation
  + http://users.cecs.anu.edu.au/~jdomke/JGMT/
  + https://www.quora.com/How-can-one-train-and-test-conditional-random-field-CRF-in-Python-on-our-own-training-testing-dataset
  + https://github.com/tpeng/python-crfsuite
  + https://github.com/chokkan/crfsuite
  + https://sites.google.com/site/zeppethefake/semantic-segmentation-crf-baseline
  + https://github.com/lucasb-eyer/pydensecrf

## RNN

  + https://github.com/fvisin/reseg
  + https://github.com/torrvision/crfasrnn
  + https://github.com/bernard24/RIS
  + https://github.com/martinkersner/train-CRF-RNN
  + https://github.com/NP-coder/CLPS1520Project [Tensorflow]
  + https://github.com/renmengye/rec-attend-public [Tensorflow]

## Medical image segmentation:

- DIGITS
  + https://github.com/NVIDIA/DIGITS/tree/master/examples/medical-imaging
  
- U-Net: Convolutional Networks for Biomedical Image Segmentation
  + http://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/
  + https://github.com/dmlc/mxnet/issues/1514
  + https://github.com/orobix/retina-unet
  + https://github.com/fvisin/reseg
  + https://github.com/yulequan/melanoma-recognition
  + http://www.andrewjanowczyk.com/use-case-1-nuclei-segmentation/
  + https://github.com/junyanz/MCILBoost
  + https://github.com/imlab-uiip/lung-segmentation-2d
  + https://github.com/scottykwok/cervix-roi-segmentation-by-unet
  + https://github.com/WeidiXie/cell_counting_v2
  
- Cascaded-FCN
  + https://github.com/IBBM/Cascaded-FCN
  
- Keras
  + https://github.com/jocicmarko/ultrasound-nerve-segmentation
  + https://github.com/EdwardTyantov/ultrasound-nerve-segmentation
  + https://github.com/intact-project/ild-cnn
  
- Using Convolutional Neural Networks (CNN) for Semantic Segmentation of Breast Cancer Lesions (BRCA)
  + https://github.com/ecobost/cnn4brca
  
- Papers:
  + https://www2.warwick.ac.uk/fac/sci/dcs/people/research/csrkbb/tmi2016_ks.pdf
  + Sliding window approach
	  - http://people.idsia.ch/~juergen/nips2012.pdf
	  
 - Data:
   - https://luna16.grand-challenge.org/
  
## Satellite images segmentation

  + https://github.com/mshivaprakash/sat-seg-thesis
  + https://github.com/KGPML/Hyperspectral
  + https://github.com/lopuhin/kaggle-dstl
  + https://github.com/mitmul/ssai
  + https://github.com/mitmul/ssai-cnn
  + https://github.com/azavea/raster-vision
  
 - Data:
  	+ https://github.com/RSIA-LIESMARS-WHU/RSOD-Dataset-

## Video segmentation

  + https://github.com/shelhamer/clockwork-fcn

## Autonomous driving

  + https://github.com/MarvinTeichmann/MultiNet
  + https://github.com/MarvinTeichmann/KittiSeg
  + https://github.com/vxy10/p5_VehicleDetection_Unet [Keras]

## Annotation Tools:

  + https://github.com/AKSHAYUBHAT/ImageSegmentation
  + https://github.com/kyamagu/js-segment-annotator
  + https://github.com/CSAILVision/LabelMeAnnotationTool
  + https://github.com/seanbell/opensurfaces-segmentation-ui
  + https://github.com/lzx1413/labelImgPlus
	
	
## Datasets:

  + [Stanford Background Dataset](http://dags.stanford.edu/projects/scenedataset.html)
  + [Sift Flow Dataset](http://people.csail.mit.edu/celiu/SIFTflow/)
  + [Barcelona Dataset](http://www.cs.unc.edu/~jtighe/Papers/ECCV10/)
  + [Microsoft COCO dataset](http://mscoco.org/)
  + [MSRC Dataset](http://research.microsoft.com/en-us/projects/objectclassrecognition/)
  + [LITS Liver Tumor Segmentation Dataset](https://competitions.codalab.org/competitions/15595)
  + [KITTI](http://www.cvlibs.net/datasets/kitti/eval_road.php)
  + [Stanford background dataset](http://dags.stanford.edu/projects/scenedataset.html)
  + [Data from Games dataset](https://download.visinf.tu-darmstadt.de/data/from_games/)
  + [Human parsing dataset](https://github.com/lemondan/HumanParsing-Dataset)


## Results:

  + [MSRC-21](http://rodrigob.github.io/are_we_there_yet/build/semantic_labeling_datasets_results.html)
  + [Cityscapes](https://www.cityscapes-dataset.com/benchmarks/)
  + [VOC2012](http://host.robots.ox.ac.uk:8080/leaderboard/displaylb.php?challengeid=11&compid=6)


## To look at
  + https://github.com/meetshah1995/pytorch-semseg
  + https://github.com/kjw0612/awesome-deep-vision#semantic-segmentation
  + https://github.com/desimone/segmentation-models
  + https://github.com/mrgloom/Semantic-Segmentation-Evaluation/issues/1
  + https://github.com/nightrome/really-awesome-semantic-segmentation


## Blog posts, other:

  + https://handong1587.github.io/deep_learning/2015/10/09/segmentation.html
  + http://www.andrewjanowczyk.com/efficient-pixel-wise-deep-learning-on-large-images/
  + https://devblogs.nvidia.com/parallelforall/image-segmentation-using-digits-5/
  + https://github.com/NVIDIA/DIGITS/tree/master/examples/binary-segmentation
  + https://github.com/NVIDIA/DIGITS/tree/master/examples/semantic-segmentation

