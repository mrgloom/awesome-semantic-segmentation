[![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/sindresorhus/awesome)

# Awesome Semantic Segmentation

## Networks by architecture
### Semantic segmentation
- U-Net [https://arxiv.org/pdf/1505.04597.pdf] [2015]
	+ https://github.com/zhixuhao/unet [Keras]
	+ https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/ [Caffe + Matlab]
	+ https://github.com/jocicmarko/ultrasound-nerve-segmentation [Keras]
	+ https://github.com/EdwardTyantov/ultrasound-nerve-segmentation [Keras]
	+ https://github.com/ZFTurbo/ZF_UNET_224_Pretrained_Model [Keras]
	+ https://github.com/yihui-he/u-net [Keras]
	+ https://github.com/jakeret/tf_unet [Tensorflow]
	+ https://github.com/DLTK/DLTK/blob/master/examples/Toy_segmentation/simple_dltk_unet.ipynb [Tensorflow]
	+ https://github.com/divamgupta/image-segmentation-keras [Keras]
	+ https://github.com/ZijunDeng/pytorch-semantic-segmentation [PyTorch]
	+ https://github.com/akirasosa/mobile-semantic-segmentation [Keras]
	+ https://github.com/orobix/retina-unet [Keras]
	+ https://github.com/masahi/nnvm-vision-demo/blob/master/unet_segmentation.py [onnx+nnvm]
	+ https://github.com/qureai/ultrasound-nerve-segmentation-using-torchnet [Torch]
	+ https://github.com/ternaus/TernausNet [PyTorch]
- SegNet [https://arxiv.org/pdf/1511.00561.pdf] [2016]
	+ https://github.com/alexgkendall/caffe-segnet [Caffe]
	+ https://github.com/developmentseed/caffe/tree/segnet-multi-gpu [Caffe]
	+ https://github.com/preddy5/segnet [Keras]
	+ https://github.com/imlab-uiip/keras-segnet [Keras]
	+ https://github.com/andreaazzini/segnet [Tensorflow]
	+ https://github.com/fedor-chervinskii/segnet-torch [Torch]
	+ https://github.com/0bserver07/Keras-SegNet-Basic [Keras]
	+ https://github.com/tkuanlun350/Tensorflow-SegNet [Tensorflow]
	+ https://github.com/divamgupta/image-segmentation-keras [Keras]
	+ https://github.com/ZijunDeng/pytorch-semantic-segmentation [PyTorch]
	+ https://github.com/chainer/chainercv/tree/master/examples/segnet [Chainer]
	+ https://github.com/ykamikawa/keras-SegNet [Keras]
- DeepLab [https://arxiv.org/pdf/1606.00915.pdf] [2017]
	+ https://bitbucket.org/deeplab/deeplab-public/ [Caffe]
	+ https://github.com/cdmh/deeplab-public [Caffe]
	+ https://bitbucket.org/aquariusjay/deeplab-public-ver2 [Caffe]
	+ https://github.com/TheLegendAli/DeepLab-Context [Caffe]
	+ https://github.com/msracver/Deformable-ConvNets/tree/master/deeplab [MXNet]
	+ https://github.com/DrSleep/tensorflow-deeplab-resnet [Tensorflow]
	+ https://github.com/muyang0320/tensorflow-deeplab-resnet-crf [TensorFlow]
	+ https://github.com/isht7/pytorch-deeplab-resnet [PyTorch]
	+ https://github.com/bermanmaxim/jaccardSegment [PyTorch]
	+ https://github.com/martinkersner/train-DeepLab [Caffe]
	+ https://github.com/chenxi116/TF-deeplab [Tensorflow]
	+ https://github.com/bonlime/keras-deeplab-v3-plus [Keras]
- FCN [https://arxiv.org/pdf/1605.06211.pdf] [2016]
	+ https://github.com/vlfeat/matconvnet-fcn [MatConvNet]
	+ https://github.com/shelhamer/fcn.berkeleyvision.org [Caffe]
	+ https://github.com/MarvinTeichmann/tensorflow-fcn [Tensorflow]
	+ https://github.com/aurora95/Keras-FCN [Keras]
	+ https://github.com/mzaradzki/neuralnets/tree/master/vgg_segmentation_keras [Keras]
	+ https://github.com/k3nt0w/FCN_via_keras [Keras]
	+ https://github.com/shekkizh/FCN.tensorflow [Tensorflow]
	+ https://github.com/seewalker/tf-pixelwise [Tensorflow]
	+ https://github.com/divamgupta/image-segmentation-keras [Keras]
	+ https://github.com/ZijunDeng/pytorch-semantic-segmentation [PyTorch]
	+ https://github.com/wkentaro/pytorch-fcn [PyTorch]
	+ https://github.com/wkentaro/fcn [Chainer]
	+ https://github.com/apache/incubator-mxnet/tree/master/example/fcn-xs [MxNet]
	+ https://github.com/muyang0320/tf-fcn [Tensorflow]
	+ https://github.com/ycszen/pytorch-seg [PyTorch]
	+ https://github.com/Kaixhin/FCN-semantic-segmentation [PyTorch]
	+ https://github.com/petrama/VGGSegmentation [Tensorflow]
	+ https://github.com/simonguist/testing-fcn-for-cityscapes [Caffe]
	+ https://github.com/hellochick/semantic-segmentation-tensorflow [Tensorflow]
- ENet [https://arxiv.org/pdf/1606.02147.pdf] [2016]
 	+ https://github.com/TimoSaemann/ENet [Caffe]
	+ https://github.com/e-lab/ENet-training [Torch]
	+ https://github.com/PavlosMelissinos/enet-keras [Keras]
	+ https://github.com/fregu856/segmentation [Tensorflow]
	+ https://github.com/kwotsin/TensorFlow-ENet [Tensorflow]
- LinkNet [https://arxiv.org/pdf/1707.03718.pdf] [2017]
	+ https://github.com/e-lab/LinkNet [Torch]
- DenseNet [https://arxiv.org/pdf/1608.06993.pdf] [2018]
	+ https://github.com/flyyufelix/DenseNet-Keras [Keras]
- Tiramisu [https://arxiv.org/pdf/1611.09326.pdf] [2017]
	+ https://github.com/0bserver07/One-Hundred-Layers-Tiramisu [Keras]
	+ https://github.com/SimJeg/FC-DenseNet [Lasagne]
- DilatedNet [https://arxiv.org/pdf/1511.07122.pdf] [2016]
	+ https://github.com/nicolov/segmentation_keras [Keras]
	+ https://github.com/fyu/dilation [Caffe]
	+ https://github.com/fyu/drn#semantic-image-segmentataion [PyTorch]
	+ https://github.com/hangzhaomit/semantic-segmentation-pytorch [PyTorch]
- PixelNet [https://arxiv.org/pdf/1609.06694.pdf] [2016]
	+ https://github.com/aayushbansal/PixelNet [Caffe]
- ICNet [https://arxiv.org/pdf/1704.08545.pdf] [2017]
	+ https://github.com/hszhao/ICNet [Caffe]
	+ https://github.com/ai-tor/Keras-ICNet [Keras]
	+ https://github.com/hellochick/ICNet-tensorflow [Tensorflow]
- ERFNet [http://www.robesafe.uah.es/personal/eduardo.romera/pdfs/Romera17iv.pdf] [?]
	+ https://github.com/Eromera/erfnet [Torch]
	+ https://github.com/Eromera/erfnet_pytorch [PyTorch]
- RefineNet [https://arxiv.org/pdf/1611.06612.pdf] [2016]
	+ https://github.com/guosheng/refinenet [MatConvNet]
- PSPNet [https://arxiv.org/pdf/1612.01105.pdf,https://hszhao.github.io/projects/pspnet/] [2017]
	+ https://github.com/hszhao/PSPNet [Caffe]
	+ https://github.com/ZijunDeng/pytorch-semantic-segmentation [PyTorch]
	+ https://github.com/mitmul/chainer-pspnet [Chainer]
	+ https://github.com/Vladkryvoruchko/PSPNet-Keras-tensorflow [Keras/Tensorflow]
	+ https://github.com/pudae/tensorflow-pspnet [Tensorflow]
	+ https://github.com/hellochick/PSPNet-tensorflow [Tensorflow]
	+ https://github.com/hellochick/semantic-segmentation-tensorflow [Tensorflow]
- DeconvNet [https://arxiv.org/pdf/1505.04366.pdf] [2015]
	+ http://cvlab.postech.ac.kr/research/deconvnet/ [Caffe]
	+ https://github.com/HyeonwooNoh/DeconvNet [Caffe]
	+ https://github.com/fabianbormann/Tensorflow-DeconvNet-Segmentation [Tensorflow]
- FRRN [https://arxiv.org/pdf/1611.08323.pdf] [2016]
	+ https://github.com/TobyPDE/FRRN [Lasagne]
- GCN [https://arxiv.org/pdf/1703.02719.pdf] [2017]
	+ https://github.com/ZijunDeng/pytorch-semantic-segmentation [PyTorch]
	+ https://github.com/ycszen/pytorch-seg [PyTorch]
- LRR [https://arxiv.org/pdf/1605.02264.pdf] [2016]
	+ https://github.com/golnazghiasi/LRR [Matconvnet]
- DUC, HDC [https://arxiv.org/pdf/1702.08502.pdf] [2017]
	+ https://github.com/ZijunDeng/pytorch-semantic-segmentation [PyTorch]
	+ https://github.com/ycszen/pytorch-seg [PyTorch]
- MultiNet [https://arxiv.org/pdf/1612.07695.pdf] [2016]
	+ https://github.com/MarvinTeichmann/MultiNet
	+ https://github.com/MarvinTeichmann/KittiSeg
- Segaware [https://arxiv.org/pdf/1708.04607.pdf] [2017]
	+ https://github.com/aharley/segaware [Caffe]
- Semantic Segmentation using Adversarial Networks [https://arxiv.org/pdf/1611.08408.pdf] [2016]
	+ https://github.com/oyam/Semantic-Segmentation-using-Adversarial-Networks [Chainer]
- PixelDCN [https://arxiv.org/pdf/1705.06820.pdf] [2017]
	+ https://github.com/HongyangGao/PixelDCN [Tensorflow]
- ShuffleSeg [https://arxiv.org/pdf/1803.03816.pdf] [2018]
	+ https://github.com/MSiam/TFSegmentation [TensorFlow]
- AdaptSegNet [https://arxiv.org/pdf/1802.10349.pdf] [2018]
	+ https://github.com/wasidennis/AdaptSegNet [PyTorch]
	
### Instance aware segmentation
- FCIS [https://arxiv.org/pdf/1611.07709.pdf]
	+ https://github.com/msracver/FCIS [MxNet]
- MNC [https://arxiv.org/pdf/1512.04412.pdf]
	+ https://github.com/daijifeng001/MNC [Caffe]
- DeepMask [https://arxiv.org/pdf/1506.06204.pdf]
	+ https://github.com/facebookresearch/deepmask [Torch]
- SharpMask [https://arxiv.org/pdf/1603.08695.pdf]
	+ https://github.com/facebookresearch/deepmask [Torch]
- Mask-RCNN [https://arxiv.org/pdf/1703.06870.pdf]
	+ https://github.com/CharlesShang/FastMaskRCNN [Tensorflow]
	+ https://github.com/jasjeetIM/Mask-RCNN [Caffe]
	+ https://github.com/TuSimple/mx-maskrcnn [MxNet]
	+ https://github.com/matterport/Mask_RCNN [Keras]
- RIS [https://arxiv.org/pdf/1511.08250.pdf]
  + https://github.com/bernard24/RIS [Torch]
- FastMask [https://arxiv.org/pdf/1612.08843.pdf]
  + https://github.com/voidrank/FastMask [Caffe]
- BlitzNet [https://arxiv.org/pdf/1708.02813.pdf]
  + https://github.com/dvornikita/blitznet [Tensorflow]

### Weakly-supervised segmentation
- SEC [https://arxiv.org/pdf/1603.06098.pdf]
  + https://github.com/kolesman/SEC [Caffe]

## RNN
- ReNet [https://arxiv.org/pdf/1505.00393.pdf]
  + https://github.com/fvisin/reseg [Lasagne]
- ReSeg [https://arxiv.org/pdf/1511.07053.pdf]
  + https://github.com/Wizaron/reseg-pytorch [PyTorch]
  + https://github.com/fvisin/reseg [Lasagne]
- RIS [https://arxiv.org/pdf/1511.08250.pdf]
  + https://github.com/bernard24/RIS [Torch]
- CRF-RNN [http://www.robots.ox.ac.uk/%7Eszheng/papers/CRFasRNN.pdf]
  + https://github.com/martinkersner/train-CRF-RNN [Caffe]
  + https://github.com/torrvision/crfasrnn [Caffe]
  + https://github.com/NP-coder/CLPS1520Project [Tensorflow]
  + https://github.com/renmengye/rec-attend-public [Tensorflow]
  + https://github.com/sadeepj/crfasrnn_keras [Keras]
 
## GANS
+ https://github.com/NVIDIA/pix2pixHD

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

## Datasets:

  + [Stanford Background Dataset](http://dags.stanford.edu/projects/scenedataset.html)
  + [Sift Flow Dataset](http://people.csail.mit.edu/celiu/SIFTflow/)
  + [Barcelona Dataset](http://www.cs.unc.edu/~jtighe/Papers/ECCV10/)
  + [Microsoft COCO dataset](http://mscoco.org/)
  + [MSRC Dataset](http://research.microsoft.com/en-us/projects/objectclassrecognition/)
  + [LITS Liver Tumor Segmentation Dataset](https://competitions.codalab.org/competitions/15595)
  + [KITTI](http://www.cvlibs.net/datasets/kitti/eval_road.php)
  + [Pascal Context](http://www.cs.stanford.edu/~roozbeh/pascal-context/)
  + [Data from Games dataset](https://download.visinf.tu-darmstadt.de/data/from_games/)
  + [Human parsing dataset](https://github.com/lemondan/HumanParsing-Dataset)
  + [Mapillary Vistas Dataset](https://www.mapillary.com/dataset/vistas)
  + [Microsoft AirSim](https://github.com/Microsoft/AirSim)
  + [MIT Scene Parsing Benchmark](http://sceneparsing.csail.mit.edu/)
  + [COCO 2017 Stuff Segmentation Challenge](http://cocodataset.org/#stuff-challenge2017)
  + [ADE20K Dataset](http://groups.csail.mit.edu/vision/datasets/ADE20K/)
  + [INRIA Annotations for Graz-02](http://lear.inrialpes.fr/people/marszalek/data/ig02/)
  + [Daimler dataset](http://www.gavrila.net/Datasets/Daimler_Pedestrian_Benchmark_D/daimler_pedestrian_benchmark_d.html)
  + [ISBI Challenge: Segmentation of neuronal structures in EM stacks](http://brainiac2.mit.edu/isbi_challenge/)
  + [INRIA Annotations for Graz-02 (IG02)](https://lear.inrialpes.fr/people/marszalek/data/ig02/)
  + [Pratheepan Dataset](http://cs-chan.com/downloads_skin_dataset.html)
  + [Clothing Co-Parsing (CCP) Dataset](https://github.com/bearpaw/clothing-co-parsing)
  + [Inria Aerial Image](https://project.inria.fr/aerialimagelabeling/)
  + [ApolloScape](http://apolloscape.auto/scene.html)
  + [UrbanMapper3D](https://community.topcoder.com/longcontest/?module=ViewProblemStatement&rd=17007&pm=14703)
  + [RoadDetector](https://community.topcoder.com/longcontest/?module=ViewProblemStatement&rd=17036&pm=14735)

## Benchmarks
  + https://github.com/ZijunDeng/pytorch-semantic-segmentation [PyTorch]
  + https://github.com/meetshah1995/pytorch-semseg [PyTorch]
  + https://github.com/GeorgeSeif/Semantic-Segmentation-Suite [Tensorflow]
  + https://github.com/MSiam/TFSegmentation [Tensorflow]
  + https://github.com/CSAILVision/sceneparsing [Caffe+Matlab]

## Annotation Tools:

  + https://github.com/AKSHAYUBHAT/ImageSegmentation
  + https://github.com/kyamagu/js-segment-annotator
  + https://github.com/CSAILVision/LabelMeAnnotationTool
  + https://github.com/seanbell/opensurfaces-segmentation-ui
  + https://github.com/lzx1413/labelImgPlus
  + https://github.com/wkentaro/labelme
  + https://github.com/labelbox/labelbox

## Results:

  + [MSRC-21](http://rodrigob.github.io/are_we_there_yet/build/semantic_labeling_datasets_results.html)
  + [Cityscapes](https://www.cityscapes-dataset.com/benchmarks/)
  + [VOC2012](http://host.robots.ox.ac.uk:8080/leaderboard/displaylb.php?challengeid=11&compid=6)

## Metrics
  + https://github.com/martinkersner/py_img_seg_eval
  
## Other lists
  + https://github.com/tangzhenyu/SemanticSegmentation_DL
  + https://github.com/nightrome/really-awesome-semantic-segmentation
  
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
  + https://github.com/yandexdataschool/YSDA_deeplearning17/blob/master/Seminar6/Seminar%206%20-%20segmentation.ipynb
  
- Cascaded-FCN
  + https://github.com/IBBM/Cascaded-FCN
  
- Keras
  + https://github.com/jocicmarko/ultrasound-nerve-segmentation
  + https://github.com/EdwardTyantov/ultrasound-nerve-segmentation
  + https://github.com/intact-project/ild-cnn
  + https://github.com/scottykwok/cervix-roi-segmentation-by-unet
  + https://github.com/lishen/end2end-all-conv
  
- Tensorflow
  + https://github.com/imatge-upc/liverseg-2017-nipsws
  
- Using Convolutional Neural Networks (CNN) for Semantic Segmentation of Breast Cancer Lesions (BRCA)
  + https://github.com/ecobost/cnn4brca
  
- Papers:
  + https://www2.warwick.ac.uk/fac/sci/dcs/people/research/csrkbb/tmi2016_ks.pdf
  + Sliding window approach
	  - http://people.idsia.ch/~juergen/nips2012.pdf
  + https://github.com/albarqouni/Deep-Learning-for-Medical-Applications#segmentation
	  
 - Data:
   - https://luna16.grand-challenge.org/
   - https://camelyon16.grand-challenge.org/
   - https://github.com/beamandrew/medical-data
  
## Satellite images segmentation

  + https://github.com/mshivaprakash/sat-seg-thesis
  + https://github.com/KGPML/Hyperspectral
  + https://github.com/lopuhin/kaggle-dstl
  + https://github.com/mitmul/ssai
  + https://github.com/mitmul/ssai-cnn
  + https://github.com/azavea/raster-vision
  + https://github.com/nshaud/DeepNetsForEO
  + https://github.com/trailbehind/DeepOSM
  
 - Data:
  	+ https://github.com/RSIA-LIESMARS-WHU/RSOD-Dataset-
	+ SpaceNet[https://spacenetchallenge.github.io/]

## Video segmentation

  + https://github.com/shelhamer/clockwork-fcn
  + https://github.com/JingchunCheng/Seg-with-SPN

## Autonomous driving

  + https://github.com/MarvinTeichmann/MultiNet
  + https://github.com/MarvinTeichmann/KittiSeg
  + https://github.com/vxy10/p5_VehicleDetection_Unet [Keras]
  + https://github.com/ndrplz/self-driving-car
  + https://github.com/mvirgo/MLND-Capstone
  + https://github.com/zhujun98/semantic_segmentation/tree/master/fcn8s_road

### Other

## Networks by framework (Older list)
- Keras
	+ https://github.com/gakarak/FCN_MSCOCO_Food_Segmentation
	+ https://github.com/abbypa/NNProject_DeepMask

- TensorFlow
	+ https://github.com/warmspringwinds/tf-image-segmentation
	
- Caffe
	+ https://github.com/xiaolonw/nips14_loc_seg_testonly
	+ https://github.com/naibaf7/caffe_neural_tool
	
- torch
	+ https://github.com/erogol/seg-torch
	+ https://github.com/phillipi/pix2pix
	
- MXNet
	+ https://github.com/itijyou/ademxapp

## Papers and Code (Older list)

- Simultaneous detection and segmentation

  + http://www.eecs.berkeley.edu/Research/Projects/CS/vision/shape/sds/
  + https://github.com/bharath272/sds_eccv2014
  
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
  + https://github.com/NVIDIA/DIGITS/tree/master/examples/semantic-segmentation
  + http://jamie.shotton.org/work/code.html 
  + https://github.com/amueller/textonboost
  
## To look at
  + https://github.com/fchollet/keras/issues/6538
  + https://github.com/warmspringwinds/tensorflow_notes
  + https://github.com/kjw0612/awesome-deep-vision#semantic-segmentation
  + https://github.com/desimone/segmentation-models
  + https://github.com/nightrome/really-awesome-semantic-segmentation
  + https://github.com/kjw0612/awesome-deep-vision#semantic-segmentation
  + http://www.it-caesar.com/list-of-contemporary-semantic-segmentation-datasets/
  + https://github.com/MichaelXin/Awesome-Caffe#23-image-segmentation
  + https://github.com/warmspringwinds/pytorch-segmentation-detection
  + https://github.com/neuropoly/axondeepseg


## Blog posts, other:

  + https://handong1587.github.io/deep_learning/2015/10/09/segmentation.html
  + http://www.andrewjanowczyk.com/efficient-pixel-wise-deep-learning-on-large-images/
  + https://devblogs.nvidia.com/parallelforall/image-segmentation-using-digits-5/
  + https://github.com/NVIDIA/DIGITS/tree/master/examples/binary-segmentation
  + https://github.com/NVIDIA/DIGITS/tree/master/examples/semantic-segmentation
  + http://blog.qure.ai/notes/semantic-segmentation-deep-learning-review
  + https://medium.com/@barvinograd1/instance-embedding-instance-segmentation-without-proposals-31946a7c53e1

