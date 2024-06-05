
---
### Prerequisites

- This project is implemented in Pytorch (>1.8). Thus please install Pytorch first.

- ctcdecode==0.4 [[parlance/ctcdecode]](https://github.com/parlance/ctcdecode)，for beam search decode.


### Data Preparation

1. Download the RWTH-PHOENIX-Weather 2014T Dataset [[download link]](https://www-i6.informatik.rwthaachen.de/~koller/RWTH-PHOENIX-2014-T/). 

2. Download the CSL-Daily Dataset [[download link]](http://home.ustc.edu.cn/~zhouh156/dataset/csl-daily/). 

3. Download the CSL-Daily Dataset [[download link]](https://github.com/MLSLT/SP-10). 

3. Download the QSL Dataset [[download link]](https://github.com/glq-1992/QSL). 


### Training

The priorities of configuration files are: command line > config file > default values of argparse. To train the GBT model on QSL dataset, run the command below:

`python main_TJU_QA_bert_sign_text_grpah_slt.py --device AVAILABLE_GPUS`




