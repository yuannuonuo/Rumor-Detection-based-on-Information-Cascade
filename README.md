# Rumor-Detection-based-on-Information-Cascade
This repository contains source code and datasets for our paper "Which Cascade is More Decisive in Rumor Detection on Social Media: Based on Comparison Between Repost and Reply Sequences" By Yue Yuan, Na Pang, Yuqi Zhang and Kan Liu. 

The trained models and two rumor datasets of this paper can be accessed here https://drive.google.com/drive/folders/109GpzxLnZdmyU9ny0e3Tc77RGXqrGy4m?usp=sharing.

# Requirements
python==3.6.5  
tensorflow==1.8.0  
gensim==3.8.3  
numpy==1.19.

# CSRD Model & Attention-weight Model & Ablation Models
## CSRD Models
The overall structure of the CSRD model is shown in Figure 1 that consists of four components: an input part, a dilated convolution part, a Bi-LSTM part, and an output part.

![Overall structure for CSRD Model](./Figure1.pdf)

The dilated convolution part is intended to extract local features from the paragraph embeddings generated during the data processing phase. 
The original dilated convolution extracts meaningful text features through convolution on 2-D paragraph embeddings in the text splicing direction; however, convolution in the non-splicing direction loses spatial information of paragraph embedding. 
Hence, the original dilated convolution cannot meet the needs of local feature extraction from paragraph embeddings. 
We modify the convolution pattern of the original dilated convolution such that it can take paragraph embeddings as input and efficiently extract the local features from the input by stacking the dilated convolution layer. 
A detailed example of the this modified dilated convolution is shown in Figure 2.
![Modified dilated convolution](./Figure2.pdf)

The Bi-LSTM part is intended to extract global text features from paragraph embeddings of repost or reply sequences. 
By capturing both the backward and forward directions of information from the input embedding sequence, the Bi-LSTM neural network extracts global sequence features from the input.

For the output part, the outputs of the dilated convolution part and Bi-LSTM part are concatenated and sent to the softmax layer for classification.
The softmax layer is built through a fully connected layer, and outputs the probability of a post being a rumor.
## Attention-weight Model
The attention-weight model here aims to measure the weights of local and global features extracted by the dilated convolution and Bi-LSTM parts to determine the effect of local and global features extracted by the CSRD model. Our attention-weight model originates from the attention-based Bi-LSTM network proposed by Zhou et al. (Zhou et al., 2016), as demonstrated in Figure 3.
![Attention-weight model for CSRD](./Figure3.pdf)
## Ablation Models
For the first ablation model (the CSRD without Dilated Convolution Part), the dilated convolutional neural network was removed from the main model so that the effect of local features of reposts or replies in rumor detection could be revealed. (Figure 4)
![CSRD Model Without Dilated Convolution Part](./Figure4.pdf)

For the second ablation model (the CSRD without Bi-LSTM Part), the Bi-LSTM part of the CSRD model was removed from the main model to validate the global features of the reposts or replies in rumor detection. (Figure 5)
![CSRD Model Without Bi-LSTM Part](./Figure5.pdf)

# How to use
## Dataset
The processed datasets can be downloaded from [here](https://drive.google.com/drive/folders/109GpzxLnZdmyU9ny0e3Tc77RGXqrGy4m?usp=sharing).  
The raw datasets have been processed according to Section 3.2.1 Data prepraration and Section 3.2.2 Data processing in our paper and is stored in Data.rar.  
Therefore, you can use the data in Data.rar directly without performing the data prepraration process described in our paper. 
To use the data, unzip the Data.rar to the directory of this repository and enter corresponding dataset setting in console when running main.py.
For Twitter rumor Dataset:

--twitter --sourceposts     # source posts -twitter rumor dataset combination

--twitter --sourceposts-replies     # source posts and replies-twitter rumor dataset combination

--twitter --sourceposts-reposts     # source posts and reposts-twitter rumor dataset combination

--twitter --sourceposts-replies-reposts     # source posts, replies, and reposts-twitter rumor dataset combination

For Weibo rumor Dataset:

--weibo --sourceposts     # source posts -weibo rumor dataset combination

--weibo --sourceposts-replies     # source posts and replies-weibo rumor dataset combination

--weibo --sourceposts-reposts     # source posts and reposts-weibo rumor dataset combination

--weibo --sourceposts-replies-reposts     # source posts, replies, and reposts-weibo rumor dataset combination

## Model
The trained network models are stored in Models.rar and can be used directly for reproducing the experimental results in the paper.
To load the trained models, unzip the Models.rar to the directory of this repository and enter corresponding model settings in console when running main.py.
For Training:

--dbcn --train YOURPATH     # train the second ablation model, your trained model is defaultly saved in YOURPATH 

--bilstm --train YOURPATH     # train the first ablation model, your trained model is defaultly saved in YOURPATH 

--dbcnbilstm --train YOURPATH     # train the CSRD model, your trained model is defaultly saved in YOURPATH 

For Testing:

--dbcn --test --     #  test the second ablation model by our trained model   

--bilstm --test --     # test the first ablation model by our trained model  

--dbcnbilstm --test --     # test the CSRD model by our trained model  

## Example Command
Example 1. If you want to train the CSRD model by yourself using the twitter rumor dataset- combination of source posts and replies, you should enter the following command on the console:

cd YOUR_REPOSITORY_PATH

python main.py --twitter --sourceposts-replies --dbcnbilstm --train YOURPATH

Following the above commands, your self-trained models will be saved in YOURPATH.

Example 2. If you want to testin the trained CSRD model provided by the author using the twitter rumor dataset-combination of source posts and replies, you should enter the following command on the console:

cd YOUR_REPOSITORY_PATH

python main.py --twitter --sourceposts-replies --dbcnbilstm --test --

Following the above commands, you will reproduce the detection results in the paper.


Still updating！
Please keep tracking this repository!
