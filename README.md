# Rumor-Detection-based-on-the-Cascading-Behavior
This repository contains code for our paper "Repost or Reply？Rumor Detection based on Different Cascading Behaviors on Social Media" By Yue Yuan, Na Pang, Yuqi Zhang and Kan Liu. 

The trained models and Twitter and Weibo rumor datasets of this paper can be accessed here https://drive.google.com/drive/folders/109GpzxLnZdmyU9ny0e3Tc77RGXqrGy4m?usp=sharing.

# Requirements
python==3.6.5  
tensorflow==1.8.0  
gensim==3.8.3  
numpy==1.19.
# RDCB Models & Ablation Models

# How to use
## Dataset
The raw dataset is collected and processed by Yue Yuan and Na Pang and can be downloaded from [here](https://drive.google.com/drive/folders/109GpzxLnZdmyU9ny0e3Tc77RGXqrGy4m?usp=sharing).  
The raw datasets have been processed according to Section 3.2.1 Data prepraration and Section 3.2.2 Data processing in our paper and is stored in Data.rar.  
Therefore, you can use the data in Data.rar directly without performing the data prepraration process described in our paper. 
To use the data, unzip the Data.rar to the directory of this repository and enter corresponding dataset setting in console when running main.py.
For Twitter rumor Dataset:
--twitter --sourceposts     # source posts only-twitter rumor dataset combination
--twitter --sourceposts-replies     # source posts and replies-twitter rumor dataset combination
--twitter --sourceposts-reposts     # source posts and reposts-twitter rumor dataset combination
--twitter --sourceposts-replies-reposts     # source posts, replies, and reposts-twitter rumor dataset combination
For Weibo rumor Dataset:
--weibo --sourceposts     # source posts only-weibo rumor dataset combination
--weibo --sourceposts-replies     # source posts and replies-weibo rumor dataset combination
--weibo --sourceposts-reposts     # source posts and reposts-weibo rumor dataset combination
--weibo --sourceposts-replies-reposts     # source posts, replies, and reposts-weibo rumor dataset combination
## Model
The trained network models are stored in Models.rar and can be used directly for reproducing the experimental results in the paper.
To load the trained models, unzip the Models.rar to the directory of this repository and enter corresponding model settings in console when running main.py.
For Training:
--dbcn --train YOURPATH     # train the dbcn model, your trained model is defaultly saved in YOURPATH 
--bilstm --train YOURPATH     # train the bilstm model, your trained model is defaultly saved in YOURPATH 
--dbcnbilstm --train YOURPATH     # train the dbcnbilstm model, your trained model is defaultly saved in YOURPATH 
For Testing:
--dbcn --test --     #  test the dbcn model by our trained model   
--bilstm --test --     # test the bilstm model by our trained model  
--dbcnbilstm --test --     # test the dbcnbilstm model by our trained model  

Still updating！
Please keep tracking this repository!
