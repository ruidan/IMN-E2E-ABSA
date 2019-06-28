# IMN-E2E-ABSA
Code and dataset for ACL2019 [[paper]](https://arxiv.org/abs/1906.06906) ‘‘An Interactive Multi-Task Learning Network for End-to-End Aspect-Based Sentiment Analysis’’. 

## Data
The preprocessed aspect-level datasets can be downloaded at [[Download]](https://drive.google.com/open?id=1zp9fykp6_zPBBEiJp9zc8LKmXGfi2zDa), and the document-level datasets can be downloaded at [[Download]](https://drive.google.com/file/d/1gOa8p3O2z4caftc4zmhKobZP5Ujax4Sf/view?usp=sharing). The zip files should be decompressed and put in the main folder.

Glove vectors (glove.840B.300d) are used for initialization of general word embeddings. Pre-trained word embeddings from [[Hu et al.]](https://github.com/howardhsu/DE-CNN) are used for initialization of domain-specific word embeddings. To save the time of reading large files, subsets of the pre-trained vectors are extracted which only contain the words that belong to the vocab for each dataset. You can download the extracted files for glove vectors and domain-specific vectors at [[Download]](https://drive.google.com/file/d/1cB785im5XNSRFhZiiVp2AwFAOCRoJLp3/view?usp=sharing) and [[Download]](https://drive.google.com/open?id=1QXHdnqKhiBizmUYokavzkvCENN-lrpPA) respectively. The zip files should be decompressed and put in the main folder. 

## Training and evaluation
Excute the command below for training and evaluating IMN.
```
CUDA_VISIBLE_DEVICES="0" python train.py --domain $domain
```
where *$domain* in ['res', 'lt', 'res_15'] denotes the corresponding aspect-level dataset. You can find all arguments defined in train.py with default values used in our experiments. At the end of each epoch, evaluation results on validation and test sets are saved in *out.log*. The test results at the epoch where the model achieves the best performance on validation set are recorded. 

## Dependencies
* Python 2.7
* Keras 2.2.4
* tensorflow 1.4.1
* numpy 1.13.3

## Cite
If you use the code, please cite the following paper:
```
@InProceedings{he_acl2019,
  author    = {He, Ruidan  and  Lee, Wee Sun  and  Ng, Hwee Tou  and  Dahlmeier, Daniel},
  title     = {An Interactive Multi-Task Learning Network for End-to-End Aspect-Based Sentiment Analysis},
  booktitle = {Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics},
  publisher = {Association for Computational Linguistics}
}
```

