## BASRec
Official source code for AAAI 2025 paper: [Augmenting Sequential Recommendation with Balanced Relevance and Diversity](https://arxiv.org/abs/2412.08300)

## Run the Code

Go to the `src` folder in the `GRU4Rec` and `SASRec` or directory, then run the following commands. 

To save time, we provide the original pre-trained model, which is the first stage in the paper. The model can be loaded by running the commands and moving to the second stage to further improve the model performance using our method.


For `GRU4Rec`: 
```
python main.py --data_name=Beauty --load_pretrain --model_idx=1 --dropout_prob=0.2 --rate_min=0.2 --rate_max=0.51
python main.py --data_name=Yelp --load_pretrain --model_idx=1 --epochs=100 --start_valid=50 --dropout_prob=0.5 
python main.py --data_name=Home --load_pretrain --model_idx=1 --dropout_prob=0.2 --rate_min=0.3 --rate_max=0.81
python main.py --data_name=Sports_and_Outdoors --load_pretrain --model_idx=1 --dropout_prob=0.2  --n_pairs=2 --n_whole_level=3
```

For `SASRec`:
```
python main.py --data_name=Beauty --model_idx=1 --load_pretrain --beta=0.4 --attention_probs_dropout_prob=0.1 --hidden_dropout_prob=0.1 --n_pairs=2 --n_whole_level=3 --rate_min=0.2 --rate_max=0.71
python main.py --data_name=Sports_and_Outdoors --model_idx=1  --load_pretrain --attention_probs_dropout_prob=0.1 --hidden_dropout_prob=0.1 --n_pairs=2 --n_whole_level=3 --rate_min=0.1 --rate_max=0.81
python main.py --data_name=Yelp --model_idx=1 --load_pretrain --epochs=100 --start_valid=50 --attention_probs_dropout_prob=0.1 --hidden_dropout_prob=0.1 --n_pairs=2 --n_whole_level=3 --beta=0.5
python main.py --data_name=Home --model_idx=1 --load_pretrain --epochs=100 --start_valid=50  --attention_probs_dropout_prob=0.05 --hidden_dropout_prob=0.05 --n_pairs=2 --n_whole_level=3 --beta=0.5 --rate_min=0.3 --rate_max=0.71
```


## Log Files
We also provide log files on these four datasets in the `src/output` directory.


## Acknowledgement

Some models are implemented based on [CoSeRec](https://github.com/YChen1993/CoSeRec) and [RecBole](https://github.com/RUCAIBox/RecBole).

Thanks for providing efficient implementation.


## Reference

Please cite our paper if you use this code.
```
@article{dang2024augmenting,
  title={Augmenting Sequential Recommendation with Balanced Relevance and Diversity},
  author={Dang, Yizhou and Zhang, Jiahui and Liu, Yuting and Yang, Enneng and Liang, Yuliang and Guo, Guibing and Zhao, Jianzhe and Wang, Xingwei},
  journal={arXiv preprint arXiv:2412.08300},
  year={2024}
}
```
