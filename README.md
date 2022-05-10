# Continual-Plug-and-Adapt-for-CuQA 
This includes an implementation of ["Plug-and-Play Adaptation for Continuously-updated QA"](https://arxiv.org/abs/2204.12785) 

### Executing program
To train the model with source Knowledge, please use the following command
```
python pretrain.py --train_path $train_path --dev_path $dev_path
```
Here are some arguments(included but not all) which might be useful. 
* `--n_gpus`: the number of gpus you will use for training
* `--dataset`: type of dataset, like `ZsRE` or `nq`
* `--train_path`: path of train dataset
* `--dev_path`: path of validation dataset
* `--init_checkpoint`: Checkpoint if you want to train model from checkpoint
* `--validation_freq`: The frequency of Validation(epoch)

To update the model with target Knowledge, please use the following command
```
python update.py --checkpoint $model_checkpoint --train_path $train_path --dev_path $dev_path --adapter $adapter --freeze_orig_param $params
```
Here are some arguments(included but not all) which might be useful.

* `--checkpoint`: parameters of original model which will be freeze while updating
* `--adapter`: type of adapter which will be exploited when updating knowledge
* `--freeze_orig_param`: parameters of original model which will be freeze while updating
* `--ours_threshold`: threshold when judging whether the data is from source or not

To evaluate the model, please use the following command

```
python eval.py --checkpoint $model_checkpoint --dev_path $dev_path --adapter $adapter
```
* `--checkpoint`: parameters of original model which will be freeze while updating
* `--adapter`: type of adapter which was exploited when updating or pretraining
* `--ours_threshold`: threshold when judging whether the data is from source or not

## Version History

* 0.1
    * Initial Release

## Citation
If you find this repo useful, please cite our preprint:

```
@article{lee2022plug,
  title={Plug-and-Play Adaptation for Continuously-updated QA},
  author={Lee, Kyungjae and Han, Wookje and Hwang, Seung-won and Lee, Hwaran and Park, Joonsuk and Lee, Sang-Woo},
  journal={arXiv preprint arXiv:2204.12785},
  year={2022}
}
```

## License
```
Copyright
Copyright 2022-present SNU-NAVER Hyperscale AI Center

Acknowledgement
This research was supported by SNU-NAVER Hyperscale AI Center, and IITP grants funded by the Korea government (MSIT) [2021-0-02068 SNU AIHub, IITP-2022-2020-0-01789].
```
