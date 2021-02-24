# RobustDataProfiling

* We propose a generic method to improve robustness for adversarial training -- remove problematic data
* See more analyses in our paper

  > [Data Profiling for Adversarial Training: On the Ruin of Problematic Data](https://arxiv.org/abs/2102.07437)

## Install

* Install [PyTorch](http://pytorch.org/)
* Clone recursively

  ```
  git clone --recursive https://github.com/shwinshaker/LipGrow.git
  ```

## Setup

* By default, build a `./data` directory which includes the datasets
* By default, build a `./checkpoints` directory to save the training output

## Training
```
./launch.sh config.yaml
```


## Estimate problematic rank

* Set `exTrack` to `True` in `config.yaml`
* Conduct adversarial training
* `count_wrong.npy` will be generated in the checkpoint directory, which records the number of epochs that every example is misclassified under adversarial attacks during training (Learning instability)
* Ranking the examples based on the learning instability yields the problematic rank

  ```
  import numpy as np

  def rank(arr):
      return np.array(arr).argsort().argsort()

  counts = np.load('checkpoints/<Your_Checkpoint_Name>/count_wrong.npy')
  prob_rank = rank(counts) / len(counts)
  ```

* Repeat the above process to get a more accurate estimation

* An estimation of the normalized problematic rank averaged by 10 repeated experiments with random intialization (PGD-10 training for pre-activation ResNet-18) has been pre-calculated in `prob_rank.npy`

## Adversarial training on friendly data only

* Select the top k friendly examples based on problematic rank estimation

  ```
  import numpy as np
  prob_rank = np.load('prob_rank.npy')
  size = 40000
  idx_friend = prob_rank.argsort()[:size]
  with open('data_subsets/id_friend_%i.npy' % size, 'wb') as f:
      np.save(f, idx_friend, allow_pickle=True)
  ```

* Set `train_subset_path` to the path of this index file to conduct adversarial training on this subset only
* A index subset of the top 25000 friendly examples (class-balanced sampling) has been pre-calculated in `data_subsets/id_friend_25000_balance.npy`

## Citation

If you find our algorithm helpful, consider citing our paper

```
@article{Dong2021DataPF,
  title={Data Profiling for Adversarial Training: On the Ruin of Problematic Data},
  author={Chengyu Dong and Liyuan Liu and Jingbo Shang},
  journal={ArXiv},
  year={2021},
  volume={abs/2102.07437}
}
```












