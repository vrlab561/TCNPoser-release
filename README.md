## Estimating Torso Direction Based on a Head-Mounted Display and Its Controllers for Steering during Virtual Locomotion


Abstract
---
> Reliable estimation of torso direction is essential for virtual locomotion as it controls the locomotion direction in virtual environments. Existing approaches rely on external sensors to track torso direction or use the poses and/or movements from the head, hands, or feet to control the direction of travel. In addition, deep learning methods based on a head-mounted display (HMD) and its controllers were proposed for torso direction estimation but are unable to produce continuous estimates or that the estimation errors were not systematically evaluated. On the other hand, 3-D full-body human motion generation methods based on an HMD and its controllers are capable of estimating torso direction but this capability was largely ignored for its use in virtual steering. We proposed a new method called TCNPoser that can perform torso direction estimation and generate 3-D full-body human motion. Through offline evaluation and a user study, we show that the proposed method achieves the state-of-the-art (SOTA) performance for real-time torso direction estimation.

## Requirements
- python >= 3.10
- pytorch >= 2.1.2
- cuda >= 11.8
- numpy >= 1.26.4
- [human_body_prior](https://github.com/nghorbani/human_body_prior)

## Preparation
### AMASS
1. Download the [AMASS](https://amass.is.tue.mpg.de/) dataset and place it in the `./data/AMASS` directory of this repo.
2. Download the required body models [SMPL+H (AMASS Version)](http://mano.is.tue.mpg.de/) and [DMPLs](http://smpl.is.tue.mpg.de). Place them in `./body_models` directory of this repo. 
3. Run  `./prepare_data.py` to preprocess the input data. The data split for training and testing data under Protocol 1 in our paper is stored under the folder `./prepare_data/data_split` (directly copied from [AvatarPoser](https://github.com/eth-siplab/AvatarPoser)).

```
python ./prepare_data.py --support_dir ./body_models/ --root_dir ./data/AMASS/ --save_dir [path_to_save, ./prepared_data/ default]
```
### human_body_prior
Download the [human_body_prior](https://github.com/nghorbani/human_body_prior/tree/master/src) lib and extract this package in this repo. 

The organization of the repo after preparation:
```
TCNPoser
├── body_models/
├──── dmpls/
├──── smplh/
├── data/
├──── AMASS/
├── human_body_prior/
├──── body_model/
├──── ...
├── prepare_data/
├──── data_split/
├── prepared_data/
├──── BioMotionLab_NTroje/
├──── CMU/
├──── MPI_HDM05/
└── ...
```

##  Training
Modify the dataset_path in `./options/train_config.yaml` to your `[path_to_save]`.
```
python train.py --config ./options/train_config.yaml
```

## Evaluation
Modify the resume_model path in `./options/test_config.yaml`.
```
python test.py --config ./options/test_config.yaml
```

## Citation
    @article{zhao_estimating_2025,
	title = {Estimating {Torso} {Direction} {Based} on a {Head}-{Mounted} {Display} and {Its} {Controllers} for {Steering} during {Virtual} {Locomotion}},
	doi = {10.1002/cav.70087},
	journal = {Computer Animation and Virtual Worlds},
	author = {Zhao, Jingbo and Ding, Mengyang and Guo, Zihao and Shao, Mingjun},
	year = {2025},
}



## License
Distributed under the MIT License. See `LICENSE` for more information.

## Acknowledgements
This code was implemented based on [HMDPoser](https://github.com/Pico-AI-Team/HMD-Poser). It can also be recursively traced back to [AvatarPoser](https://github.com/eth-siplab/AvatarPoser) and other open-source projects. We thank all authors for their efforts.