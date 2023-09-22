# α-MDF: An Attention-based Multimodal Differentiable Filter for Robot State Estimation

<p align="center">
<img src = "img/teaser.png" width ="800" />
</p>

Differentiable Filters are recursive Bayesian estimators that derive the state transition and measurement models from data alone. Their data-driven nature eschews the need for explicit analytical models, while remaining algorithmic components of the filtering process intact. As a result, the gain mechanism -- a critical component of the filtering process -- remains non-differentiable and cannot be adjusted to the specific nature of the task or context. In this paper, we propose an attention-based Multimodal Differentiable Filter ($\alpha$-MDF) which utilizes modern attention mechanisms to learn multimodal latent representations. Unlike previous differentiable filter frameworks, $\alpha$-MDF substitutes the traditional gain, e.g., the Kalman gain, with a neural attention mechanism. The approach generates specialized, context-dependent gains that can effectively combine multiple input modalities and observed variables. We validate $\alpha$-MDF on a diverse set of robot state estimation tasks in real world and simulation. Our results show $\alpha$-MDF achieves significant reductions in state estimation errors, demonstrating nearly 4-fold improvements compared to state-of-the-art sensor fusion strategies for rigid body robots. Additionally, the $\alpha$-MDF consistently outperforms differentiable filter baselines by up to 45% in soft robotics tasks.
<p align="center">
<img src = "img/framework.gif" width ="800" />
</p>

## Getting Started
We provide implementation using `Pytorch`. Clone the repo `git clone https://github.com/ir-lab/alpha-MDF.git` and then there are two options for running the code.

#### 1. Python Scripts

Intall [PyTorch](https://pytorch.org/get-started/previous-versions/) and then set up the environment using `pip install -r requirements.txt`. Make sure to have corresponding libraries and dependencies installed on your local environment, i.e., we use PyTorch 1.8.0 with cuda11.1.

For training or testing, Go to `./UR5` and then Run 

```
python train.py --config ./config/xxx.yaml
```

#### 2. docker workflow
Edit the `conf.sh` file to set the environment variables used to start the docker 
containers. 

```
IMAGE_TAG=  # unique tag to be used for the docker image.
CONTAINER_NAME=UR5  # name of the docker container.
DATASET_PATH=/home/xiao/datasets/  # Dataset path on the host machine.
CUDA_VISIBLE_DEVICES=0  # comma-separated list of GPU's to set visible.
```
Build the docker image by running `./build.sh`.


##### Training or testing
Create or a modify a yaml file found in `./latent_space/config/xxx.yaml`, and set the mode parameter to perform the training or testing routine. 

```
mode:
    mode: 'train'  # 'train' | 'test'
```

Run the training and test script using the bash file `./run_filter.sh $CONFIG_FILE` 
where `$CONFIG_FILE` is the path to the config file.  
```shell
`./run_filter.sh ./config/xxx.yaml`
```
View the logs with `docker logs -f $CONTAINER_NAME`


##### Tensorboard

Use the docker logs to copy the tensorboard link to a browser

```docker logs -f $CONTAINER_NAME-tensorboard```


## Results
We conduct a series of experiments to evaluate the efficacy of the $\alpha$-MDF framework. Specifically, we aim to answer the following questions:
1. Can the $\alpha$-MDF framework generalize across various tasks? 
2. To what extent does the new filtering mechanism improve state tracking performance when compared to the current state-of-the-art?
3. How does the use of multiple modalities compare to a subset of modalities for state estimation with differentiable filters?

### UR5 multimodal manipulation task
We use $\alpha$-MDF for monitoring the state of a UR5 robot during tabletop arrangement tasks.
<p align="center">
<img src = "img/result.gif" width ="800" />
</p>

**Left**: manipulation in a simulated environment with modalities [RGB, Depth, Joints]  with The attention maps indicate the attention weights assigned to each modality during model inference. In the visualization, regions in **blue** correspond to low attention values, while those in **red** indicate high attention values.
**Right**: real-time predicted joint angle trajectories.

### Soft robot task
This experiment involves implementing the $\alpha$-MDF to model the dynamics of a soft robot system, especially Tensegrity robot.
<p align="center">
<img src = "img/soft_robot.png" width ="800" />
</p>

Predicted end-effector (EE) positions and quaternion vectors **q** in the soft robot modeling task. The **top** row displays the actual robot posture at the corresponding time, with the orange circle indicating the EE positions, which are not included in the RGB modality input.

## Datasets
#### KITTI_dataset
https://www.cvlibs.net/datasets/kitti/eval_odometry.php
#### UR5_dataset
sim2real UR5 Dataset https://www.dropbox.com/sh/qgd3hc9iu1tb1cd/AABDfyYLyGpso605-19kbOhCa?dl=0 (Yifan: yzhou298@asu.edu)
#### Tensegrity_dataset
The Dataset is available upon request. (Dr. Ikemoto: ikemoto@brain.kyutech.ac.jp)

## Citation
* Please cite the paper if you used any materials from this repo, Thanks.
```
@inproceedings{liu2023alpha,
  title = {$\alpha$-MDF: An Attention-based Multimodal Differentiable Filter for Robot State Estimation},
  author = {Liu, Xiao and Zhou, Yifan and Ikemoto, Shuhei and Amor, Heni Ben},
  booktitle = {7th Annual Conference on Robot Learning},
  year = {2023}
}
```
