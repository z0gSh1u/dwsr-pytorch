# dwsr-pytorch

This is an **UNOFFICIAL** PyTorch implementation for DWSR - [Deep Wavelet Prediction for Image Super-resolution](http://openaccess.thecvf.com/content_cvpr_2017_workshops/w12/papers/Guo_Deep_Wavelet_Prediction_CVPR_2017_paper.pdf) . **Only 4x upsampling is provided.**

![image-20201209162704163.png](https://i.loli.net/2020/12/09/H2Dk6pSqYvuVLtF.png)

## Requirements

- PyTorch >= 1.0
- torchvision
- TensorBoard
- [LPIPS](https://github.com/richzhang/PerceptualSimilarity) (`pip install lpips`)
- tqdm
- NVIDIA GPU with CUDA
- **Not for Windows**

## Start Training

- Modify `option.py`

  - Set `train_dir` to the path to your HR images of training set. LR images aren't needed since they are generated on-the-fly.
  - Set `val_dir` to the path to your HR images of validation set. LR images aren't needed either.
  - Change other settings if you want:
    - crop_size: How big the dataset will be cropped into. Must be multiple of 4.
    - n_conv: How many conv layers between Conv1 and ConvN.
    - lr: Initial learning rate. Default 1e-2 and decays by x0.75 every 20 epochs.
    - ...

- Initialize output directory

  ```shell
  $ cd script
  $ chmod 755 *
  $ ./init_out.sh
  ```

- Trigger training

  Modify `train.sh` if you want to change which GPU to run on. Then run:

  ```shell
  $ ./train.sh
  ```

  DWSR is so lightweight that there is no need to train on multiple cards.

## After Training

As you can see, PSNR and [LPIPS](https://github.com/richzhang/PerceptualSimilarity) (Perceptual Loss using VGG) are calculated, while SSIM isn't.

- Checkpoint files are stored at `/out/ckp`

- Metric record file are stored at `/out/metric.txt`

- TensorBoard log files are stored at `/out/tb`. Run this command under `/out` folder to see the curves of PSNR and LPIPS:

  ```shell
  $ tensorboard --logdir=./tb --bind_all
  ```

## Start Testing

- Modify `option.py`

  - Set `test_dir` to the path to your LR images of test set.
  - Change `test_ckp` if you want to use another checkpoint.

- Trigger testing

  Modify `train.sh` if you want to change which GPU to run on. Then run:

  ```shell
  $ cd script
  $ ./test.sh
  ```

  Testing results are output to `/out/test_result`.

