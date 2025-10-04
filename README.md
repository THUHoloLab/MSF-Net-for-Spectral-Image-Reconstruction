# MSF-Net for Spectral Image Reconstruction

A PyTorch implementation of  
**"A dual-camera coded aperture snapshot spectral imager using a reflective mask"**

---

## Requirements
- NVIDIA GPU  
- Linux OS  
- Python 2.7  
- PyTorch with CUDA Toolkit and cuDNN  

---

## Usage
To train or test the model, please refer to the provided scripts in the corresponding folders.

---

## MSF_net_DD_CASSI
Reconstructs spectral images from the **DD-CASSI system** using a single coded measurement.

- `args.py`: defines hyperparameters (learning rate, batch size, save frequency, etc.)  
- `get_coded_aperture.py`: generates the coded aperture mask matrix  
- `get_metrics.py`: computes evaluation metrics (PSNR, SSIM, SAM)  
- `load_dataset.py`: loads training and testing datasets  
- `My_model.py`: defines the MSF-Net model architecture  
- `reconstruct_test.py`: performs spectral image reconstruction with a trained model  
- `train.py`: main training script  
  - Can be run with **PyCharm** or **VS Code**  
  - Automatically creates an **`image_save/`** folder to store reconstructed images during training  
  - Automatically creates a **`model_save/`** folder to store trained models  

---

## MSF_net_RDC_CASSI
Reconstructs spectral images from the **RDC-CASSI system** using two measurements:  
1. A grayscale measurement  
2. An encoded measurement from the CASSI system  

> The file names and functions are the same as in **MSF_net_DD_CASSI**.

---

## Datasets

### Test Dataset
Test datasets are stored as `.mat` files.  
The dataset we used is available at:  
[https://cloud.tsinghua.edu.cn/d/e8c7ea9591f14049b153/](https://cloud.tsinghua.edu.cn/d/e8c7ea9591f14049b153/)

### Training Dataset
Training datasets are stored as `.mat` files.  
The dataset we used is also available at:  
[https://cloud.tsinghua.edu.cn/d/e8c7ea9591f14049b153/](https://cloud.tsinghua.edu.cn/d/e8c7ea9591f14049b153/)  

All training and test data are processed from the public datasets:  
- [CAVE Multispectral Dataset](https://cave.cs.columbia.edu/repository/Multispectral)  
- [KAIST Hyperspectral Dataset](https://vclab.kaist.ac.kr/siggraphasia2017p1/)  

---

## Authors
- **Xinyu Liu** (liuxinyu@mail.tsinghua.edu.cn)  
- **Liangcai Cao** (clc@tsinghua.edu.cn)  
