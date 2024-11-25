<div align="center">
    <h1>
    STAR: Spatial-Temporal Augmentation with Text-to-Video Models for Real-World Video Super-Resolution
    </h1>
    <div>
        <a href='https://github.com/CSRuiXie' target='_blank'>Rui Xie<sup>1*</sup></a>,&emsp;
        <a href='https://github.com/yhliu04' target='_blank'>Yinhong Liu<sup>1*</sup></a>,&emsp;
        <a href='https://scholar.google.com/citations?user=Uhp3JKgAAAAJ&hl=zh-CN&oi=sra' target='_blank'>Chen Zhao<sup>1</sup></a>,&emsp;
        <a href='https://scholar.google.com/citations?hl=zh-CN&user=yWq1Fd4AAAAJ' target='_blank'>Penghao Zhou<sup>2</sup></a>,&emsp;
        <a href='https://scholar.google.com/citations?hl=zh-CN&user=Ds5wwRoAAAAJ' target='_blank'>Zhenheng Yang<sup>2</sup></a><br>
        <a href='https://scholar.google.com/citations?hl=zh-CN&user=w03CHFwAAAAJ' target='_blank'>Jun Zhou<sup>3</sup></a>,&emsp;
        <a href='https://cszn.github.io/' target='_blank'>Kai Zhang<sup>1</sup></a>,&emsp;
        <a href='https://jessezhang92.github.io/' target='_blank'>Zhenyu Zhang<sup>1</sup></a>,&emsp;
        <a href='https://scholar.google.com.hk/citations?user=6CIDtZQAAAAJ&hl=zh-CN' target='_blank'>Jian Yang<sup>1</sup></a>,&emsp;
        <a href='https://tyshiwo.github.io/index.html' target='_blank'>Ying Tai<sup>1&#8224</sup></a>
    </div>
    <div>
        <sup>1</sup>Nanjing University,&emsp;<sup>2</sup>ByteDance,&emsp; <sup>3</sup>Southwest University
    </div>
    <div>
        <h4 align="center">
            <a href="https://nju-pcalab.github.io/projects/STAR" target='_blank'>
                <img src="https://img.shields.io/badge/🌟-Project%20Page-blue">
            </a>
            <a href="https://arxiv.org/abs/2407.07667" target='_blank'>
                <img src="https://img.shields.io/badge/arXiv-2312.06640-b31b1b.svg">
            </a>
            <a href="https://youtu.be/hx0zrql-SrU" target='_blank'>
                <img src="https://img.shields.io/badge/Demo%20Video-%23FF0000.svg?logo=YouTube&logoColor=white">
            </a>
        </h4>
    </div>
</div>


### 🔆 Updates
- **2024.12.01**  The pretrained STAR model (I2VGen-XL version) and inference code have been released.


## 🔎 Method Overview
![STAR](assets/overview.png)


## 📷 Results Display
[<img src="figs/flower.png" height="320px"/>](https://imgsli.com/MjUyNTc5) [<img src="figs/building.png" height="320px"/>](https://imgsli.com/MjUyNTkx) 
[<img src="figs/nature.png" height="320px"/>](https://imgsli.com/MjUyNTgx) [<img src="figs/human.png" height="320px"/>](https://imgsli.com/MjUyNTky)



![STAR](assets/real_world.png)


## ⚙️ Dependencies and Installation
```
## git clone this repository
git clone https://github.com/NJU-PCALab/STAR.git
cd STAR

## create an environment
conda create -n star python=3.10
conda activate star
pip install -r requirements.txt
sudo apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
```

## 🚀 Inference
#### Step 1: Download the pretrained model STAR from [HuggingFace](https://huggingface.co/stabilityai/stable-diffusion-2-base).
We provide two verisions, `heavy_deg.pt` for heavy degraded videos and `light_deg.pt` for light degraded videos (e.g., the low-resolution video downloaded from Bilibili).

You can put the weight into `pretrained_weight/`.


#### Step 2: Prepare testing data
You can put the testing videos in the `input/video/`.

As for the prompt, there are three options: 1. No prompt. 2. Automatically generate a prompt [using Pllava](https://github.com/hpcaitech/Open-Sora/tree/main/tools/caption#pllava-captioning). 3. Manually write the prompt. You can put the txt file in the `input/text/`.


#### Step 3: Change the path
You need to change the paths in `inference_sr.sh` to your local corresponding paths, including `video_folder_path`, `txt_file_path`, `model_path`, and `save_dir`.


#### Step 4: Running inference command
```
bash video_super_resolution/scripts/inference_sr.sh
```


## ❤️ Acknowledgments
This project is based on [I2VGen-XL](https://github.com/ali-vilab/VGen), [VEnhancer](https://github.com/Vchitect/VEnhancer) and [CogVideoX](https://github.com/THUDM/CogVideo). Thanks for their awesome works.


## 📧 Contact
If you have any inquiries, please don't hesitate to reach out via email at `ruixie0097@gmail.com`


## 🎓Citations
If our project helps your research or work, please consider citing our paper:

```
@misc{xie2024addsr,
      title={AddSR: Accelerating Diffusion-based Blind Super-Resolution with Adversarial Diffusion Distillation}, 
      author={Rui Xie and Ying Tai and Kai Zhang and Zhenyu Zhang and Jun Zhou and Jian Yang},
      year={2024},
      eprint={2404.01717},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

## :page_facing_up: License
This project is distributed under the terms of the [Apache 2.0 license](LICENSE). Since AddSR is based on [SeeSR](https://github.com/cswry/SeeSR?tab=readme-ov-file#-license), [StyleGAN-T](https://github.com/autonomousvision/stylegan-t#license), and [ADD](https://github.com/Stability-AI/generative-models/tree/main/model_licenses), users must also follow their licenses to use this project.
