# Perceptual Depth Super-Resolution
This is the official repository for the paper [Perceptual Deep Depth Super-Resolution](http://openaccess.thecvf.com/content_ICCV_2019/html/Voynov_Perceptual_Deep_Depth_Super-Resolution_ICCV_2019_paper.html).
It contains trained MSG-V models for x4 and x8 super-resolution and IPython notebook with a usage example.

[[paper]](http://openaccess.thecvf.com/content_ICCV_2019/papers/Voynov_Perceptual_Deep_Depth_Super-Resolution_ICCV_2019_paper.pdf) [[supp]](http://openaccess.thecvf.com/content_ICCV_2019/supplemental/Voynov_Perceptual_Deep_Depth_ICCV_2019_supplemental.pdf) [[project page]](http://adase.group/3ddl/projects/perceptual-depth-sr)

## Installation
To run the code you will need `python3.7` and the packages from `environment.yml`.
All of them can be installed via [`conda`](https://docs.conda.io/en/latest/miniconda.html) with 
```bash
conda env create -f environment.yml
```

Alternatively, you can build an [Nvdia-Docker](https://github.com/NVIDIA/nvidia-docker) image with all required dependencies using the provided `Dockerfile`:
```bash
git clone https://github.com/voyleg/perceptual-depth-sr
cd perceptual-depth-sr
docker build -t perceptual-depth-sr .
```
and run Jupyter in the container
```bash
nvidia-docker run --rm -it -p 8888:8888 --mount type=bind,source=$(pwd),target=/code perceptual-depth-sr bash -c 'cd /code && jupyter notebook --ip="*" --no-browser --allow-root'
```

# Citation
```
@inproceedings{voynov2019perceptual,
  title={Perceptual deep depth super-resolution},
  author={Voynov, Oleg and Artemov, Alexey and Egiazarian, Vage and Notchenko, Alexander and Bobrovskikh, Gleb and Burnaev, Evgeny and Zorin, Denis},
  booktitle={Proceedings of the IEEE International Conference on Computer Vision},
  pages={5653--5663},
  year={2019}
}
```
