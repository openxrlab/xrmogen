# Please build this dockerfile with nvidia-container-runtime
# Otherwise you will need to re-install mmcv-full
FROM nvidia/cuda:10.1-devel-ubuntu18.04

# Install apt packages
RUN apt-get update && \
    apt-get install -y \
        wget git vim \
        libblas-dev liblapack-dev libatlas-base-dev\
    && \
    apt-get autoclean

# Install miniconda
RUN wget -q \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh

# Update in bashrc
RUN echo "source /root/miniconda3/etc/profile.d/conda.sh" >> /root/.bashrc && \
    echo "conda deactivate" >> /root/.bashrc

# Prepare pytorch env
RUN . /root/miniconda3/etc/profile.d/conda.sh && \
    conda create -n xrmogen python=3.8 -y && \
    conda activate xrmogen && \
    conda install ffmpeg -y && \
    conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=10.1 -c pytorch && \
    conda clean -y --all

# Install mmhuman3d
RUN . /root/miniconda3/etc/profile.d/conda.sh && \
    conda activate xrmogen && \
    mkdir /workspace && cd /workspace && \
    git clone https://github.com/open-mmlab/mmhuman3d.git && \
    cd mmhuman3d && pip install -e . && \
    pip cache purge

# install xrmogen requirements
RUN . /root/miniconda3/etc/profile.d/conda.sh && \
    conda activate xrmogen && \
    pip install imageio==2.15.0 && \
    pip install mmcv==1.6.1 && \
    pip install numpy && \
    pip install opencv_python && \
    pip install Pillow && \
    pip install scipy && \
    pip install tqdm && \
    pip install xrprimer && \
    pip install -e . && \
    pip cache purge