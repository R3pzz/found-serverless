# Use a base image with PyTorch and CUDA
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel

# Set environment variables to reduce interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    python3-pip \
    python3-dev \
    wget \
    zip \
    && rm -rf /var/lib/apt/lists/*

# Install Python and Pytorch dependencies
RUN pip install --upgrade pip
RUN pip install gdown

# Get Python version
RUN echo "Python version: $(python3 --version)"

# -----------------------------------------------------
# Dependencies
# -----------------------------------------------------

# ----------------------------------------------------
# Pytorch3d-------------------------------------------
WORKDIR /app
RUN git clone --recurse-submodules https://github.com/facebookresearch/pytorch3d.git
WORKDIR /app/pytorch3d
RUN pip install -e .
# ----------------------------------------------------

# ----------------------------------------------------
# SAM2 ------------------------------------------------
WORKDIR /app
RUN git clone --recurse-submodules https://github.com/facebookresearch/sam2.git

# Install SAM2 and download the checkpoints
WORKDIR /app/sam2
RUN pip3 install -e .
WORKDIR /app/sam2/checkpoints
RUN ./download_ckpts.sh
# ----------------------------------------------------

# ----------------------------------------------------
# FOUND -----------------------------------------------
WORKDIR /app
RUN git clone -b no-keypoints --recurse-submodules https://github.com/R3pzz/FOUND.git

# Install requirements
WORKDIR /app/FOUND
RUN pip install -r requirements.txt
RUN pip install iopath
RUN pip install --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py310_cu121_pyt240/download.html

# Download FIND model checkpoints
RUN mkdir /app/FOUND/data
WORKDIR /app/FOUND/data
RUN gdown --folder https://drive.google.com/drive/folders/1XWmEVo3AdnhJU2fs6igls-emp93beQpm
# ----------------------------------------------------

# ----------------------------------------------------
# SNU -------------------------------------------------
WORKDIR /app
RUN git clone --recurse-submodules https://github.com/R3pzz/surface_normal_uncertainty.git
# ----------------------------------------------------

# -----------------------------------------------------
# Main
# -----------------------------------------------------

# Copy the orchestration script from the notebook
WORKDIR /app
COPY /server/src /app/src

# Expose necessary ports (optional, for debugging)
EXPOSE 8000

# Default command to run the pipeline
CMD ["python3", "src/run.py"]