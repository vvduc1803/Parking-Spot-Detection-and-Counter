# Initialize base for image
FROM ubuntu:22.10

# Inialize file
WORKDIR /save

# Update ubuntu and install python
RUN apt-get update -y
RUN apt-get install -y libsm6 libxext6 ffmpeg libfontconfig1 libxrender1 libgl1-mesa-glx
RUN apt-get install -y python3-pip build-essential pkg-config
RUN pip3 install --upgrade pip
RUN pip3 install torch
RUN pip3 install torchvision

# Clone all file to image
COPY . .

# Install necessary package
RUN pip3 install -r requirements.txt

# Run app
CMD ["python3","run.py"]