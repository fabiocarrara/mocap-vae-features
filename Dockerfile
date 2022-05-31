FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime

ADD requirements.txt .
RUN pip install -r requirements.txt