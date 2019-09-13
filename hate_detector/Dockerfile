# https://github.com/huggingface/pytorch-pretrained-BERT/blob/master/docker/Dockerfile

FROM pytorch/pytorch:1.1.0-cuda10.0-cudnn7.5-devel

RUN git clone https://github.com/NVIDIA/apex.git && cd apex && python setup.py install --cuda_ext --cpp_ext

# install requirements only. most recent repository version will be installed when K8s job is run.
RUN git clone -b nohate https://github.com/tripl3a/FARM.git && pip install -r FARM/requirements.txt && rm -r FARM

RUN pip install tensorboard tensorboardX

COPY container_start.sh /workspace
RUN chmod +x /workspace/container_start.sh

WORKDIR /workspace

# docker build . -t registry.datexis.com/aallhorn/farm:0.0.8 --no-cache
# docker push registry.datexis.com/aallhorn/farm:0.0.8

