# https://github.com/huggingface/pytorch-pretrained-BERT/blob/master/docker/Dockerfile

FROM pytorch/pytorch:1.1.0-cuda10.0-cudnn7.5-devel

RUN git clone https://github.com/NVIDIA/apex.git && cd apex && python setup.py install --cuda_ext --cpp_ext

# install requirements only. most recent repository version will be installed when K8s job is run.
RUN git clone https://github.com/tripl3a/FARM.git && pip install -r FARM/requirements.txt && rm -r FARM

RUN pip install tensorboard tensorboardX

RUN mkdir /start
COPY container_start.sh /start
RUN chmod +x /start/container_start.sh

WORKDIR /workspace

# docker build . --file 00_FARM.Dockerfile -t registry.datexis.com/aallhorn/farm:0.0.2
# docker push registry.datexis.com/aallhorn/farm:0.0.2

