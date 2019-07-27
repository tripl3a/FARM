#!/usr/bin/env bash

# install FARM itself (requirements should already be satisfied from Docker image but will be extended as needed)
git clone -b nohate https://github.com/tripl3a/FARM.git && cd FARM && pip install -r requirements.txt && pip install .

/bin/bash run_hate_detector.sh