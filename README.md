# Pytorch i3d AML
 Distributed training of  DeepMind's i3d in Pytorch using AML

Cookiecutter created project from [template](https://github.com/microsoft/DistributedDeepLearning) to facilitate distributed training on AML. Model file derived from this [https://github.com/piergiaj/pytorch-i3d](https://github.com/piergiaj/pytorch-i3d)

## Instructions

sudo usermod -aG docker $USER
newgrp docker 
make build
make run
# may need to fix path: export PYTHONPATH
# inv --complete
inv pytorch-i3d.submit.remote.submit

## Issues

... Validation-loss should converge to 0.8
