"""Module for running PyTorch training on Imagenet data
"""
from invoke import task, Collection
import os
from config import load_config


_BASE_PATH = os.path.dirname(os.path.abspath(__file__))
env_values = load_config()


@task
def submit_local(c, epochs=1):

    from aml_compute import PyTorchExperimentCLI

    exp = PyTorchExperimentCLI("pytorch_synthetic_images_local")
    run = exp.submit_local(
        os.path.join(_BASE_PATH, "src"),
        "charades_pytorch_horovod.py",
        {
            "--root": "/data/Charades_v1_rgb/",
            "--train_split": "charades.json",
            "--save_folder": "outputs",
            "--use_gpu":True
        },
        dependencies_file=os.path.join(_BASE_PATH, "environment_gpu.yml"),
        docker_args=["-v", f"{env_values['DATA']}:/data"],
        wait_for_completion=True,
    )
    print(run)


@task
def submit_remote(c, node_count=int(env_values["CLUSTER_MAX_NODES"]), epochs=1):

    from aml_compute import PyTorchExperimentCLI

    exp = PyTorchExperimentCLI("pytorch_real_images_remote")
    run = exp.submit(
        os.path.join(_BASE_PATH, "src"),
        "charades_pytorch_horovod.py",
        {
            "--root": "{datastore}/train/",
            "--train_split": "{datastore}/train/charades.json",
            "--save_folder": "outputs",
            "--use_gpu":True
        },
        node_count=node_count,
        dependencies_file=os.path.join(_BASE_PATH, "environment_gpu.yml"),
        wait_for_completion=True,
    )
    print(run)

remote_collection = Collection("remote")
remote_collection.add_task(submit_remote, "submit")


local_collection = Collection("local")
local_collection.add_task(submit_local, "submit")


submit_collection = Collection("submit", local_collection, remote_collection)
namespace = Collection("pytorch_i3d", submit_collection)

