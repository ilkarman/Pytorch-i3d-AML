import logging
import logging.config
from invoke import task, Collection
from dotenv import find_dotenv, set_key
from invoke.exceptions import Failure
from config import load_config
import os
# Experiment imports Template Benchmark Imagenet PyTorch Imagenet 
import pytorch_i3d #  PyTorch Benchmark 
from invoke.executor import Executor
 # {%- endif -%} PyTorch Imagenet {% if cookiecutter.type == "imagenet" or cookiecutter.type == "pytorch_imagenet" or cookiecutter.type == "all"%}
import storage
import image

logging.config.fileConfig(os.getenv("LOG_CONFIG", "logging.conf"))
env_values = load_config()
# Imagenet switch variable 
_USE_IMAGENET = False # Don't use imagenet data 


def _is_loged_in(c):
    try:
        result = c.run("az account show", hide="both")
        return "Please run 'az login'" not in result.stdout
    except Failure:
        return False


def _prompt_sub_id_selection(c):
    # Importing here to avoid overhead to the rest of the application
    from tabulate import tabulate
    from toolz import pipe
    import json
    from prompt_toolkit import prompt

    results = c.run(f"az account list", pty=True, hide="out")
    sub_dict = json.loads(results.stdout)
    sub_list = [
        {"Index": i, "Name": sub["name"], "id": sub["id"]}
        for i, sub in enumerate(sub_dict)
    ]
    pipe(sub_list, tabulate, print)
    prompt_result = prompt("Please type in index of subscription you want to use: ")
    sub_id = sub_list[int(prompt_result)]["id"]
    print(f"You selected index {prompt_result} sub id {sub_id}")
    return sub_id


@task
def select_subscription(c, sub_id=env_values.get("SUBSCRIPTION_ID", None)):
    """Select Azure subscription to use
    
    Note:
        If sub_id isn't provided or found in env values interactive prompt is created asking for sub id selection
        The selection is then recorded in the env file

    Args:
        sub_id (string, optional): [description]. Defaults to env_values.get("SUBSCRIPTION_ID", None).
    """
    env_file = find_dotenv(raise_error_if_not_found=True)
    if sub_id is None or sub_id == "":
        sub_id = _prompt_sub_id_selection(c)
        set_key(env_file, "SUBSCRIPTION_ID", sub_id)
    c.run(f"az account set -s {sub_id}", pty=True)


@task(post=[select_subscription])
def login(c):
    """Log in to Azure CLI
    """
    if _is_loged_in(c):
        return None
    c.run("az login -o table", pty=True)


@task(aliases=("i"))
def interactive(c):
    """Open IPython terminal and load in modules to work with AML
    """
    c.run("python /workspace/control/src/aml_compute.py -- --interactive", pty=True)


@task
def delete(c, resource_group=env_values.get("RESOURCE_GROUP")):
    """Delete the resource group and all associated resources
    """
    c.run(f"az group delete --resource-group {resource_group} --no-wait --yes")


@task
def setup(c, use_imagenet=_USE_IMAGENET, use_tfrecords=True):
    """Setup the environment and process the imagenet data
    
    Args:
        use_imagenet (bool, optional): Process imagenet data and upload to cloud. Defaults to True.
        use_tfrecords (bool, optional): Convert imagenet data to tfrecords and upload to cloud. Defaults to True.
    """
    logger = logging.getLogger(__name__)

    c.invoke_execute(c, "login")

    if use_imagenet:
        logger.info("Preparing Imagenet data")
        # Need to use invoke_execute to ensure that pretasks get executed
        image.prepare_imagenet(c)
        c.invoke_execute(c, "storage.image.upload_data")
        if use_tfrecords:
            tfrecords.generate_tf_records(c)
            c.invoke_execute(c, "storage.tfrecords.upload_data")
    logger.info("Setup complete")


@task
def tensorboard(c, experiment, runs=None):
    """Runs tensorboard in a seperate tmux session
    
    Note:
        If no runs are specified it will simply look for the run that are still running. 
        To see runs that completed or failed simply also include the run identifier
    Args:
        experiment (string): The name of the experiment you wish to display the logged information for
        runs (list[tring], optional): The list of run identifiers you want to display in tensorboard from the experiment. Defaults to None.
    """
    cmd = f"tmux neww -d -n tensorboard python control/src/aml_compute.py tensorboard --experiment {experiment} "
    if runs:
        cmd = cmd + f"--runs {runs}"
    c.run(cmd)


@task
def runs(
    c,
    experiment,
    resource_group=env_values["RESOURCE_GROUP"],
    workspace=env_values["WORKSPACE"],
    last=10,
):
    """Prints information on last N runs in specified experiment

    Args:
        experiment (string): Name of experiment to return runs from
        resource_group (string, optional): Resource group. Defaults to env_values["RESOURCE_GROUP"].
        workspace (string, optional): Workspace name. Defaults to env_values["WORKSPACE"].
        last (int, optional): The number of runs to return. Defaults to 10.
    """
    cmd = f"az ml run list --experiment-name {experiment} --resource-group {resource_group} --workspace-name {workspace} -o table"
    c.run(cmd)


@task
def experiments(
    c, resource_group=env_values["RESOURCE_GROUP"], workspace=env_values["WORKSPACE"]
):
    """Prints list of experiments

    Args:
        resource_group (string, optional): Resource group. Defaults to env_values["RESOURCE_GROUP"].
        workspace (string, optional): Workspace name. Defaults to env_values["WORKSPACE"].
    """
    cmd = f"az ml experiment list --resource-group {resource_group} --workspace-name {workspace} -o table"
    c.run(cmd)


def invoke_execute(context, command_name, **kwargs):
    """
    Helper function to make invoke-tasks execution easier.
    """
    results = Executor(namespace, config=context.config).execute((command_name, kwargs))
    target_task = context.root_namespace[command_name]
    return results[target_task]


namespace = Collection(
    setup,
    delete,
    interactive,
    login,
    select_subscription,
    tensorboard,
    runs,
    experiments,
)

# Experiment # Benchmark # Imagenet 

# PyTorch Benchmark 
pytorch_i3d_collection = Collection.from_module(pytorch_i3d)
namespace.add_collection(pytorch_i3d_collection)

storage_collection = Collection.from_module(storage)
storage_collection.add_collection(Collection.from_module(image))
namespace.add_collection(storage_collection)

#

# PyTorch Imagenet 

namespace.configure({
    'root_namespace': namespace,
    'invoke_execute': invoke_execute,
})