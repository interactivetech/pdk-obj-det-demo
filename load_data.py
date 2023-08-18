# General modules
import os
import shutil
import random
import numpy as np

# Torch modules
import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import models, transforms

# Image modules
from PIL import Image
from skimage import io
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Pachyderm modules
import python_pachyderm
from python_pachyderm.proto.v2.pfs.pfs_pb2 import FileType
from python_pachyderm.pfs import Commit


def download_data(pachyderm_host, pachyderm_port, repo, branch, project, download_dir, token):
    
    files = download_pach_repo(
        pachyderm_host,
        pachyderm_port,
        repo,
        branch,
        download_dir,
        token,
        project,
    )
    
    # Return list local destination path for each file
    return [des for src, des in files ]

def safe_open_wb(path):
    ''' Open "path" for writing, creating any parent directories as needed.
    '''
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return open(path, 'wb')

def download_pach_repo(
    pachyderm_host,
    pachyderm_port,
    repo,
    branch,
    root,
    token,
    project="default",
    previous_commit=None,
):
    print(f"Starting to download dataset: {repo}@{branch} --> {root}")

    if not os.path.exists(root):
        os.makedirs(root)

    client = python_pachyderm.Client(
        host=pachyderm_host, port=pachyderm_port, auth_token=token
    )
    files = []
    if previous_commit is not None:
        for diff in client.diff_file(
            Commit(repo=repo, id=branch, project=project), "/",
            Commit(repo=repo, id=previous_commit, project=project),
        ):
            src_path = diff.new_file.file.path
            des_path = os.path.join(root, src_path[1:])
            print(f"Got src='{src_path}', des='{des_path}'")

            if diff.new_file.file_type == FileType.FILE:
                if src_path != "":
                    files.append((src_path, des_path))
    else:
        for file_info in client.walk_file(
            Commit(repo=repo, id=branch, project=project), "/"):
            src_path = file_info.file.path
            des_path = os.path.join(root, src_path[1:])
            # print(f"Got src='{src_path}', des='{des_path}'")

            if file_info.file_type == FileType.FILE:
                if src_path != "":
                    files.append((src_path, des_path))

    for src_path, des_path in files:
        src_file = client.get_file(
            Commit(repo=repo, id=branch, project=project), src_path
        )
        # print(f"Downloading {src_path} to {des_path}")

        with safe_open_wb(des_path) as dest_file:
            shutil.copyfileobj(src_file, dest_file)

    print("Download operation successful!")
    return files




# Create transforms for data (resize, crop, flip, noramlize)
def get_train_transforms():
    return transforms.Compose([
        transforms.Resize(240),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])