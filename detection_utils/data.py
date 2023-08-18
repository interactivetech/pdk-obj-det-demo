"""
COCO dataset which returns image_id for evaluation.
Mostly copy-paste from https://github.com/pytorch/vision/blob/13b35ff/references/detection/coco_utils.py
"""
import os
from io import BytesIO
import boto3
from tqdm import tqdm
from google.cloud import storage
from determined.util import download_gcs_blob_with_backoff
import os
import torch
import torchvision
from PIL import Image
from attrdict import AttrDict
# from misc import nested_tensor_from_tensor_list
from detection_utils.coco import ConvertCocoPolysToMask, make_coco_transforms
from pathlib import Path
from time import time

def unwrap_collate_fn(batch):
    batch = list(zip(*batch))
    return tuple(batch)


class LocalBackend:
    """
    This class will load data from harddrive.
    COCO dataset will be downloaded from source in model_def.py if
    local backend is specified.
    """

    def __init__(self, outdir):
        assert os.path.isdir(outdir)
        self.outdir = outdir

    def get(self, filepath):
        with open(os.path.join(self.outdir, filepath), "rb") as f:
            img_str = f.read()
        return img_str


class CocoDetection(torchvision.datasets.CocoDetection):
    def __init__(
        self,
        backend,
        root_dir,
        img_folder,
        ann_file,
        transforms,
        return_masks,
        catIds=[],
    ):
        super(CocoDetection, self).__init__(img_folder, ann_file)
        self.img_folder = img_folder
        self._transforms = transforms
        self.prepare = ConvertCocoPolysToMask(return_masks)
        if backend == "local":
            self.backend = LocalBackend(root_dir)
        else:
            raise NotImplementedError

        self.catIds = catIds
        self.catIds = self.coco.getCatIds()
        '''
        Remapping to set background class to zero, so can support FasterRCNN models
        '''
        self.catIdtoCls = {
            catId: i+1 for i, catId in zip(range(len(self.catIds)), self.catIds)
        }
        self.clstoCatId = {
            v:k for k,v in self.catIdtoCls.items()
        }
        self.num_classes = len(list(self.catIdtoCls.values()))+1

    def __getitem__(self, idx):
        coco = self.coco
        img_id = self.ids[idx]
        ann_ids = coco.getAnnIds(imgIds=img_id, catIds=self.catIds)
        target = coco.loadAnns(ann_ids)
        path = coco.loadImgs(img_id)[0]["file_name"]
        img_bytes = BytesIO(self.backend.get(os.path.join(self.img_folder, path)))

        img = Image.open(img_bytes).convert("RGB")
        # img.save('test.png')
        image_id = self.ids[idx]
        target = {"image_id": image_id, "annotations": target}
        img, target = self.prepare(img, target)
        if self._transforms is not None:
            img, target = self._transforms(img, target)
        target["labels"] = torch.tensor(
                [self.catIdtoCls[l.item()] for l in target["labels"]], dtype=torch.int64
            )

        return img, target

    def __len__(self):
        return len(self.ids)


def build_xview_dataset_filtered(image_set, args):
    root = args.data_dir
    mode = "instances"

    PATHS = {
        "train": ("train_images_rgb_no_neg_filt_32/train_images_640_02_filt_32", 'train_images_rgb_no_neg_filt_32/train_640_02_filtered_32.json'),
        "val": ("val_images_rgb_no_neg_filt_32/val_images_640_02_filt_32", 'val_images_rgb_no_neg_filt_32/val_640_02_filtered_32.json'),
    }

    catIds = [] if "cat_ids" not in args else args.cat_ids
    img_folder, ann_file = PATHS[image_set]
    
    dataset = CocoDetection(
        args.backend,
        args.data_dir,
        img_folder,
        os.path.join(args.data_dir,ann_file),
        transforms=make_coco_transforms(image_set),
        return_masks=args.masks,
        catIds=catIds,
    )
    return dataset, dataset.num_classes

if __name__ == '__main__':
    DATA_DIR='determined-ai-xview-coco-dataset/train_sliced_no_neg/train_images_300_02/'
    data, n_classes = build_xview_dataset(image_set='train',args=AttrDict({
                                                'data_dir':DATA_DIR,
                                                'backend':'aws',
                                                'masks': None,
                                                }))
    for im, ann in tqdm(data):
            print(im.shape)
            print(len(ann))
            print(ann)
            break
    DATA_DIR='determined-ai-xview-coco-dataset/val_sliced_no_neg/val_images_300_02/'
    data, n_classes = build_xview_dataset(image_set='val',args=AttrDict({
                                                'data_dir':DATA_DIR,
                                                'backend':'aws',
                                                'masks': None,
                                                }))
    for im, ann in tqdm(data):
            print(im.shape)
            print(len(ann))
            print(ann)
            break