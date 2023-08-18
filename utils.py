import torch
import random
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pycocotools.coco import COCO

def visualize_coco_annotations(coco_ann_file_path: str, coco_img_dir: str) -> None:
    """
    Visualize COCO annotations on randomly selected images.

    Parameters:
        coco_ann_file_path (str): Path to the COCO annotation file in JSON format.
        coco_img_dir (str): Path to the directory containing the COCO images.
    """
    # Create a COCO object and load annotations from the file
    coco = COCO(coco_ann_file_path)

    # Get all image IDs present in the dataset
    image_ids = coco.getImgIds()

    # Randomly select 12 image IDs
    random_image_ids = random.sample(image_ids, 8)

    # Load images by index from files and add to a list
    imgs = []
    for image_id in random_image_ids:
        img_info = coco.loadImgs(image_id)[0]
        img_path = coco_img_dir + "/" + img_info['file_name']
        imgs.append(mpimg.imread(img_path))

    # Create subplots
    f, axarr = plt.subplots(4, 2, figsize=(15, 15))
    f.subplots_adjust(wspace=0.1, hspace=0/1)
    # Step 6: Fill subplots with images and annotations
    i_idx = 0
    for i in range(4):
        for ii in range(2):
            img_id = random_image_ids[i_idx]
            img_info = coco.loadImgs(img_id)[0]
            ann_ids = coco.getAnnIds(imgIds=img_id)
            anns = coco.loadAnns(ann_ids)
            ax = axarr[i, ii]
            ax.imshow(imgs[i_idx])
            ax.axis('off')  # Remove axes
            for ann in anns:
                bbox = ann['bbox']
                x, y, w, h = bbox
                class_id = ann['category_id']
                class_name = coco.loadCats(class_id)[0]['name']
                rect = plt.Rectangle((x, y), w, h, fill=False, edgecolor='red', linewidth=2)
                ax.add_patch(rect)
                ax.text(x, y, '{}: {}'.format(class_id, class_name),
                                  bbox=dict(facecolor='white', alpha=0.7))

            i_idx += 1
    plt.show()
    
def calculate_coco_stats(coco_file: str) -> None:
    '''
    Calculate statistics of annotations in a COCO annotation file
    Parameters:
        coco_file (str): Path to the COCO annotation file in JSON format
    '''
    print("COCO annotation file: ",coco_file)
    coco = COCO(coco_file)
    num_annotations = len(coco.dataset['images'])

    cat_ids = coco.getCatIds()

    # All categories.
    cats = coco.loadCats(cat_ids)
    cat_names = [cat["name"] for cat in cats]
    # Step 3: Get the number of annotations in the dataset

    # Step 4: Calculate the number of annotations per category
    annotations_per_category = {}
    for category_id in cat_ids:
        annotation_ids = coco.getAnnIds(catIds=category_id)
        num_annotations = len(annotation_ids)
        category_name = coco.loadCats(category_id)[0]['name']
        annotations_per_category[category_name] = num_annotations



    # Initialize variables for tracking min, max, and total annotations
    min_annotations = float('inf')
    max_annotations = 0
    total_annotations = 0
    image_ids = coco.getImgIds()
    # Step 4: Calculate the minimum, maximum, and average number of annotations
    for image_id in image_ids:
        annotation_ids = coco.getAnnIds(imgIds=image_id)
        num_annotations = len(annotation_ids)

        # Update minimum and maximum if needed
        min_annotations = min(min_annotations, num_annotations)
        max_annotations = max(max_annotations, num_annotations)

        # Add to total for calculating average
        total_annotations += num_annotations

    # Calculate the average number of annotations
    average_annotations = total_annotations / len(image_ids)
    # Initialize variables for tracking metrics
    min_annotation_area = float('inf')
    max_annotation_area = 0
    total_annotation_area = 0
    num_annotations = 0

    # Initialize dictionaries to track per-category metrics
    min_annotation_area_per_category = {}
    max_annotation_area_per_category = {}

    # Step 4: Calculate metrics
    for image_id in image_ids:
        image_annotations = coco.loadAnns(coco.getAnnIds(imgIds=image_id))
        for annotation in image_annotations:
            area = annotation['area']
            total_annotation_area += area
            num_annotations += 1

            # Update min and max annotation area
            min_annotation_area = min(min_annotation_area, area)
            max_annotation_area = max(max_annotation_area, area)

            # Update min and max annotation area per category
            category_id = annotation['category_id']
            category_name = coco.loadCats(category_id)[0]['name']
            if category_name not in min_annotation_area_per_category:
                min_annotation_area_per_category[category_name] = float('inf')
                max_annotation_area_per_category[category_name] = 0
            min_annotation_area_per_category[category_name] = min(min_annotation_area_per_category[category_name], area)
            max_annotation_area_per_category[category_name] = max(max_annotation_area_per_category[category_name], area)

    # Calculate average annotation area
    avg_annotation_area = total_annotation_area / num_annotations


    return (num_annotations,
           cat_ids, 
           cat_names, 
           annotations_per_category, 
           min_annotations,
           max_annotations,
           average_annotations,
           min_annotation_area,
           max_annotation_area,
           avg_annotation_area,
           min_annotation_area_per_category,
           max_annotation_area_per_category)

def load_model(loaded_model,path):
    '''
    '''
    model_state_dict = torch.load(path +'/state_dict.pth')['models_state_dict'][0]
    try:
        loaded_model.load_state_dict(model_state_dict)
    except Exception:
        # If the checkpointed model is non-DDP and the current model is DDP, append
        # module prefix to the checkpointed data
        if isinstance(loaded_model, torch.nn.parallel.DistributedDataParallel):
            print("Loading non-DDP checkpoint into a DDP model.")
            torch.nn.modules.utils._add_prefix_in_state_dict_if_not_present(model_state_dict, "module.")
        else:
            # If the checkpointed model is DDP and if we are currently running in
            # single-slot mode, remove the module prefix from checkpointed data
            print("Loading DDP checkpoint into a non-DDP model.")
            torch.nn.modules.utils.consume_prefix_in_state_dict_if_present(
                model_state_dict, "module."
            )
        loaded_model.load_state_dict(model_state_dict)
    return loaded_model
'''
Single Example:
det_labels = ["Fixed-wing Aircraft", "Cargo Plane"]
# Get one sample from the data set and show the image for validation
for sample in data_loader_test:
    img, targets = sample
    img = img[0].clone().cpu()
    boxes = targets[0]['boxes'].cpu().numpy()
    labels = targets[0]['labels'].cpu().numpy()
    fig, ax = plt.subplots(1)
    ax.set_title("Example")
    ax.imshow(np.clip(img.permute(1, 2, 0), 0, 1))
    # Plot bounding boxes as rectangles on the image
    for box, label in zip(boxes, labels):
        x_min, y_min, x_max, y_max = box
        rect = plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                             fill=False, edgecolor='red', linewidth=2,alpha=0.5)
        ax.add_patch(rect)
        ax.text(x_min, y_min - 5, f'Label: {det_labels[label-1]}', color='red', fontsize=10)
    
    # Model prediction
    break
plt.axis('off')
plt.show()

model.eval()
for sample in data_loader_test:
    img, targets = sample
    img2 = img[0].clone().cpu()
    with torch.no_grad():
        model_time = time.time()
        images = [list(i.to(device) for i in img)[0]]
        targets = [[{k: v.to(device) for k, v in t.items()} for t in targets][0]]
        fig, ax = plt.subplots(1)
        ax.set_title("Example")
        ax.imshow(np.clip(img2.permute(1, 2, 0), 0, 1))
        outputs = model(images)
        print(outputs)
        outputss = []
        for t in outputs:
            outputss.append({k: v.to('cpu') for k, v in t.items()})
        model_time = time.time() - model_time
        res = {target["image_id"].item(): output for target, output in zip(targets, outputss)}
        model_time_str = str(datetime.timedelta(seconds=int(model_time)))
        boxes = outputs[0]['boxes'].cpu().numpy()
        labels = outputs[0]['labels'].cpu().numpy()
        scores = outputs[0]['scores'].cpu().numpy()
        # Plot bounding boxes with scores greater than 0.05 as rectangles on the image
        for box, label, score in zip(boxes, labels, scores):
            if score > 0.05:
                x_min, y_min, x_max, y_max = box
                rect = plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                                     fill=False, edgecolor='green', linewidth=2,alpha=0.5)
                ax.add_patch(rect)
                ax.text(x_min, y_min - 5, f'Label: {label}, Score: {score:.2f}', color='green', fontsize=12)
    break
plt.axis('off')
plt.show()
'''