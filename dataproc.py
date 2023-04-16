from pycocotools.coco import COCO
import pycocotools.mask as mask_util
from pycocotools import mask

import matplotlib.pyplot as plt
import skimage.io as io
import numpy as np

import datasets
from datasets import DatasetDict, Dataset, Features, Value
from datasets import Image as ImageFeature
from transformers import Blip2Processor, Blip2ForConditionalGeneration

import torch

from PIL import Image, ImageDraw

import json
import os

from tqdm.auto import tqdm
from loguru import logger


def generation_captions(images: np.ndarray, processor, captioning_model, device):
    """Generates captions for a batch of images.

    Args:
        images: A batch of images in the RGB format.

    Returns:
        A list of generated captions.
    """
    inputs = processor(images=images, return_tensors="pt").to(device)

    generated_ids = captioning_model.generate(**inputs)
    generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)
    generated_texts = [text.strip() for text in generated_texts]
    return generated_texts

def save_keypoints(image, coco, img_id):

    # Get annotation IDs for the current image
    #image_info = coco.loadImgs(coco.getImgIds(filename=image_path))[0]
    #ann_ids = coco.getAnnIds(imgIds=image_info['id'])
    #anns = coco.loadAnns(ann_ids)
    
    catIds = coco.getCatIds(catNms=['human']);
    imgIds = coco.getImgIds(catIds=catIds );
    imgIds = coco.getImgIds(imgIds = [img_id])
    #image_info = coco.loadImgs(imgIds[np.random.randint(0,len(imgIds))])[0]
    
    annIds = coco.getAnnIds(imgIds=imgIds, catIds=catIds, iscrowd=None)
    anns = coco.loadAnns(annIds)

    annotated_img = Image.new('RGB', image.size)
    # Draw keypoints on image
    draw = ImageDraw.Draw(annotated_img)
    for ann in anns:
        if 'keypoints' in ann and type(ann['keypoints']) == list:
            # turn skeleton into zero-based index
            sks = np.array(coco.loadCats(ann['category_id'])[0]['skeleton']) - 1
            kp = np.array(ann['keypoints'])
            x = kp[0::3]
            y = kp[1::3]
            v = kp[2::3]
            c = (255, 0, 0)  # color of keypoints
            for sk in sks:
                if np.all(v[sk] > 0):
                    draw.line((x[sk[0]], y[sk[0]], x[sk[1]], y[sk[1]]), fill=c, width=3)
            for i in range(len(x)):
                if v[i] > 0:
                    draw.ellipse((x[i] - 4, y[i] - 4, x[i] + 4, y[i] + 4), fill=c, outline=c)

    return np.array(annotated_img)

def create_dataset(image_folder, annotations, coco, processor, captioning_model, device, mask_folder=None, keypoint_folder=None, debug=False):
    """Load image and segmentation mask data into a Hugging Face dataset."""
    mask_folder = os.path.join(image_folder, "masks")
    os.makedirs(mask_folder, exist_ok=True)
    
    keypoint_folder = os.path.join(image_folder, "keypoints")
    os.makedirs(keypoint_folder, exist_ok=True)

    # Create empty lists for image and mask data
    images = []
    segmentation_masks = []
    keypoint_masks = []
    captions = []
    raw_imgs = []

    # Loop through images and masks
    for i, image_info in tqdm(enumerate(annotations["images"]), desc="Loading dataset", total=len(annotations["images"])):
        if debug:
            if i == 5:
                break
        # Load image
        image_path = os.path.join(image_folder, image_info["file_name"])
        raw_image = Image.open(image_path)
        image = np.array(raw_image)
        height, width, _ = image.shape
        
        images.append(image_path)
        raw_imgs.append(image)

        # Load segmentation mask
        annotation = annotations["annotations"][i]
        rle = mask.frPyObjects(annotation['segmentation'], height, width)
        mask_arr = mask.decode(rle)
        mask_arr = np.sum(mask_arr, axis=2)  # Convert to binary mask
        mask_arr = np.where(mask_arr > 0, 255, 0).astype(np.uint8)  # Ensure binary mask has correct shape and data type

        # Save segmentation mask as image
        if mask_folder is None:
            mask_folder = os.path.join(image_folder, "masks")
        if not os.path.exists(mask_folder):
            os.makedirs(mask_folder)
        
        mask_path = os.path.join(mask_folder, f"{image_info['file_name'][:-4]}.png")
        dir_path = os.path.dirname(mask_path)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        Image.fromarray(mask_arr).save(mask_path)
        
        segmentation_masks.append(mask_path)
        
        # Save keypoint mask as image
        keypoint_arr = save_keypoints(raw_image, coco, i)
        if keypoint_folder is None:
            keypoint_folder = os.path.join(image_folder, "keypoints")
        if not os.path.exists(keypoint_folder):
            os.makedirs(keypoint_folder)
        
        mask_path = os.path.join(keypoint_folder, f"{image_info['file_name'][:-4]}.png")
        dir_path = os.path.dirname(mask_path)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        Image.fromarray(keypoint_arr).save(mask_path)
        
        keypoint_masks.append(mask_path)
            
        if i % 300 == 0:
            captions.append(generation_captions(raw_imgs, processor, captioning_model, device))
            raw_imgs = []
            
    if len(raw_imgs) > 0:
        captions[0].extend(generation_captions(raw_imgs, processor, captioning_model, device))
        raw_imgs = []

    # Create Hugging Face dataset from image and mask lists
    # dataset = Dataset.from_dict({"image": images, "mask": segmentation_masks, "keypoints": keypoint_masks, "caption": captions[0]})

    # Add dataset to dictionary and return
    # dataset_dict["train"] = dataset
    return images, segmentation_masks, keypoint_masks, captions[0]

def gen_examples():
    for i in tqdm(range(len(image_paths)), total=len(image_paths)):
        yield {
            "original_image": {"path": image_paths[i]},
            "segment_image": {"path": segmentation_paths[i]},
            "keypoint_image": {"path": keypoint_paths[i]},
            "caption": captions[i],
        }
        
def push_dataset(image_paths, segmentation_paths, keypoint_paths, captions):
    final_dataset = Dataset.from_generator(
    gen_examples,
    features=Features(
            original_image=ImageFeature(),
            segment_image=ImageFeature(),
            keypoint_image=ImageFeature(),
            caption=Value("string"),
        ),
        num_proc=196,
    )

    ds_name = "amateur_drawings-controlnet-dataset"
    final_dataset.push_to_hub(ds_name)
    
def main():
    logger.info('Loading annotations...')
    with open('/dev/data/amateur_drawings_annotations.json', "r") as f:
        annotations = json.load(f)
        
    device = "cuda" if torch.cuda.is_available() else "cpu"

    processor = Blip2Processor.from_pretrained("Salesforce/blip2-flan-t5-xl")
    captioning_model = Blip2ForConditionalGeneration.from_pretrained(
        "Salesforce/blip2-flan-t5-xl", #torch_dtype=torch.float16
    )
    captioning_model = captioning_model.to(device)
    
    coco=COCO('amateur_drawings_annotations.json')
    push_dataset(create_dataset("/mnt/disks/persist/data/", annotations, coco, processor, captioning_model, device))

if __name__ == "__main__":
    main()
        
    