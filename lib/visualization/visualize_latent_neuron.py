import os
import torch
import json
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch.nn.functional as F
import matplotlib.patches as patches
import matplotlib.image as mpimg
from collections import defaultdict

from tqdm import tqdm
from torch.utils.data import DataLoader
from albumentations import *
from PIL import Image
from torch import nn
from PIL import Image
from torch.utils.data import Dataset
from sparse_autoencoder import SparseAutoencoder
from lib.configs.ckpt_dict import ckpt_dict
from lib.dncbm.utils import common_init, get_sae_ckpt
from lib.dncbm import arg_parser
from collections import Counter
from torch.utils.data import Subset
from skimage.measure import label, regionprops
from breastclip.model.modules import load_image_encoder, LinearClassifier
from Datasets.dataset_utils import get_dataloader_RSNA

dir_path = os.path.dirname(os.path.realpath(__file__))


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_args():

    parser = arg_parser.get_common_parser()
    args = parser.parse_args()
    common_init(args)

    set_seed(42)

    outpath = args.save_dir_sae_ckpts[args.modality]

    return args


class BreastClipClassifier(nn.Module):
    def __init__(self, ckpt, n_class, image_encoder_type):
        super(BreastClipClassifier, self).__init__()

        self.image_encoder = load_image_encoder(
            ckpt["config"]["model"]["image_encoder"]
        )
        image_encoder_weights = {}
        for k in ckpt["model"].keys():
            if k.startswith("image_encoder."):
                image_encoder_weights[".".join(k.split(".")[1:])] = ckpt["model"][k]
        self.image_encoder.load_state_dict(image_encoder_weights, strict=True)
        self.image_encoder_type = image_encoder_type

        print("freezing image encoder to not be trained")
        for param in self.image_encoder.parameters():
            param.requires_grad = False

        self.classifier = LinearClassifier(
            feature_dim=self.image_encoder.out_dim, num_class=n_class
        )
        self.raw_features = None
        self.pool_features = None

    def get_image_encoder_type(self):
        return self.image_encoder_type

    def encode_image(self, image):

        input_dict = {"image": image, "breast_clip_train_mode": True}
        image_features, raw_features = self.image_encoder(input_dict)
        self.raw_features = raw_features
        self.pool_features = image_features
        return image_features

    def forward(self, images):
        if self.image_encoder_type.lower() == "swin":
            images = images.squeeze(1).permute(0, 3, 1, 2)
        image_feature = self.encode_image(images)
        logits = self.classifier(image_feature)
        return logits


def get_transforms(args):

    width = 1520
    height = 912
    alpha = 10
    sigma = 15
    p = 1

    return Compose(
        [
            HorizontalFlip(),
            VerticalFlip(),
            Affine(rotate=20, translate_percent=0.1, scale=[0.8, 1.2], shear=20),
            ElasticTransform(alpha=alpha, sigma=sigma),
        ],
        p=p,
    )


class MammoDataset(Dataset):
    def __init__(
        self,
        args,
        df,
        transform=None,
        data_dir="",
        img_dir="",
        dataset="",
        image_encoder_type="",
        label="",
        arch="",
        mean="",
        std="",
    ):

        self.df = df
        self.dir_path = os.path.join(data_dir, img_dir)
        self.dataset = dataset
        self.transform = transform
        self.image_encoder_type = image_encoder_type
        self.label = label
        self.arch = arch
        self.mean = mean
        self.std = std

        # print(transform)

    def __len__(self):
        return len(self.df)

    def get_all_boxes_for_instance(self, df, row):
        """
        Given a row from the DataFrame, return all bboxes from rows
        with matching patient_id, series_id, image_id, and Y-label.

        Args:
            df (pd.DataFrame): Full annotation DataFrame
            row (pd.Series): A single row (e.g., self.df.iloc[idx])

        Returns:
            List of bboxes in format [x_min, y_min, x_max, y_max]
        """
        filters = (
            (df["patient_id"] == row["patient_id"])
            & (df["series_id"] == row["series_id"])
            & (df["image_id"] == row["image_id"])
            & (df[self.label] == row[self.label])
        )

        matching_rows = df[filters]

        boxes = matching_rows[
            ["resized_xmin", "resized_ymin", "resized_xmax", "resized_ymax"]
        ].values.tolist()
        return boxes

    def __getitem__(self, idx):
        data = self.df.iloc[idx]
        img_path = os.path.join(
            self.dir_path,
            str(self.df.iloc[idx]["patient_id"]),
            str(self.df.iloc[idx]["image_id"]),
        )
        if self.dataset.lower() == "rsna":
            img_path = f"{img_path}.png"
        if (
            self.arch.lower() == "upmc_breast_clip_det_b5_period_n_ft"
            or self.arch.lower() == "upmc_vindr_breast_clip_det_b5_period_n_ft"
            or self.arch.lower() == "upmc_breast_clip_det_b5_period_n_lp"
            or self.arch.lower() == "upmc_vindr_breast_clip_det_b5_period_n_lp"
            or self.arch.lower() == "upmc_breast_clip_det_b2_period_n_ft"
            or self.arch.lower() == "upmc_vindr_breast_clip_det_b2_period_n_ft"
            or self.arch.lower() == "upmc_breast_clip_det_b2_period_n_lp"
            or self.arch.lower() == "upmc_vindr_breast_clip_det_b2_period_n_lp"
        ):
            img = Image.open(img_path).convert("RGB")
        else:
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)

        if self.transform and (
            self.arch.lower() == "swin_tiny_custom_norm"
            or self.arch.lower() == "swin_base_custom_norm"
        ):
            img = self.transform(img)
        elif self.transform:
            img = np.array(img)
            augmented = self.transform(image=img)
            img = augmented["image"]

            img = img.astype("float32")
            img -= img.min()
            img /= img.max()
            img = torch.tensor((img - self.mean) / self.std, dtype=torch.float32)
        else:
            img = np.array(img)
            img = img.astype("float32")
            img -= img.min()
            img /= img.max()
            img = torch.tensor((img - self.mean) / self.std, dtype=torch.float32)

        data_id = data["data_id"]

        gt_bbox = torch.tensor(
            [
                data["resized_xmin"],
                data["resized_ymin"],
                data["resized_xmax"],
                data["resized_ymax"],
            ]
        )

        gt_bbox = torch.tensor(
            self.get_all_boxes_for_instance(
                self.df,
                data,
            )
        )

        return {
            "x": img.unsqueeze(0),
            "y": torch.tensor(data[self.label], dtype=torch.long),
            "img_path": str(img_path),
            "data_id": str(data_id),
            "bboxes": gt_bbox,
        }


def collator_mammo_dataset_w_concepts(batch):
    return {
        "x": torch.stack([item["x"] for item in batch]),
        "y": torch.from_numpy(
            np.array([item["y"] for item in batch], dtype=np.float32)
        ),
        "img_path": [item["img_path"] for item in batch],
        "data_id": [item["data_id"] for item in batch],
        "bboxes": [item["bboxes"] for item in batch],
    }


def get_dataloader_RSNA(args, image_encoder_type, label, dataset, arch):

    train_tfm = None
    val_tfm = None
    batch_size = 1
    num_workers = 4
    csv_file = "datasets/shantanughosh/vindr-mammogram-dataset-dicom-to-png/versions/1/vindr_detection_v1_folds_updated.csv"
    data_dir = "/workspace/datasets"
    img_dir = "datasets/shantanughosh/vindr-mammogram-dataset-dicom-to-png/versions/1/images_png"
    mean = 0.3089279
    std = 0.25053555408335154

    df = pd.read_csv(os.path.join(data_dir, csv_file))
    df = df.fillna(0)

    # df = df[df[label] == 1].reset_index(drop=True)

    train_folds = df[df["split"] == "training"].reset_index(drop=True)
    valid_folds = df[df["split"] == "test"].reset_index(drop=True)

    # valid_folds = valid_folds[(valid_folds["Suspicious_Calcification"] == 1) &  (valid_folds["Mass"]==1)]

    if True:
        # Filter by class
        class_0 = valid_folds[valid_folds[label] == 0]
        class_1 = valid_folds[valid_folds[label] == 1]

        # Sample 115 from each, no replacement
        sampled_0 = class_0.sample(n=len(class_1), replace=False, random_state=42)
        sampled_1 = class_1.sample(n=len(class_1), replace=False, random_state=42)

        # Combine and shuffle
        valid_folds = (
            pd.concat([sampled_1])
            .sample(frac=1, random_state=42)
            .reset_index(drop=True)
        )

   

    train_tfm = get_transforms(args)
    val_tfm = None

    train_dataset = MammoDataset(
        args=args,
        df=train_folds,
        transform=train_tfm,
        data_dir=data_dir,
        img_dir=img_dir,
        dataset=dataset,
        image_encoder_type=image_encoder_type,
        label=label,
        mean=mean,
        std=std,
        arch=arch,
    )
    valid_dataset = MammoDataset(
        args=args,
        df=valid_folds,
        transform=val_tfm,
        data_dir=data_dir,
        img_dir=img_dir,
        dataset=dataset,
        image_encoder_type=image_encoder_type,
        label=label,
        mean=mean,
        std=std,
        arch=arch,
    )



    print(f"Valid dataset size: {len(valid_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=collator_mammo_dataset_w_concepts,
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=collator_mammo_dataset_w_concepts,
        prefetch_factor=4,  # Optional, helps pre-load batches
    )

    return train_loader, valid_loader








def get_bboxes_from_heatmap(
    heatmap,
    quantile=0.98,
    min_area=10,
    save_path=None,
    data_id="",
    mask_threshold="",
    concept_id=None,
):
    """
    Extract bounding boxes from a raw conv-feature heatmap.
    Optionally save the normalized heatmap and binary mask.

    Args:
        heatmap (np.ndarray): Raw CNN conv-channel (H, W), unnormalized
        quantile (float): Threshold quantile (default: 0.95)
        min_area (int): Minimum region area to keep
        save_basename (str): Base filename to save heatmap and mask

    Returns:
        List of bounding boxes: [x1, y1, x2, y2]
    """
    # Normalize heatmap to [0, 1] for thresholding and visualization
    heatmap_min, heatmap_max = heatmap.min(), heatmap.max()
    heatmap_norm = (heatmap - heatmap_min) / (heatmap_max - heatmap_min + 1e-6)
    heatmap_uint8 = (heatmap_norm * 255).astype(np.uint8)

    # Threshold based on quantile of raw values (not normalized!)
    threshold = np.quantile(heatmap, quantile)
    binary_mask = (heatmap > threshold).astype(np.uint8) * 255

    # Apply fixed threshold
    use_relative = True
    threshold = mask_threshold
    actual_thresh = threshold * heatmap_max if use_relative else threshold
    binary_mask = (heatmap > actual_thresh).astype(np.uint8) * 255

    # Save visualizations if requested
    if 0 and save_path:
        # os.makedirs(os.path.dirname(save_basename), exist_ok=True)
        cv2.imwrite(
            save_path + f"/data_id_{data_id}_concept_{concept_id}_heatmap.png",
            heatmap_uint8,
        )
        cv2.imwrite(
            save_path + f"/data_id_{data_id}_concept_{concept_id}_mask.png", binary_mask
        )

    # Find contours from binary mask
    contours, _ = cv2.findContours(
        binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    bboxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w * h >= min_area:
            bboxes.append([x, y, x + w, y + h])

    return bboxes


def check_valid_dimension(gt_bboxes):

    for box in gt_bboxes:
        xmin, ymin, xmax, ymax = box
        assert (
            0 <= xmin <= 950 and 0 <= xmax <= 950
        ), f"check box xmin: {xmin}, xmax: {xmax}"
        assert (
            0 <= ymin <= 1520 and 0 <= ymax <= 1520
        ), f"check box xmin: {ymin}, xmax: {ymax}"

    return


def clip_boxes_to_image(boxes, max_width=912, max_height=1520):
    """
    Clips each box in the list to fit within image dimensions.

    Args:
        boxes: List of [x1, y1, x2, y2] boxes
        max_width: Maximum image width
        max_height: Maximum image height

    Returns:
        List of clipped boxes
    """
    clipped = []
    for box in boxes:
        x1, y1, x2, y2 = box
        x1 = max(0, min(x1, max_width))
        x2 = max(0, min(x2, max_width))
        y1 = max(0, min(y1, max_height))
        y2 = max(0, min(y2, max_height))
        clipped.append([x1, y1, x2, y2])
    return clipped


def visualize_topk_concepts(
    images,
    concepts,
    labels,
    mean,
    std,
    k=2,
    save_dir="./outputs",
    batch_index=0,
    latent_neurons=[],
    img_paths=[],
    gt_boxes=[],
    data_ids=[],
    mask_threshold="",
):
    os.makedirs(save_dir, exist_ok=True)

    if isinstance(mean, float):
        mean = [mean] * 3
    if isinstance(std, float):
        std = [std] * 3

    mean_tensor = torch.tensor(mean).view(1, 3, 1, 1).to(images.device)
    std_tensor = torch.tensor(std).view(1, 3, 1, 1).to(images.device)
    images = images * std_tensor + mean_tensor
    images = images.clamp(0, 1)

    B, C, H, W = images.shape
    concepts = concepts.permute(0, 3, 1, 2)  # [B, Nc, Hc, Wc]
    concepts_upsampled = F.interpolate(
        concepts, size=(H, W), mode="bilinear", align_corners=False
    )

    results = {}

    for i in range(B):



        img = images[i].permute(1, 2, 0).cpu().numpy()
        concept_maps = concepts_upsampled[i]
        label = int(labels[i])
        gt_bboxes = gt_boxes[i].tolist()

        #print(f"Number of gt_boxes: {len(gt_bboxes)}")

        data_id = data_ids[i]
        gt_bboxes = clip_boxes_to_image(gt_bboxes)

        img_id = data_id  
        results[img_id] = {}
        os.makedirs(os.path.join(save_dir, 
                                  f"class={label}", f"image_id={data_id}"), 
                                  exist_ok=True)



        # Figure 1: Save raw image with GT boxes overlaid
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.imshow(img)
        for gt_box in gt_bboxes:
            rect = patches.Rectangle(
                (gt_box[0], gt_box[1]),
                gt_box[2] - gt_box[0],
                gt_box[3] - gt_box[1],
                linewidth=2,
                edgecolor="red",
                facecolor="none",
            )
            ax.add_patch(rect)
        ax.axis("off")
 

        # save figure 1
        raw_img_path = os.path.join(
            save_dir, f"class={label}", f"image_id={data_id}", "original_image.png"
        )


        fig.savefig(raw_img_path, bbox_inches="tight", pad_inches=0)
        plt.close()
        

        # loop over selected neurons
        for j, idx in enumerate(latent_neurons):


            heatmap = concept_maps[j].cpu().numpy()

            mean_act = float(np.mean(heatmap))
            pred_bboxes = get_bboxes_from_heatmap(
                heatmap,
                save_path=save_dir,
                data_id=data_id,
                mask_threshold=mask_threshold,
                concept_id=idx,
            )

            concept_key = f"concept_{idx}"
            results[img_id][concept_key] = {
                
                "gt_boxes": gt_bboxes,
                "pred_boxes": pred_bboxes,
                "mean_act": mean_act,
            }

            # Figure 2: Save heatmap overlay with ground-truth boxes
            fig, ax = plt.subplots(figsize=(4, 4))
            ax.imshow(img)
            ax.imshow(heatmap, cmap="jet", alpha=0.5)
            for gt_box in gt_bboxes:
                rect = patches.Rectangle(
                    (gt_box[0], gt_box[1]),
                    gt_box[2] - gt_box[0],
                    gt_box[3] - gt_box[1],
                    linewidth=2,
                    edgecolor="red",
                    facecolor="none",
                )
                ax.add_patch(rect)

            ax.axis("off")

            heatmap_fname = (
                f"neuron={idx}_heatmap.png"
            )

            fig.savefig(
                os.path.join(save_dir, f"class={label}", 
                f"image_id={data_id}", heatmap_fname), 
                bbox_inches="tight", pad_inches=0
            )

            # Figure 3: add pred-boxes as well
            for pred_box in pred_bboxes:
                rect = patches.Rectangle(
                    (pred_box[0], pred_box[1]),
                    pred_box[2] - pred_box[0],
                    pred_box[3] - pred_box[1],
                    linewidth=2,
                    edgecolor="yellow",
                    facecolor="none",
                )
                ax.add_patch(rect)
            ax.axis("off")
            heatmap_fname = f"neuron={idx}_heatmap_with_predboxes.png"
            fig.savefig(
                os.path.join(save_dir, f"class={label}", f"image_id={data_id}", heatmap_fname), bbox_inches="tight", pad_inches=0
            )
            plt.close()

   
    return results







def eval(args, alpha, latent_neurons, ):


    label = args.label
    dataset = "ViNDr"
    arch = "upmc_breast_clip_det_b5_period_n_ft"
    clip_chk_pt_path = "./Mammo-CLIP_weights/b5-model-best-epoch-7.tar"

    ckpt = torch.load(clip_chk_pt_path, map_location="cpu", weights_only=False)
    image_encoder_type = ckpt["config"]["model"]["image_encoder"]["name"]

    if args.label.lower() == "density":
        n_class = 4
    elif args.label.lower() == "birads":
        n_class = 3
    else:
        n_class = 1

    target_model = (
        BreastClipClassifier(
            ckpt=ckpt, n_class=n_class, image_encoder_type=image_encoder_type
        )
        .cuda()
        .eval()
    )

    downstream_classifier_chk_path = ckpt_dict[args.checkpoint_mode][args.label]

    weights = torch.load(downstream_classifier_chk_path, weights_only=False)["model"]
    target_model.load_state_dict(weights)
    train_loader, valid_loader = get_dataloader_RSNA(
        args, image_encoder_type, label, dataset, arch
    )

    pos_wt = torch.tensor([37.2967]).cuda()

    predictions_all = []
    labels_all = []
    target_model.eval()

    autoencoder_input_dim = args.autoencoder_input_dim_dict[
        args.ae_input_dim_dict_key[args.modality]
    ]
    n_learned_features = int(autoencoder_input_dim * args.expansion_factor)
    autoencoder = SparseAutoencoder(
        n_input_features=autoencoder_input_dim,
        n_learned_features=n_learned_features,
        n_components=len(args.hook_points),
    ).to(args.device)

    autoencoder = get_sae_ckpt(args, autoencoder)

    features = {}

    def hook_fn(module, input, output):

        batch_size, channels, height, width = output.shape

        if channels != autoencoder_input_dim:
            return

        batch_size, channels, height, width = output.shape
        flattened_features = output.permute(0, 2, 3, 1).reshape(-1, channels)
        concepts, reconstructions = autoencoder(flattened_features)

        concepts, reconstructions = concepts.squeeze(), reconstructions.squeeze()
        concepts = concepts.reshape(batch_size, height, width, -1)

        features["sae_concepts"] = concepts[:, :, :, latent_neurons].detach().cpu()

    if args.sae_layer == -1:
        args.sae_layer = len(target_model.image_encoder._blocks) - 1
        layer = target_model.image_encoder._blocks[args.sae_layer]
    elif args.sae_layer == 39:
        layer = target_model.image_encoder._swish
    else:
        layer = target_model.image_encoder._blocks[args.sae_layer]

    print(
        f"Adding hook to layer: {args.sae_layer} out of "
        f"{len(target_model.image_encoder._blocks)} layers"
    )

    hook = layer.register_forward_hook(hook_fn)
    print(f"Adding the hook to the layer: {args.sae_layer }")
    total = 0

    split = args.split

    if split == "val":
        loader = valid_loader

    elif split == "train":
        loader = train_loader

    mean = 0.3089279
    std = 0.25053555408335154

    if not os.path.exists(f"./results/visualization/{args.label}_{args.checkpoint_mode}"):
        os.makedirs(f"./results/visualization/{args.label}_{args.checkpoint_mode}")

    all_metrics = []
    save_dir = f"./results/visualization/{args.label}_{args.checkpoint_mode}"

    os.makedirs(
       save_dir, exist_ok=True
    )


    num_images = 0
    with torch.no_grad():
        for index, batch in enumerate(tqdm(loader)):

            inputs = batch["x"]

            assert inputs.shape[0] == 1, "Single batch only"

            inputs = inputs.squeeze(1).permute(0, 3, 1, 2).cuda()
            total += inputs.shape[0]
            img_paths = batch["img_path"]

            with torch.no_grad():
                y_preds = target_model(inputs)

            labels = batch["y"].float().cuda()

            gt_boxes = batch["bboxes"]
            concepts = features["sae_concepts"]
            data_ids = batch["data_id"]

       

            results = visualize_topk_concepts(
                inputs,
                concepts,
                batch["y"],
                mean,
                std,
                k=len(latent_neurons),
                latent_neurons=latent_neurons,
                batch_index=index,
                save_dir=save_dir + f"/visuals_mask_th={args.mask_threshold}",
                img_paths=img_paths,
                gt_boxes=gt_boxes,
                data_ids=data_ids,
                mask_threshold=args.mask_threshold,
            )

            results[data_ids[0]].update(
                {
                    "agg": {
                        "y_label": labels[0].cpu().tolist(),
                        "y_pred": y_preds.squeeze(1)
                        .sigmoid()
                        .to("cpu")
                        .numpy()
                        .tolist(),
                    }
                }
            )

            all_metrics.append(results)


            with open(
                save_dir + f"/image_wise_results_maskth={args.mask_threshold}.json", "w"
            ) as f:
                json.dump(all_metrics, f, indent=4)

            num_images += inputs.shape[0]
            if args.num_images> 0 and num_images > args.num_images:
                print(f"Exiting program as the max images: {num_images} finished!")
                break

            #exit()

        return results


if __name__ == "__main__":

    args = get_args()

    outpath = "./results/visualization"

    if args.label.lower() == "density":
        n_class = 4
    elif args.label.lower() == "birads":
        n_class = 3
    else:
        n_class = 2

    os.makedirs(outpath, exist_ok=True)


    all_classes_ranks = {}

    for class_idx in range(n_class):

        top_k_concept_path = os.path.join(
            "results",
            "class_wise_concepts",
            args.split_for_top_concepts,
            f"top_k_concepts_class={class_idx}_{args.label}_{args.checkpoint_mode}.json",
        )

        with open(top_k_concept_path, "r") as f:
            all_classes_ranks[class_idx] = json.load(f)


    k = 10
    latent_neurons = []
    for key in all_classes_ranks.keys():
        latent_neurons = latent_neurons + [
            info["index"] for info in all_classes_ranks[key][:k]
        ]
    latent_neurons = list(set(latent_neurons))

    print(
        f"Conducting topk_on intervention for latent neurons: {latent_neurons} "
        f" upto topk: {k}  ranks for class: all"
    )

    #latent_neurons = [13867,]
    # for neuron_index in latent_neurons:
    #     print("Running visualization ")
    #     result = eval(args, alpha=0.0, latent_neurons=[neuron_index])
    

    print(f"Latent Neurons: {latent_neurons}")
    result = eval(args, alpha=0.0, latent_neurons=latent_neurons)



    