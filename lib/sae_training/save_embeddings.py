import os
import torch
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from albumentations import *
from PIL import Image
from torch import nn
import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from breastclip.model.modules import load_image_encoder, LinearClassifier
from Datasets.dataset_utils import get_dataloader_RSNA
from metrics import (
    pfbeta_binarized,
    pr_auc,
    compute_auprc,
    auroc,
    compute_accuracy_np_array,
)

dir_path = os.path.dirname(os.path.realpath(__file__))


def get_args():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--downstream_classifier_chk_path",
    )
    parser.add_argument(
        "--label",
    )
    parser.add_argument("--sae_layer", type=int)
    parser.add_argument("--split", type=str)

    parser.add_argument("--num_images", type=int)

    args = parser.parse_args()
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

    if 1:  # args.img_size[0] == 1520 and args.img_size[1] == 912:
        return Compose(
            [
                HorizontalFlip(),
                VerticalFlip(),
                Affine(rotate=20, translate_percent=0.1, scale=[0.8, 1.2], shear=20),
                ElasticTransform(alpha=alpha, sigma=sigma),
            ],
            p=p,
        )
    else:
        return Compose(
            [
                Resize(width=int(width), height=int(height)),
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

        print(transform)

    def __len__(self):
        return len(self.df)

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

        return {
            "x": img.unsqueeze(0),
            "y": torch.tensor(data[self.label], dtype=torch.long),
            "img_path": str(img_path),
        }


def collator_mammo_dataset_w_concepts(batch):
    return {
        "x": torch.stack([item["x"] for item in batch]),
        "y": torch.from_numpy(
            np.array([item["y"] for item in batch], dtype=np.float32)
        ),
        "img_path": [item["img_path"] for item in batch],
    }


def get_dataloader_RSNA(args, image_encoder_type, label, dataset, arch):

    train_tfm = None
    val_tfm = None
    batch_size = 8
    num_workers = 0
    csv_file = "datasets/shantanughosh/vindr-mammogram-dataset-dicom-to-png/versions/1/vindr_detection_v1_folds.csv"
    data_dir = "/workspace/datasets"
    img_dir = "datasets/shantanughosh/vindr-mammogram-dataset-dicom-to-png/versions/1/images_png"
    mean = 0.3089279
    std = 0.25053555408335154

    df = pd.read_csv(os.path.join(data_dir, csv_file))
    df = df.fillna(0)

    train_folds = df[df["split"] == "training"].reset_index(drop=True)
    valid_folds = df[df["split"] == "test"].reset_index(drop=True)

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
    )

    return train_loader, valid_loader


def reshape_features(feats):

    print(feats.shape)

    batch_size, D, H, W = feats.shape

    feats = feats.permute(0, 2, 3, 1)
    feats = feats.reshape(-1, D)
    num_rows = feats.shape[0]
    indices = torch.randperm(num_rows)
    feats = feats[indices[: int(num_rows * 0.25)]]
    return feats


def eval(args):

    # label = "Suspicious_Calcification"

    label = args.label
    dataset = "ViNDr"
    arch = "upmc_breast_clip_det_b5_period_n_ft"
    clip_chk_pt_path = "./checkpoints/b5-model-best-epoch-7.tar"

    ckpt = torch.load(clip_chk_pt_path, map_location="cpu", weights_only=False)
    image_encoder_type = ckpt["config"]["model"]["image_encoder"]["name"]

    target_model = (
        BreastClipClassifier(
            ckpt=ckpt, n_class=1, image_encoder_type=image_encoder_type
        )
        .cuda()
        .eval()
    )


    # downstream_classifier_chk_path = "/workspace/Mammo-CLIP/checkpoints/Downstream_evalualtion_b5_fold0/classification/Models/Models/Classifier/fine_tune/calcification/upmc_breast_clip_det_b5_period_n_ft_seed_10_fold0_best_aucroc_ver084.pth"

    downstream_classifier_chk_path = args.downstream_classifier_chk_path

    weights = torch.load(downstream_classifier_chk_path, weights_only=False)["model"]
    target_model.load_state_dict(weights)
    train_loader, valid_loader = get_dataloader_RSNA(
        args, image_encoder_type, label, dataset, arch
    )

    pos_wt = torch.tensor([37.2967]).cuda()
    criterion = torch.nn.BCEWithLogitsLoss(reduction="mean", pos_weight=pos_wt)

    predictions_all = []
    labels_all = []
    target_model.eval()

    features = {}

    def hook_fn(module, input, output):
        features["embeddings"] = output.detach().cpu()

    if args.sae_layer == -1:
        args.sae_layer = len(target_model.image_encoder._blocks) - 1
        layer = target_model.image_encoder._blocks[args.sae_layer]

    elif args.sae_layer == 39:
        layer = target_model.image_encoder._swish
    else:
        layer = target_model.image_encoder._blocks[args.sae_layer]

    print(
        f"Adding hook to layer: {args.sae_layer} out of {len(target_model.image_encoder._blocks)} layers"
    )

    hook = layer.register_forward_hook(hook_fn)
    print(f"Adding the hook to the layer: {args.sae_layer }")
    total = 0

    out = None
    split = args.split

    if split == "val":
        loader = valid_loader

    elif split == "train":
        loader = train_loader

    with torch.no_grad():
        for index, batch in enumerate(tqdm(loader)):
            inputs = batch["x"]
            inputs = inputs.squeeze(1).permute(0, 3, 1, 2).cuda()
            total += inputs.shape[0]
            with torch.no_grad():
                y_preds = target_model(inputs)

            labels = batch["y"].float().cuda()
            loss = criterion(y_preds.view(-1, 1), labels.view(-1, 1))
            predictions_all.append(y_preds.squeeze(1).sigmoid().to("cpu").numpy())
            labels_all.append(labels.view(-1, 1).squeeze().to("cpu").numpy())

            if index == 0:
                print(
                    f"Inputs: {inputs.shape}, features: {features['embeddings'].shape}"
                )

            if out is None:
                out = reshape_features(features["embeddings"])
            else:
                feats = reshape_features(features["embeddings"])
                out = torch.vstack((out, feats))

            if args.num_images != -1 and total > args.num_images:
                break

        predictions_all = np.concatenate(predictions_all)
        labels_all = np.concatenate(labels_all)

        print(predictions_all.shape, labels_all.shape)
        aucroc = auroc(labels_all, predictions_all)

        print("auroc", aucroc)

        save_dir_activations = f"./results/activations/{dataset.lower()}/efficientb5/layer_{args.sae_layer}"
        os.makedirs(save_dir_activations, exist_ok=True)
        torch.save(
            out,
            os.path.join(save_dir_activations, f"{split}.pth"),
        )
        print(
            f"Save the activations of size: {out.shape} at layer: {args.sae_layer} to {save_dir_activations}"
        )


if __name__ == "__main__":

    args = get_args()
    eval(
        args,
    )
