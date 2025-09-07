import os
import torch
import pandas as pd
import numpy as np
import cv2
import json

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
from lib.breastclip.model.modules import load_image_encoder, LinearClassifier
from lib.Datasets.dataset_utils import get_dataloader_RSNA

from lib.metrics.metrics import (
    auroc,
)

from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
)


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

        print("Freezing image encoder to not be trained")
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
    num_workers = 4

    # adjust the paths
    # borrowed as it is from Mammoclip paper
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
        prefetch_factor=4,
    )

    return train_loader, valid_loader


def reshape_features(feats):

    batch_size, D, H, W = feats.shape

    feats = feats.permute(0, 2, 3, 1)
    feats = feats.reshape(-1, D)
    num_rows = feats.shape[0]
    indices = torch.randperm(num_rows)
    feats = feats[indices[: int(num_rows * 0.25)]]
    return feats


def eval(args, alpha, latent_neurons):

    label = args.label
    dataset = "ViNDr"
    arch = "upmc_breast_clip_det_b5_period_n_ft"

    # path to foundation model to retrieve few variables
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

    # load the attribute classifier
    downstream_classifier_chk_path = ckpt_dict[args.checkpoint_mode][args.label]
    print(f"Downstream Classifier weights: {downstream_classifier_chk_path}")
    weights = torch.load(downstream_classifier_chk_path, weights_only=False)["model"]
    target_model.load_state_dict(weights)
    train_loader, valid_loader = get_dataloader_RSNA(
        args, image_encoder_type, label, dataset, arch
    )

    labels_all = []
    target_model.eval()

    # load autoencoder
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

    # hook to extract the features
    def hook_fn(module, input, output):

        batch_size, channels, height, width = output.shape

        if channels != autoencoder_input_dim:
            return

        batch_size, channels, height, width = output.shape
        flattened_features = output.permute(0, 2, 3, 1).reshape(-1, channels)
        concepts, reconstructions = autoencoder(flattened_features)

        N = concepts.size(2)
        mask = torch.ones(N, device=concepts.device, dtype=torch.bool)
        mask[latent_neurons] = False

        # all positions except latent_neuron indexes are set to zero
        concepts[:, :, mask] = alpha

        x = autoencoder.decoder(concepts)
        reconstructions = autoencoder.post_decoder_bias(x)

        reconstructions = reconstructions.view(batch_size, height, width, channels)
        reconstructions = reconstructions.permute(0, 3, 1, 2)
        return reconstructions

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
    total = 0

    split = args.split

    if split == "val":
        loader = valid_loader

    elif split == "train":
        loader = train_loader

    else:
        raise NotImplementedError()

    pred_labels_all = None

    with torch.no_grad():
        for index, batch in enumerate(tqdm(loader)):
            inputs = batch["x"]
            inputs = inputs.squeeze(1).permute(0, 3, 1, 2).cuda()
            total += inputs.shape[0]

            img_paths = batch["img_path"]

            with torch.no_grad():
                y_preds = target_model(inputs)

            labels = batch["y"].float().cuda()

            if pred_labels_all is None:
                labels_all = labels.view(-1, 1).squeeze().cpu().numpy().tolist()
                img_paths_all = list(img_paths)

                if args.label in ["Mass", "Suspicious_Calcification"]:
                    pred_labels_all = (
                        y_preds.squeeze(1).sigmoid().to("cpu").numpy().tolist()
                    )
                elif args.label in ["density"]:
                    _, y_preds_labels = torch.max(y_preds, 1)
                    pred_labels_all = y_preds_labels.to("cpu").numpy().tolist()

            else:
                labels_all.extend(labels.view(-1, 1).squeeze().cpu().numpy().tolist())
                img_paths_all.extend(img_paths)

                if args.label in ["Mass", "Suspicious_Calcification"]:
                    pred_labels_all.extend(
                        y_preds.squeeze(1).sigmoid().to("cpu").numpy().tolist()
                    )
                elif args.label in ["density"]:
                    _, y_preds_labels = torch.max(y_preds, 1)
                    pred_labels_all.extend(y_preds_labels.to("cpu").numpy().tolist())

            if index == 0:
                print(f"Inputs: {inputs.shape}")

            if args.num_images != -1 and total > args.num_images:
                break

        pred_labels_all = np.array(pred_labels_all)
        labels_all = np.array(labels_all)

        print(f"Predictions: {pred_labels_all.shape}, Labels: {labels_all.shape}")

        results = {}
        results["alpha"] = alpha
        results["topk"] = len(latent_neurons)

        # compute the metrics
        if args.label in ["Mass", "Suspicious_Calcification"]:
            aucroc = auroc(labels_all, pred_labels_all)
            pred_binary = (pred_labels_all >= 0.5).astype(int)
            accuracy = 100 * accuracy_score(labels_all, pred_binary)

            auprc = average_precision_score(labels_all, pred_labels_all)
            f1 = f1_score(labels_all, pred_binary)
            precision = precision_score(labels_all, pred_binary)
            recall = recall_score(labels_all, pred_binary)

            results["aucroc"] = aucroc
            results["auprc"] = auprc
            results["precision"] = precision
            results["recall"] = recall
            results["f1"] = f1

        elif args.label in ["density"]:
            accuracy = 100 * np.mean(pred_labels_all == labels_all)
            f1 = f1_score(pred_labels_all, labels_all, average="macro")
            results["accuracy"] = accuracy
            results["f1"] = f1

        else:
            raise NotImplementedError()

        print(f"Final Results: {results}")

        return results


if __name__ == "__main__":

    args = get_args()
    outpath = args.save_dir_sae_ckpts[args.modality]

    outpath = "./results/topk_on_intervention"  # args.save_dir_sae_ckpts[args.modality]

    os.makedirs(outpath, exist_ok=True)

    if args.label.lower() == "density":
        n_class = 4
    elif args.label.lower() == "birads":
        n_class = 3

    # for mass and calcification
    else:
        n_class = 2

    if not os.path.exists(
        os.path.join(
            outpath, f"topk_on_intervention_{args.label}_{args.checkpoint_mode}"
        )
    ):
        os.makedirs(
            os.path.join(
                outpath, f"topk_on_intervention_{args.label}_{args.checkpoint_mode}"
            )
        )

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

    # vary the topk intervention sizes
    topks = [0, 1, 5, 10, 50, 100, 250, 500, 1000, 5000, 10000, 16384]
    results = []

    for k in topks:

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

        result = eval(args, alpha=0.0, latent_neurons=latent_neurons)
        results.append(result)
        with open(
            os.path.join(
                outpath,
                f"topk_on_intervention_{args.label}_{args.checkpoint_mode}",
                f"latent_neurons_topkon_class=all.json",
            ),
            "w",
        ) as f:
            json.dump(results, f, indent=4)
