import os
import torch
import os
import json
from torch.utils.data import TensorDataset
from dncbm import arg_parser
from sparse_autoencoder import SparseAutoencoder
from tqdm import tqdm
import os.path as osp
from torchvision import models
from scripts.lib.load_image_dataset import ImageDataset
from torchvision import models, datasets, transforms, utils
from dncbm.utils import common_init, get_sae_ckpt
from torch.utils.data import DataLoader
from PIL import Image, ImageDraw


def save(
    image_paths,
    labels_all,
    pred_labels_all,
    neuron,
    top_k_importance,
    outpath,
    embeddings_name,
):
    transform = transforms.Compose(
        [transforms.Resize((128, 128)), transforms.ToTensor()]
    )
    image_tensors = []
    for img_path, label, pred_label, score in zip(
        image_paths, labels_all, pred_labels_all, top_k_importance
    ):
        img = Image.open(
            os.path.join("/workspace/datasets/imagenet", img_path)
        ).convert("RGB")
        img = transform(img)
        img_pil = transforms.ToPILImage()(img)
        img_pil = img_pil.convert("RGB")
        draw = ImageDraw.Draw(img_pil)

        draw.text(
            (20, 20),
            str(label.item()) + ", " + str(round(score.item(), 2)),
            font=None,
            fill="white",
        )

        border_color = "green" if label == pred_label else "red"
        border_width = 5  # Thickness of the border

        draw.rectangle(
            [(0, 0), (img_pil.width - 1, img_pil.height - 1)],
            outline=border_color,
            width=border_width,
        )

        img_tensor = transforms.ToTensor()(img_pil)
        image_tensors.append(img_tensor)

    image_grid = torch.stack(image_tensors)
    grid_image = utils.make_grid(image_grid, nrow=10)

    if not os.path.exists(os.path.join(outpath, "top_k_images_per_latent")):
        os.makedirs(os.path.join(outpath, "top_k_images_per_latent"))

    savepath = os.path.join(
        outpath,
        "top_k_images_per_latent",
        f"top_k_images_neuron={neuron}_{embeddings_name}.png",
    )
    utils.save_image(grid_image, savepath)
    print(f"Image grid saved for neuron: {neuron}")


def plot_top_k_images(data, neuron, outpath, embeddings_name):
    K = 100
    image_strenth = data["sae_concepts"][:, neuron]
    top_k_importance, top_K_indices = torch.topk(image_strenth, k=100)
    image_paths = [data["img_paths"][x] for x in top_K_indices]
    labels_all = [data["labels"][x] for x in top_K_indices]
    pred_labels_all = [data["pred_labels"][x] for x in top_K_indices]

    save(
        image_paths,
        labels_all,
        pred_labels_all,
        neuron,
        top_k_importance,
        outpath,
        embeddings_name,
    )


def find_top_concepts_by_class(data, label, outpath, embeddings_name):

    indexes = data["labels"] == label
    sae_concepts_class_wise = data["sae_concepts"][indexes]
    concept_importance = torch.mean(sae_concepts_class_wise, dim=0).abs()
    top_k_importance, top_K_indices = torch.topk(
        concept_importance, k=sae_concepts_class_wise.shape[1]
    )

    if not os.path.exists(
        os.path.join(outpath, "class_wise_concepts", embeddings_name)
    ):
        os.makedirs(os.path.join(outpath, "class_wise_concepts", embeddings_name))

    result = []

    for rank, (index, importance) in enumerate(zip(top_K_indices, top_k_importance)):
        result.append(
            {"rank": rank, "index": index.item(), "importance": importance.item()}
        )

    with open(
        os.path.join(
            outpath,
            "class_wise_concepts",
            embeddings_name,
            f"top_k_concepts_class={label}.json",
        ),
        "w",
    ) as f:
        json.dump(result, f, indent=4)


def find_top_concepts_by_classagnostic(data, label, outpath, embeddings_name):

    sae_concepts_class_wise = data["sae_concepts"]
    concept_importance = torch.mean(sae_concepts_class_wise, dim=0).abs()
    top_k_importance, top_K_indices = torch.topk(
        concept_importance, k=sae_concepts_class_wise.shape[1]
    )

    if not os.path.exists(
        os.path.join(outpath, "class_wise_concepts", embeddings_name)
    ):
        os.makedirs(os.path.join(outpath, "class_wise_concepts", embeddings_name))

    result = []

    for rank, (index, importance) in enumerate(zip(top_K_indices, top_k_importance)):
        result.append(
            {"rank": rank, "index": index.item(), "importance": importance.item()}
        )

    with open(
        os.path.join(
            outpath,
            "class_wise_concepts",
            embeddings_name,
            f"top_k_concepts_class=all.json",
        ),
        "w",
    ) as f:
        json.dump(result, f, indent=4)


def main(args):

    embeddings_name = args.probe_split
    outpath = args.save_dir_sae_ckpts[args.modality]

    data = torch.load(
        os.path.join(outpath, f"sae_concepts_imagenet_{embeddings_name}_global.pth")
    )

    # for neuron in range(data["sae_concepts"].shape[1]):
    #     plot_top_k_images(data, neuron, outpath, embeddings_name)

    find_top_concepts_by_classagnostic(data, "all", outpath, embeddings_name)

    # for label in tqdm(range(1000)):
    #     find_top_concepts_by_class(data, label, outpath, embeddings_name)


if __name__ == "__main__":
    parser = arg_parser.get_common_parser()
    args = parser.parse_args()
    common_init(args)
    main(
        args,
    )
