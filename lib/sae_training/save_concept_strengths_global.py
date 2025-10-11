import os
import torch
import os
import numpy as np
import random
from torch.utils.data import TensorDataset
from dncbm import arg_parser
from sparse_autoencoder import SparseAutoencoder
from tqdm import tqdm
import os.path as osp
from torchvision import models
from scripts.lib.load_image_dataset import ImageDataset
from torchvision import models, datasets, transforms
from dncbm.utils import common_init, get_sae_ckpt
from torch.utils.data import DataLoader


def set_seed(seed):

    np.random.seed(seed)
    random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main(args):

    model = models.vgg16(pretrained=True).cuda().eval()
    tranforms = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    dataset = ImageDataset(
        data_dir="/workspace/datasets/imagenet/",
        images_path=f"/workspace/DaN/dataset/imagenet_{args.probe_split}.txt",
        split="",
        convolve_image=False,
        transform=tranforms,
        target_transform=None,
        keep_difficult=False,
        is_train=True,
        data_aug=None,
    )

    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)

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

    def hook_function_sae(module, input, output):

        batch_size, channels, height, width = output.shape
        flattened_features = output.permute(0, 2, 3, 1).reshape(-1, channels)
        concepts, reconstructions = autoencoder(flattened_features)
        concepts, reconstructions = concepts.squeeze(), reconstructions.squeeze()
        concepts = concepts.reshape(batch_size, height, width, -1)
        concepts = concepts.permute(0, 3, 1, 2)
        concepts = torch.mean(concepts, (2, 3))
        features["sae_concepts"] = concepts.detach().cpu()

        if True:
            reconstructions = reconstructions.reshape(
                batch_size, height, width, channels
            )
            reconstructions = reconstructions.permute(0, 3, 1, 2)
            return reconstructions

    layer = model.features[18]
    hook = layer.register_forward_hook(hook_function_sae)

    total = 0
    correct = 0
    all_concepts = None
    labels_all = None
    img_paths_all = None
    pred_labels_all = None

    with torch.no_grad():
        for batch_index, (inputs, labels, img_paths) in enumerate(
            tqdm(dataloader, desc="Processing batches", unit="batch")
        ):

            inputs = inputs.cuda()
            labels = labels.cuda()

            logits = model(inputs)
            pred_labels = logits.argmax(dim=-1)

            correct += (pred_labels == labels).sum().item()
            total += inputs.shape[0]

            concepts = features["sae_concepts"]

            if batch_index < 5:
                print(f"Batch: {batch_index}, labels: {labels}")

            if all_concepts is None:
                all_concepts = concepts.detach().cpu()
                labels_all = labels.cpu().numpy().tolist()
                img_paths_all = list(img_paths)
                pred_labels_all = pred_labels.cpu().numpy().tolist()

            else:
                all_concepts = torch.vstack((all_concepts, concepts.detach().cpu()))
                labels_all.extend(labels.cpu().numpy().tolist())
                img_paths_all.extend(img_paths)
                pred_labels_all.extend(pred_labels.cpu().numpy().tolist())

            print(
                f"Total: {total}, Correct: {correct}, Accuracy: {100*correct/total:.2f}%"
            )

    labels_all = torch.tensor(labels_all)
    pred_labels_all = torch.tensor(pred_labels_all)

    outpath = args.save_dir_sae_ckpts[args.modality]
    torch.save(
        {
            "sae_concepts": all_concepts,
            "labels": labels_all,
            "img_paths": img_paths_all,
            "pred_labels": pred_labels_all,
        },
        os.path.join(outpath, f"sae_concepts_imagenet_{args.probe_split}_global.pth"),
    )


if __name__ == "__main__":
    parser = arg_parser.get_common_parser()
    args = parser.parse_args()
    common_init(args)

    set_seed(42)

    main(
        args,
    )
