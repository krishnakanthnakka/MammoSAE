## Mammo-SAE


### Introduction




---

### Installation

-  To install all required dependencies, run:

    ```sh
    pip install -r requirments.txt
    ```


---

### Checkpoints

- **Sparse Autoencoder (SAE):** Pretrained SAE checkpoints are included with this repository.  
- **MammoCLIP:** Download the pretrained MammoCLIP checkpoints from Hugging Face and place them in the `Mammo_CLIP_weights/` directory.  

---


### SAE Performance


-  We reconstruct the local features at the last layer of the Mammo-CLIP backbone with SAE using

    ```sh
    bash scripts/eval_with_sae_reconstruction.sh
    ```


### Launch Intervention

- We provide scripts to reproduce Figure 2 using the commands below. 

#### Intervention: Top-k Latent Neurons Activated

- To launch an intervention that activates only the `top-k` neurons, run:

    ```sh
    bash scripts/topk_activate_latent_neuron_intervention.sh
    ```


#### Intervention: Top-k Latent Neurons Deactivated

- To launch an intervention that deactivates only the top-k neurons, run:
    ```sh

    bash scripts/topk_deactivate_latent_neuron_interventions.sh

    ```

### Spatial Alignment b/w Latent Neurons and True Regions

- To compute the spatial alignment of the heatmap with ground-truth regions, please run:

    ```sh
    bashs scripts/compute_iou_alignment.sh
    ```