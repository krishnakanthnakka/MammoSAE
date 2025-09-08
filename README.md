
<h2 align="center"> Mammo-SAE: Interpreting Breast Cancer Concept Learning with Sparse Autoencoders [Deep Breast Imaging Workshop, MICCAI 2025 🔥]</h2>

![](https://i.imgur.com/waxVImv.png)

[Krishna Kanth Nakka](https://krishnakanthnakka.github.io/)* 

**Munich, Bavaria, Germany**

<h5 align="center"> If you like our project, please give us a star ⭐ on GitHub for the latest update.</h5>


## 📣 Latest Updates
- **Sep 7-2025**: Partial Code Release. Working to update remaining parts of the code.


## 🔥 Highlights
- **Mammo-SAE**:  First SAE trained on Mammogram Vision-Language Foundation Model to interpret breast concepts such as mass, calcification.
- 



---

### Installation

-  To install all required dependencies, run:

    ```sh
    pip install -r requirments.txt
    ```


---

### Checkpoints

- **Sparse Autoencoder (SAE):** Pretrained SAE checkpoints are included with this repository.  

    ```sh

    # minimal example to download the MammoSAE
    ckpt_path = hf_hub_download(
        repo_id=f"KKNakka/MammoSAE",
        filename=f"sparse_autoencoder_final.pt",
        repo_type="model"
    )

    print(f"Loading SAE checkpoint from: {ckpt_path}")
    state_dict = torch.load(ckpt_path)
    autoencoder.load_state_dict(state_dict)

    ```

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

### Visualization of Latent Neurons

- To visualize the top-k latent neurons, please run:

    ```sh
    bashs scripts/visualizations.sh
    ```





## 📝 Citation

If you find this paper useful, please consider staring 🌟 this repo and citing 📑 our paper:

```

@article{nakka2025mammo,
  title={Mammo-SAE: Interpreting Breast Cancer Concept Learning with Sparse Autoencoders},
  author={Nakka, Krishna Kanth},
  journal={arXiv preprint arXiv:2507.15227},
  year={2025}
}


```

## 🙏 Acknowledgement
- This project is built upon [Mammo-CLIP](https://arxiv.org/abs/2405.12255) and [VisionSAE]() codebases. Thank you to both of them for open sourcing their codebases.