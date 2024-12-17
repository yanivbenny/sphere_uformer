# SphereUFormer
This repo contains the code for "**SphereUFormer: A U-Shaped Transformer for Spherical 360 Perception**".

## Setup

1. Setup environment:
```
conda create -n sphereformer python=3.8
conda activate sphereformer
pip install -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html
```

2. Download the data (stanford2d3d) to your preferred folder <DATA_DIR>

3. Recommended: Set up weights & biases account
 
## Train the model
1. enter `./src` folder
2. Run the command:

```python train.py --task <TASK> --dataset_root_dir <DATA_DIR>```

Notes:
- For task, select ["depth", "segmentation"].
- If you wish to use weights & biases. add `--wandb_entity <WANDB_ENTITY> --wandb_project <WANDB_PROJECT>`.

## Render results
To render results, run:

```python render_spheres.py --task <TASK> --dataset_door_dir <DATA_DIR> --wandb_entity <WANDB_ENTITY> --wandb_project <WANDB_PROJECT> --wandb_task <WANDB_TASK>```

Note that for rendering, training with weights & biases is required. 
Please note that rendering can only be performed on system with an X server.
