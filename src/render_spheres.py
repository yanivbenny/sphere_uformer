import argparse
import os

import imageio
import numpy as np
import torch
import wandb
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from data.stanford2d3d import Stanford2D3D
from network.sphere_model import SphereUFormer
from visualization import SphereVisualizer, ViewPoint


parser = argparse.ArgumentParser(description="360 Degree Depth Estimation Training")
parser.add_argument("--task", type=str, default="depth", choices=["depth", "segmentation"])

parser.add_argument("--dataset_root_dir", type=str)

parser.add_argument("--wandb_entity", type=str)
parser.add_argument("--wandb_project", type=str)
parser.add_argument("--wandb_task", type=str)

LOAD_WEIGHTS_DIR = "./artifacts"

API = wandb.Api()

RENDER_RGB = True
START_FROM = 0
END_AT = 40


def load_model(model, entity, project_name, task_id):
    pretrained_path = os.path.join(LOAD_WEIGHTS_DIR, task_id, "model.pth")
    if not os.path.isfile(pretrained_path):
        load_run = API.run(f"{entity}/{project_name}/{task_id}")
        artifacts = load_run.logged_artifacts()
        model_art = [art for art in artifacts if art.type == "model" and "latest" in art.aliases][0]
        model_art.download(os.path.join(LOAD_WEIGHTS_DIR, task_id))

    pretrained_state_dict = torch.load(pretrained_path)
    model.load_state_dict(pretrained_state_dict)


def get_dataset(data_root_dir):  # 52
    dataset = Stanford2D3D(
            root_dir=data_root_dir,
            list_file="./data/splits_2d3d/stanford2d3d_val.txt",
            dataset_kwargs={"sphere_rank": 7, "grid_width": 512, "sphere_node_type": "vertex"},
            augmentation_kwargs=None,
            is_training=False,
        )
    return dataset


def main(args):
    dataset = get_dataset(args.dataset_root_dir)

    model = SphereUFormer(
            img_rank=7,
            node_type="vertex",
            out_channels=1 if args.task=="depth" else dataset.NUM_CLASSES,
            in_scale_factor=2,
            win_size_coef=2,
            d_head_coef=2,
            enc_num_heads=[2,4,8,16],
            dec_num_heads=[16, 16, 8, 4],
            downsample="center",
            upsample="interpolate",
        )
    load_model(model, args.wandb_entity, args.wandb_project, args.wandb_task)
    model.eval()

    dataset_ = Subset(dataset, range(START_FROM, len(dataset)))

    dataloader = DataLoader(
        dataset_,
        batch_size=1,
        num_workers=1,
        shuffle=False, drop_last=False, pin_memory=False, persistent_workers=True,
    )

    visualizer = SphereVisualizer(rank=7, node_type="vertex", depth_color_map="turbo", depth_invert=True, sem_colors=dataset.sem_colors)

    model.cuda()

    for i, batch_image in enumerate(tqdm(dataloader), start=START_FROM):
        if i == END_AT:
            break

        if RENDER_RGB:
            rgb_image = batch_image["sphere_rgb"][0].numpy()
            rgb_image = (visualizer.vertices_to_faces(rgb_image) * 255).astype(np.uint8)
            render_rgb(visualizer, rgb_image, i, resolution=640)

        mask = batch_image["sphere_valid_mask"].cuda()
        valid_mask_0 = mask[0].cpu().numpy().astype(np.bool_)
        valid_mask_0 = visualizer.mask_vertices_to_faces(valid_mask_0)

        x = batch_image["normalized_sphere_rgb"].cuda()

        with torch.no_grad():
            pred = model(x)
        pred = pred.squeeze(2)

        if args.task == "depth":
            masked_pred = pred * mask
            masked_pred_0 = masked_pred[0].clip(min=0, max=1).cpu().numpy()
            masked_pred_0 = (visualizer.vertices_to_faces(masked_pred_0) * 255).astype(np.uint8)
            render_depth(visualizer, masked_pred_0, valid_mask_0, i, resolution=640)
        else:
            pred_0 = pred[0].softmax(-1).cpu().numpy()
            pred_0 = visualizer.vertices_to_faces(pred_0)
            pred_0 = pred_0[:, 1:].argmax(-1) + 1
            render_sem(visualizer, pred_0, valid_mask_0, i, resolution=640)


def render_rgb(visualizer, rgb_image, i, resolution):
    visualizer.reset_mesh()
    visualizer.put_rgb_data(rgb_image)
    visualizer.set_viewpoint(ViewPoint.side1)
    rgb_rend_view1 = visualizer.render(resolution=(resolution, resolution), smooth=True)
    visualizer.set_viewpoint(ViewPoint.side2)
    rgb_rend_view2 = visualizer.render(resolution=(resolution, resolution), smooth=True)
    visualizer.set_viewpoint(ViewPoint.side3)
    rgb_rend_view3 = visualizer.render(resolution=(resolution, resolution), smooth=True)
    visualizer.set_viewpoint(ViewPoint.side4)
    rgb_rend_view4 = visualizer.render(resolution=(resolution, resolution), smooth=True)

    rgb_rend_all_views = np.concatenate(
        (rgb_rend_view1, rgb_rend_view2, rgb_rend_view3, rgb_rend_view4),
        axis=1)
    save_path = f"./images/stanford2d/rgb/{i:04d}.png"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    imageio.imwrite(save_path, rgb_rend_all_views)


def render_depth(visualizer, depth_image, mask, i, resolution):
    visualizer.reset_mesh()
    visualizer.put_depth_data(depth_image, mask)
    visualizer.set_viewpoint(ViewPoint.side1)
    depth_rend_view1 = visualizer.render(resolution=(resolution, resolution))
    visualizer.set_viewpoint(ViewPoint.side2)
    depth_rend_view2 = visualizer.render(resolution=(resolution, resolution))
    visualizer.set_viewpoint(ViewPoint.side3)
    depth_rend_view3 = visualizer.render(resolution=(resolution, resolution))
    visualizer.set_viewpoint(ViewPoint.side4)
    depth_rend_view4 = visualizer.render(resolution=(resolution, resolution))

    depth_rend_all_views = np.concatenate((depth_rend_view1, depth_rend_view2, depth_rend_view3, depth_rend_view4),
                                          axis=1)

    save_path = f"./images/stanford2d/depth/{i:04d}.png"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    imageio.imwrite(save_path, depth_rend_all_views)


def render_sem(visualizer, semantic_image, mask, i, resolution):
    visualizer.reset_mesh()
    visualizer.put_semantic_data(semantic_image, mask)
    visualizer.set_viewpoint(ViewPoint.side1)
    sem_rend_view1 = visualizer.render(resolution=(resolution, resolution))
    visualizer.set_viewpoint(ViewPoint.side2)
    sem_rend_view2 = visualizer.render(resolution=(resolution, resolution))
    visualizer.set_viewpoint(ViewPoint.side3)
    sem_rend_view3 = visualizer.render(resolution=(resolution, resolution))
    visualizer.set_viewpoint(ViewPoint.side4)
    sem_rend_view4 = visualizer.render(resolution=(resolution, resolution))

    semantic_rend_all_views = np.concatenate((sem_rend_view1, sem_rend_view2, sem_rend_view3, sem_rend_view4),
                                          axis=1)

    save_path = f"./images/stanford2d/semantic/{i:04d}.png"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    imageio.imwrite(save_path, semantic_rend_all_views)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
