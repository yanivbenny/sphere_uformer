from __future__ import absolute_import, division, print_function

import math
import argparse


parser = argparse.ArgumentParser(description="360 Degree Depth Estimation Training")

# system settings
parser.add_argument("--num_workers", type=int, default=4, help="number of dataloader workers")

# data settings
parser.add_argument("--task", type=str, default="depth", choices=["depth", "segmentation"])
parser.add_argument("--dataset_name", type=str, default="stanford2d3d")
parser.add_argument("--dataset_root_dir", type=str, help="root location for the data")

# model settings
parser.add_argument("--mode", type=str, default="vertex", choices=["face", "vertex"], help="folder to save the model in")
parser.add_argument("--img_rank", type=int, default=7)
parser.add_argument("--img_width", type=int, default=512)
parser.add_argument("--num_scales", type=int, default=4)
parser.add_argument("--win_size_coef", type=int, default=2)
parser.add_argument("--scale_factor", type=int, default=2)
parser.add_argument("--abs_pos_enc_in", type=int, default=True)
parser.add_argument("--abs_pos_enc", type=int, default=True)
parser.add_argument("--rel_pos_bias", type=int, default=True)
parser.add_argument("--rel_pos_bias_size", type=int, default=7)
parser.add_argument("--rel_pos_init_variance", type=float, default=1)
parser.add_argument("--d_head_coef", type=int, default=2)
parser.add_argument("--enc_num_heads", nargs="+", type=int, default=[2,4,8,16])
parser.add_argument("--dec_num_heads", nargs="+", type=int, default=[16,16,8,4])
parser.add_argument("--bottleneck_num_heads", type=int, default=None)
parser.add_argument("--scale_depth", type=int, default=2)
parser.add_argument("--debug_skip_attn", type=int, default=False)
parser.add_argument("--append_self", type=int, default=False)
parser.add_argument("--use_checkpoint", type=int, default=True)

parser.add_argument("--dr", type=float, default=0.)
parser.add_argument("--dpr", type=float, default=0.)
parser.add_argument("--adr", type=float, default=0.)
parser.add_argument("--aodr", type=float, default=0.)
parser.add_argument("--posdr", type=float, default=0.)

# parser.add_argument("--abs_pos_enc_in", nargs="+", type=str, default=None)
# parser.add_argument("--abs_pos_enc", nargs="+", type=str, default=None)

parser.add_argument("--downsample", type=str, default="center")
parser.add_argument("--upsample", type=str, default="interpolate")

# optimization settings
parser.add_argument("--learning_rate", type=float, default=1e-4, help="learning rate")
parser.add_argument("--ltr", dest="limit_train_batches", type=int, default=math.inf, help="limit train batches")
parser.add_argument("--train_batch_size", type=int, default=16, help="batch size")
parser.add_argument("--val_batch_size", type=int, default=10, help="batch size")
parser.add_argument("--num_epochs", type=int, default=400, help="number of epochs")
parser.add_argument("--accum_grads", type=int, default=1, help="number of epochs")


# loading and logging settings
parser.add_argument("--log_frequency", type=int, default=100, help="number of batches between each tensorboard log")
parser.add_argument("--enable_save", type=int, default=True, help="save model")
parser.add_argument("--save_frequency", type=int, default=10, help="number of epochs between each save")
parser.add_argument("--load_weights_task", type=str, default=None)

# data augmentation settings
parser.add_argument("--disable_color_augmentation",  dest="color_augmentation", action="store_false",
                    help="if set, do not use color augmentation")
parser.add_argument("--disable_lr_flip_augmentation", dest="lr_flip_augmentation", action="store_false",
                    help="if set, do not use left-right flipping augmentation")
parser.add_argument("--disable_yaw_rotation_augmentation", dest="yaw_rotation_augmentation", action="store_false",
                    help="if set, do not use yaw rotation augmentation")

# wandb settings
parser.add_argument("--exp_name", default="train_sphereuformer", type=str)
parser.add_argument("--log_dir", type=str, help="log directory")
parser.add_argument("--wandb_entity", type=str)
parser.add_argument("--wandb_project", type=str)
parser.add_argument("--wandb_group", default=None, type=str)


parser.add_argument("--vis_color_map", type=str, default="viridis", help="color map for depth visualization")
parser.add_argument("--vis_color_map_invert", action="store_true", help="invert color map for depth visualization")


parser.add_argument("--no_gpu", dest="use_gpu", action="store_false")
parser.add_argument("--test", dest="test", action="store_true")

args = parser.parse_args()


def main():
    if args.task == "depth":
        from trainer_dep import Trainer
    elif args.task == "segmentation":
        from trainer_seg import Trainer
    else:
        raise NotImplementedError(args.task)

    trainer = Trainer(args)

    if not args.test:
        trainer.train()
    else:
        trainer.test()


if __name__ == "__main__":
    main()
