import os
import time
import warnings

import torch
import torch.nn as nn
import tqdm
import wandb

from data.get_dataloaders import get_dataloaders
from metrics.segmentation import Evaluator
from network.sphere_model import SphereUFormer


class Trainer:
    def __init__(self, args):
        self.args = args

        self.device = torch.device("cuda") if args.use_gpu else torch.device("cpu")

        self.config = dict(
            img_rank=self.args.img_rank,
            img_width=self.args.img_width,
            node_type=self.args.mode,
            num_scales=self.args.num_scales,
            win_size_coef=self.args.win_size_coef,
            scale_factor=self.args.scale_factor,
            downsample=self.args.downsample,
            scale_depth=self.args.scale_depth,
        )

        sphere_img_rank = self.args.img_rank
        grid_img_width = self.args.img_width

        self.wandb_run = None
        os.makedirs(args.log_dir, exist_ok=True)
        if args.wandb_project:
            self.wandb_run = wandb.init(
                entity=args.wandb_entity,
                project=args.wandb_project,
                name=args.exp_name,
                group=args.wandb_group,
                dir=args.log_dir,
            )
            self._run = wandb.Api().run(f"{args.wandb_entity}/{args.wandb_project}/{self.wandb_run.id}")

        # Configure data
        self.loader_train, self.loader_val = get_dataloaders(
            dataset_name=self.args.dataset_name,
            dataset_root_dir=self.args.dataset_root_dir,
            dataset_kwargs={
                "sphere_rank": sphere_img_rank,
                "grid_width": grid_img_width,
                "sphere_node_type": self.config["node_type"],
            },
            augmentation_kwargs=dict(
                color_augmentation=False,
                lr_flip_augmentation=self.args.lr_flip_augmentation,
                yaw_rotation_augmentation=self.args.yaw_rotation_augmentation,
            ),
            train_batch_size=self.args.train_batch_size,
            val_batch_size=self.args.val_batch_size or self.args.train_batch_size,
            num_workers=self.args.num_workers,
            pin_memory=False,
        )

        self.model = SphereUFormer(
            img_rank=sphere_img_rank,
            node_type=self.config["node_type"],
            in_channels=3,
            out_channels=self.loader_train.dataset.NUM_CLASSES,
            in_scale_factor=self.args.scale_factor,
            num_scales=self.args.num_scales,
            win_size_coef=self.args.win_size_coef,
            enc_depths=self.args.scale_depth,
            dec_depths=self.args.scale_depth,
            bottleneck_depth=self.args.scale_depth,
            d_head_coef=self.args.d_head_coef,
            enc_num_heads=self.args.enc_num_heads,
            bottleneck_num_heads=self.args.bottleneck_num_heads,
            dec_num_heads=self.args.dec_num_heads,
            #
            abs_pos_enc_in=self.args.abs_pos_enc_in,
            abs_pos_enc=self.args.abs_pos_enc,
            rel_pos_bias=self.args.rel_pos_bias,
            rel_pos_bias_size=self.args.rel_pos_bias_size,
            rel_pos_init_variance=self.args.rel_pos_init_variance,
            downsample=self.args.downsample,
            upsample=self.args.upsample,
            #
            drop_rate=self.args.dr,
            drop_path_rate=self.args.dpr,
            attn_drop_rate=self.args.adr,
            attn_out_drop_rate=self.args.aodr,
            pos_drop_rate=self.args.posdr,
            #
            debug_skip_attn=self.args.debug_skip_attn,
            append_self=self.args.append_self,
            use_checkpoint=self.args.use_checkpoint,
        )

        total_params = sum(p.numel() for p in self.model.parameters())
        if self.wandb_run:
            wandb.log({f"total_params": total_params}, step=0)

        self.model.to(self.device)
        self.parameters_to_train = list(self.model.parameters())

        self.optimizer = torch.optim.Adam(self.parameters_to_train, self.args.learning_rate)

        if self.args.load_weights_task is not None:
            self.load_model()

        print("Training is using:\n ", self.device)
        print("Total parameters:\n ", total_params)

        self.compute_loss = nn.CrossEntropyLoss(ignore_index=0)
        self.evaluator = Evaluator(num_classes=self.loader_train.dataset.NUM_CLASSES)

        if self.wandb_run:
            self.wandb_run.config.update(self.config)

    def inputs_to_device(self, inputs):
        keys = inputs.keys()
        keys = [key for key in keys if "depth" not in key]
        return {key: inputs[key].to(self.device) for key in keys}

    def test(self):
        self.epoch = 0
        self.step = 0
        self.start_time = time.time()

        self.validate()

    def train(self):
        """Run the entire training pipeline"""
        self.epoch = 0
        self.mini_step = 0
        self.step = 0
        self.start_time = time.time()
        self.optimizer.zero_grad()

        self.validate()
        for self.epoch in range(1, self.args.num_epochs+1):
            self.train_one_epoch()
            self.validate()
            if self.args.enable_save and self.epoch % self.args.save_frequency == 0:
                self.save_model()

        for a in self._run.logged_artifacts():
            if a.type in ("model", "optimizer") and "latest" not in a.aliases:
                a.delete()

    def train_one_epoch(self):
        """Run a single epoch of training"""
        self.model.train()

        self.evaluator.reset_eval_metrics()

        pbar = tqdm.tqdm(self.loader_train)
        pbar.set_description(f"## {self.args.exp_name} ## Training Epoch_{self.epoch}")
        for batch_idx, inputs in enumerate(pbar, start=1):
            self.mini_step += 1

            inputs = self.inputs_to_device(inputs)
            outputs, losses = self.process_batch(inputs)

            # losses["loss"].backward()
            losses["loss"].div(self.args.accum_grads).backward()
            if self.mini_step % self.args.accum_grads == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.step += 1

            # Track eval metrics
            with torch.no_grad():
                mask = inputs["sphere_valid_mask"]
                pred_sem = outputs["pred_sem"]
                gt_sem = inputs["sphere_gt_sem"]

                self.evaluator.compute_eval_metrics(gt_sem, pred_sem, mask, track=True)

                if self.mini_step % self.args.accum_grads == 0:
                    if self.step % self.args.log_frequency == 0:
                        errors = self.evaluator.get_results(update_best=False)
                        self.log("train", inputs, outputs, losses, errors, best_errors=None)

            if batch_idx / self.args.accum_grads >= self.args.limit_train_batches:
                break

    def validate(self):
        """Validate the model on the validation set
        """
        self.model.eval()

        self.evaluator.reset_eval_metrics()

        pbar = tqdm.tqdm(self.loader_val)
        pbar.set_description(f"Validating Epoch_{self.epoch}")

        with torch.no_grad():
            for batch_idx, inputs in enumerate(pbar):
                inputs = self.inputs_to_device(inputs)

                outputs, losses = self.process_batch(inputs)

                mask = inputs["sphere_valid_mask"]
                pred_sem = outputs["pred_sem"].detach()
                gt_sem = inputs["sphere_gt_sem"]

                self.evaluator.compute_eval_metrics(gt_sem, pred_sem, mask, track=True)

        errors, best_errors = self.evaluator.get_results(update_best=True)
        self.evaluator.print()
        self.log("val", inputs, outputs, losses, errors, best_errors)
        del inputs, outputs, losses, errors

    def process_batch(self, inputs):
        x = inputs["normalized_sphere_rgb"]
        mask = inputs["sphere_valid_mask"]
        gt = inputs["sphere_gt_sem"]

        pred = self.model(x)

        loss_rec = self.compute_loss(pred.permute(0, 2, 1), gt.long())

        losses = {
            "loss": loss_rec,
        }

        outputs = {
            "pred_sem": pred.detach(),
        }

        return outputs, losses

    def log(self, mode, inputs, outputs, losses, errors, best_errors=None):
        """Write an event to the tensorboard events file
        """

        if self.wandb_run is None:
            return

        wandb.log({f"losses_{mode}/{loss_key}": loss_val
                   for loss_key, loss_val in losses.items()
                   },
                  step=self.step)

        wandb.log({f"{key.split('/')[0]}_{mode}/{key.split('/')[1]}": val
                   for key, val in errors.items()
                   },
                  step=self.step)

        if best_errors is not None:
            wandb.log({f"best_{key.split('/')[0]}_{mode}/{key.split('/')[1]}": val
                       for key, val in best_errors.items()
                       },
                      step=self.step)

    def save_model(self):
        """Save model weights to disk
        """

        if self.wandb_run:
            save_folder = os.path.join(self.wandb_run.dir, "models")
        else:
            save_folder = os.path.join(self.args.log_dir, "models")
        os.makedirs(save_folder, exist_ok=True)

        print(f"Saving model at {save_folder}")

        model_save_path = os.path.join(save_folder, "{}.pth".format("model"))
        model_state_dict = self.model.state_dict()
        torch.save(model_state_dict, model_save_path)

        opt_save_path = os.path.join(save_folder, "{}.pth".format("optimizer"))
        opt_state_dict = self.optimizer.state_dict()
        torch.save(opt_state_dict, opt_save_path)

        if self.wandb_run:
            artifact = wandb.Artifact(name=f"model-{self.wandb_run.id}", type="model", metadata=self.config)
            artifact.add_file(model_save_path)
            self.wandb_run.log_artifact(artifact)

            artifact = wandb.Artifact(name=f"optimizer-{self.wandb_run.id}", type="optimizer", metadata={})
            artifact.add_file(opt_save_path)
            self.wandb_run.log_artifact(artifact)

    def load_model(self):
        """Load model from disk
        """
        load_run = wandb.Api().run(f"{self.args.wandb_entity}/{self.args.wandb_project}/{self.args.load_weights_task}")

        artifacts = load_run.logged_artifacts()

        model_art = [art for art in artifacts if art.type == "model" and "latest" in art.aliases]
        assert len(model_art) == 1, f"Loaded weights task should have 1 latest model, got {len(model_art)}"
        opt_art = [art for art in artifacts if art.type == "optimizer" and "latest" in art.aliases]
        assert len(opt_art) <= 1, f"Loaded weights task should at most 1 latest optimizer, got {len(opt_art)}"

        model_dir = model_art[0].download(f"{self.args.log_dir}/PRETRAINED/{load_run.id}")
        if len(opt_art):
            opt_dir = opt_art[0].download(f"{self.args.log_dir}/PRETRAINED/{load_run.id}")

        print(f"Loading Model weights from {model_dir}")
        pretrained_dict = torch.load(os.path.join(model_dir, "model.pth"))
        # pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # model_dict.update(pretrained_dict)
        missing_keys, unexpected_keys = self.model.load_state_dict(pretrained_dict, strict=False)
        if len(missing_keys):
            warnings.warn(f"MISSING KEYS : {missing_keys}")
        assert len(unexpected_keys) == 0, f"{unexpected_keys}"

        if len(opt_art) and hasattr(self, "optimizer"):
            print("Loading Optimizer weights")
            optimizer_dict = torch.load(os.path.join(opt_dir, "optimizer.pth"))
            self.optimizer.load_state_dict({k: v for k, v in optimizer_dict.items() if k not in missing_keys})
        else:
            print("Optimizer weights were not saved")