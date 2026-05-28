import numpy as np
import sys
import ntpath
import time
from . import util, html
from pathlib import Path
import wandb
import os
import torch.distributed as dist


def save_images(webpage, visuals, image_path, aspect_ratio=1.0, width=256):
    image_dir = webpage.get_image_dir()
    name = Path(image_path[0]).stem

    webpage.add_header(name)
    ims, txts, links = [], [], []
    for label, im_data in visuals.items():
        im = util.tensor2im(im_data)
        image_name = f"{name}_{label}.png"
        save_path = image_dir / image_name
        util.save_image(im, save_path, aspect_ratio=aspect_ratio)
        ims.append(image_name)
        txts.append(label)
        links.append(image_name)
    webpage.add_images(ims, txts, links, width=width)


class Visualizer:
    def __init__(self, opt):
        self.opt = opt
        self.use_html = opt.isTrain and not opt.no_html
        self.win_size = opt.display_winsize
        self.name = opt.name
        self.saved = False
        self.use_wandb = opt.use_wandb
        self.current_epoch = 0

        if self.use_wandb:
            if not dist.is_initialized() or dist.get_rank() == 0:
                self.wandb_entity = getattr(opt, "wandb_entity", "Sketch2Image")
                wandb_project_alias = getattr(opt, "wandb_project", "")
                self.wandb_project_name = wandb_project_alias if wandb_project_alias else getattr(opt, "wandb_project_name", "pix2pix-sketch2image")

                run_id_file = Path(opt.checkpoints_dir) / opt.name / "wandb_run_id.txt"
                run_id_file.parent.mkdir(parents=True, exist_ok=True)

                if run_id_file.exists():
                    run_id = run_id_file.read_text().strip()
                else:
                    run_id = wandb.util.generate_id()
                    run_id_file.write_text(run_id)

                self.wandb_run = wandb.init(
                    entity=self.wandb_entity,
                    project=self.wandb_project_name,
                    name=opt.name,
                    id=run_id,
                    resume='allow',
                    config=opt
                ) if not wandb.run else wandb.run
                self.wandb_run._label(repo="pix2pix-sketch2image")
            else:
                self.wandb_run = None

        if self.use_html:
            self.web_dir = Path(opt.checkpoints_dir) / opt.name / "web"
            self.img_dir = self.web_dir / "images"
            print(f"create web directory {self.web_dir}...")
            util.mkdirs([self.web_dir, self.img_dir])

        self.log_name = Path(opt.checkpoints_dir) / opt.name / "loss_log.txt"
        with open(self.log_name, "a") as log_file:
            now = time.strftime("%c")
            log_file.write(f"================ Training Loss ({now}) ================\n")

    def reset(self):
        self.saved = False

    def set_dataset_size(self, dataset_size):
        self.dataset_size = dataset_size

    def _calculate_global_step(self, epoch, epoch_iter):
        return (epoch - 1) * self.dataset_size + epoch_iter

    def display_current_results(self, visuals, epoch: int, total_iters: int, save_result=False, prefix="results", save_html=True):
        if "LOCAL_RANK" in os.environ and dist.is_initialized() and dist.get_rank() != 0:
            return

        if self.use_wandb:
            ims_dict = {}
            for label, image in visuals.items():
                image_numpy = util.tensor2im(image)
                wandb_image = wandb.Image(image_numpy, caption=f"{label} - Step {total_iters}")
                ims_dict[f"{prefix}/{label}"] = wandb_image
            self.wandb_run.log(ims_dict, step=total_iters)

        if save_html and self.use_html and (save_result or not self.saved):
            self.saved = True
            for label, image in visuals.items():
                image_numpy = util.tensor2im(image)
                img_path = self.img_dir / f"epoch{epoch:03d}_{label}.png"
                util.save_image(image_numpy, img_path)

            webpage = html.HTML(self.web_dir, f"Experiment name = {self.name}", refresh=1)
            for n in range(epoch, 0, -1):
                webpage.add_header(f"epoch [{n}]")
                ims, txts, links = [], [], []
                for label, image in visuals.items():
                    img_path = f"epoch{n:03d}_{label}.png"
                    ims.append(img_path)
                    txts.append(label)
                    links.append(img_path)
                webpage.add_images(ims, txts, links, width=self.win_size)
            webpage.save()

    def plot_current_losses(self, total_iters, losses):
        if dist.is_initialized() and dist.get_rank() != 0:
            return

        if self.use_wandb:
            self.wandb_run.log(losses, step=total_iters)

    def print_current_losses(self, epoch, iters, losses, t_comp, t_data):
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        message = f"[Rank {local_rank}] (epoch: {epoch}, iters: {iters}, time: {t_comp:.3f}, data: {t_data:.3f}) "
        for k, v in losses.items():
            message += f", {k}: {v:.3f}"
        message += "\n"
        print(message)

        if local_rank == 0:
            with open(self.log_name, "a") as log_file:
                log_file.write(f"{message}\n")
