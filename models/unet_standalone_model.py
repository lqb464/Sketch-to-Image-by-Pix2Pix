import torch
import torch.nn as nn

from .base_model import BaseModel
from . import networks


class UnetStandaloneModel(BaseModel):
    """Standalone U-Net generator trained with pixel reconstruction loss.

    This model intentionally has no discriminator. It keeps the same public
    tensors as pix2pix (`real_A`, `fake_B`, `real_B`) so the existing
    visualizer and W&B test-set evaluator can log images and metrics without
    model-specific branches.
    """

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.set_defaults(dataset_mode="aligned", netG="unet_256", norm="batch")
        parser.add_argument(
            "--loss",
            type=str,
            default="l1",
            choices=("l1", "l2"),
            help="pixel reconstruction loss for standalone U-Net",
        )
        parser.add_argument(
            "--lambda_L1",
            type=float,
            default=100.0,
            help="weight for the pixel reconstruction loss",
        )
        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        self.loss_mode = opt.loss.lower()
        self.loss_names = ["G_L1"] if self.loss_mode == "l1" else ["G_L2"]
        self.visual_names = ["real_A", "fake_B", "real_B"]
        self.model_names = ["G"]

        self.netG = networks.define_G(
            opt.input_nc,
            opt.output_nc,
            opt.ngf,
            opt.netG,
            norm=opt.norm,
            use_dropout=not opt.no_dropout,
            freeze_encoder=opt.freeze_encoder,
            init_type=opt.init_type,
            init_gain=opt.init_gain,
        )

        if self.isTrain:
            if self.loss_mode == "l1":
                self.criterion_pixel = nn.L1Loss()
            else:
                self.criterion_pixel = nn.MSELoss()
            self.optimizer_G = torch.optim.Adam(
                self.netG.parameters(),
                lr=opt.lr,
                betas=(opt.beta1, 0.999),
            )
            self.optimizers.append(self.optimizer_G)

    def set_input(self, input):
        AtoB = self.opt.direction == "AtoB"
        self.real_A = input["A" if AtoB else "B"].to(self.device)
        self.real_B = input["B" if AtoB else "A"].to(self.device)
        self.image_paths = input["A_paths" if AtoB else "B_paths"]

    def forward(self):
        self.fake_B = self.netG(self.real_A)

    def backward_G(self):
        pixel_loss = self.criterion_pixel(self.fake_B, self.real_B) * self.opt.lambda_L1
        if self.loss_mode == "l1":
            self.loss_G_L1 = pixel_loss
        else:
            self.loss_G_L2 = pixel_loss
        self.loss_G = pixel_loss
        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()
