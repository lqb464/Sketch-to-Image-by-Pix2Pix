from copy import deepcopy
from typing import Dict

import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader

from data import find_dataset_using_name


def _is_main_process() -> bool:
    return not dist.is_initialized() or dist.get_rank() == 0


class TestSetEvaluator:
    """Evaluate pix2pix outputs on a held-out split and log to W&B."""

    def __init__(self, opt, visualizer):
        self.opt = opt
        self.visualizer = visualizer
        self.device = opt.device
        self.enabled = bool(getattr(opt, "eval_test_freq", 0) > 0)

        self.lpips_metric = None
        self.psnr_metric = None
        self.ssim_metric = None
        self.msssim_metric = None
        self.fid_metric = None
        self.kid_metric = None
        self.face_embedder = None

        self._warned = set()
        self._test_iterator = None
        self.test_dataloader = None

        if not self.enabled:
            return

        self.test_dataloader = self._build_test_dataloader()
        self._setup_metrics()

    def _warn_once(self, key: str, message: str):
        if key in self._warned:
            return
        self._warned.add(key)
        if _is_main_process():
            print(message)

    def _build_test_dataloader(self):
        eval_opt = deepcopy(self.opt)
        eval_opt.phase = getattr(self.opt, "eval_phase", "test")
        eval_opt.batch_size = int(getattr(self.opt, "eval_batch_size", 1))
        eval_opt.num_threads = int(getattr(self.opt, "eval_num_threads", 0))
        eval_opt.serial_batches = True
        eval_opt.no_flip = not bool(getattr(self.opt, "eval_flip", False))
        eval_opt.max_dataset_size = getattr(self.opt, "eval_max_dataset_size", self.opt.max_dataset_size)

        dataset_class = find_dataset_using_name(eval_opt.dataset_mode)
        dataset = dataset_class(eval_opt)
        dataloader = DataLoader(
            dataset,
            batch_size=eval_opt.batch_size,
            shuffle=False,
            num_workers=eval_opt.num_threads,
            drop_last=False,
        )
        if _is_main_process():
            print(f"[Eval] dataset split='{eval_opt.phase}', images={len(dataset)}")
        return dataloader

    def _setup_metrics(self):
        try:
            from torchmetrics.image import PeakSignalNoiseRatio
            from torchmetrics.image import StructuralSimilarityIndexMeasure
            from torchmetrics.image import MultiScaleStructuralSimilarityIndexMeasure

            self.psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(self.device)
            self.ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(self.device)
            self.msssim_metric = MultiScaleStructuralSimilarityIndexMeasure(data_range=1.0).to(self.device)
        except Exception as exc:
            self._warn_once("torchmetrics_basic", f"[Eval] Unable to initialize PSNR/SSIM/MS-SSIM metrics: {exc}")

        try:
            from torchmetrics.image import LearnedPerceptualImagePatchSimilarity

            self.lpips_metric = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=False).to(self.device)
        except Exception as exc:
            self._warn_once("torchmetrics_lpips", f"[Eval] Unable to initialize LPIPS metric: {exc}")

        try:
            from torchmetrics.image.fid import FrechetInceptionDistance
            from torchmetrics.image.kid import KernelInceptionDistance

            kid_subsets = int(getattr(self.opt, "eval_kid_subsets", 10))
            kid_subset_size = int(getattr(self.opt, "eval_kid_subset_size", 50))
            self.fid_metric = FrechetInceptionDistance(normalize=False).to(self.device)
            self.kid_metric = KernelInceptionDistance(
                subsets=max(1, kid_subsets),
                subset_size=max(2, kid_subset_size),
                normalize=False,
            ).to(self.device)
        except Exception as exc:
            self._warn_once("torchmetrics_fid_kid", f"[Eval] Unable to initialize FID/KID metrics: {exc}")

        try:
            from facenet_pytorch import InceptionResnetV1

            self.face_embedder = InceptionResnetV1(pretrained="vggface2").eval().to(self.device)
        except Exception as exc:
            self._warn_once("face_embedder", f"[Eval] Unable to initialize face identity metric: {exc}")

    @staticmethod
    def _to_01(x: torch.Tensor) -> torch.Tensor:
        return ((x + 1.0) * 0.5).clamp(0.0, 1.0)

    @staticmethod
    def _to_uint8(x_01: torch.Tensor) -> torch.Tensor:
        return (x_01 * 255.0).round().to(torch.uint8)

    def _next_test_batch(self):
        if self._test_iterator is None:
            self._test_iterator = iter(self.test_dataloader)
        try:
            batch = next(self._test_iterator)
        except StopIteration:
            self._test_iterator = iter(self.test_dataloader)
            batch = next(self._test_iterator)
        return batch

    def _log_test_visuals(self, model, epoch: int, total_iters: int):
        if not self.visualizer.use_wandb:
            return
        if getattr(self.opt, "no_eval_test_images", False):
            return

        test_data = self._next_test_batch()
        model.set_input(test_data)
        model.test()
        visuals = model.get_current_visuals()
        self.visualizer.display_current_results(
            visuals=visuals,
            epoch=epoch,
            total_iters=total_iters,
            save_result=False,
            prefix="test/results",
            save_html=False,
        )

    def _log_eval_metrics(self, metrics: Dict[str, float], total_iters: int):
        if not self.visualizer.use_wandb:
            return
        if not _is_main_process():
            return
        self.visualizer.wandb_run.log(metrics, step=total_iters)

    def _reset_metrics(self):
        for metric in (
            self.lpips_metric,
            self.psnr_metric,
            self.ssim_metric,
            self.msssim_metric,
            self.fid_metric,
            self.kid_metric,
        ):
            if metric is not None:
                metric.reset()

    def _compute_scalar(self, metric):
        if metric is None:
            return float("nan")
        value = metric.compute()
        if isinstance(value, tuple):
            # KID returns (mean, std)
            return value
        return float(value.detach().cpu().item())

    def _identity_cosine(self, fake: torch.Tensor, real: torch.Tensor):
        if self.face_embedder is None:
            return None
        fake_160 = F.interpolate(fake, size=(160, 160), mode="bilinear", align_corners=False)
        real_160 = F.interpolate(real, size=(160, 160), mode="bilinear", align_corners=False)
        with torch.no_grad():
            fake_emb = self.face_embedder(fake_160)
            real_emb = self.face_embedder(real_160)
        return F.cosine_similarity(fake_emb, real_emb, dim=1).mean()

    def evaluate_and_log(self, model, epoch: int, total_iters: int):
        if not self.enabled or self.test_dataloader is None:
            return

        eval_freq = int(getattr(self.opt, "eval_test_freq", 0))
        if eval_freq <= 0 or total_iters % eval_freq != 0:
            return

        model.eval()
        self._reset_metrics()
        max_test = int(getattr(self.opt, "num_test", 100))
        max_fid = int(getattr(self.opt, "eval_num_fid_images", max_test))
        fid_seen = 0
        sample_count = 0
        mae_sum = torch.zeros(1, device=self.device)
        id_cos_sum = torch.zeros(1, device=self.device)
        id_cos_count = torch.zeros(1, device=self.device)

        with torch.no_grad():
            for i, data in enumerate(self.test_dataloader):
                if i * self.test_dataloader.batch_size >= max_test:
                    break

                model.set_input(data)
                model.test()

                fake = model.fake_B
                real = model.real_B
                bsz = fake.size(0)
                sample_count += bsz

                fake_01 = self._to_01(fake)
                real_01 = self._to_01(real)

                mae_batch = torch.abs(fake_01 - real_01).mean()
                mae_sum += mae_batch * bsz

                if self.psnr_metric is not None:
                    self.psnr_metric.update(fake_01, real_01)
                if self.ssim_metric is not None:
                    self.ssim_metric.update(fake_01, real_01)
                if self.msssim_metric is not None:
                    self.msssim_metric.update(fake_01, real_01)
                if self.lpips_metric is not None:
                    self.lpips_metric.update(fake, real)

                if fid_seen < max_fid and (self.fid_metric is not None or self.kid_metric is not None):
                    remain = max_fid - fid_seen
                    take = min(remain, bsz)
                    fake_uint8 = self._to_uint8(fake_01[:take])
                    real_uint8 = self._to_uint8(real_01[:take])
                    if self.fid_metric is not None:
                        self.fid_metric.update(real_uint8, real=True)
                        self.fid_metric.update(fake_uint8, real=False)
                    if self.kid_metric is not None:
                        self.kid_metric.update(real_uint8, real=True)
                        self.kid_metric.update(fake_uint8, real=False)
                    fid_seen += take

                id_batch = self._identity_cosine(fake, real)
                if id_batch is not None:
                    id_cos_sum += id_batch * bsz
                    id_cos_count += bsz

        if sample_count == 0:
            self._warn_once("no_test_samples", "[Eval] No test samples were evaluated. Check dataroot/test and num_test.")
            model.train()
            return

        # Reduce hand-rolled aggregates across ranks if using DDP
        mae_global = mae_sum
        sample_tensor = torch.tensor(float(sample_count), device=self.device)
        if dist.is_initialized():
            dist.all_reduce(mae_global, op=dist.ReduceOp.SUM)
            dist.all_reduce(sample_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(id_cos_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(id_cos_count, op=dist.ReduceOp.SUM)

        mae_value = float((mae_global / sample_tensor).detach().cpu().item())
        identity_value = float("nan")
        if id_cos_count.item() > 0:
            identity_value = float((id_cos_sum / id_cos_count).detach().cpu().item())

        psnr_value = self._compute_scalar(self.psnr_metric) if self.psnr_metric is not None else float("nan")
        ssim_value = self._compute_scalar(self.ssim_metric) if self.ssim_metric is not None else float("nan")
        msssim_value = self._compute_scalar(self.msssim_metric) if self.msssim_metric is not None else float("nan")
        lpips_value = self._compute_scalar(self.lpips_metric) if self.lpips_metric is not None else float("nan")
        fid_value = self._compute_scalar(self.fid_metric) if self.fid_metric is not None else float("nan")

        kid_mean = float("nan")
        kid_std = float("nan")
        if self.kid_metric is not None:
            kid_result = self.kid_metric.compute()
            if isinstance(kid_result, tuple) and len(kid_result) == 2:
                kid_mean = float(kid_result[0].detach().cpu().item())
                kid_std = float(kid_result[1].detach().cpu().item())

        metrics = {
            "test/MAE": mae_value,
            "test/PSNR": psnr_value,
            "test/SSIM": ssim_value,
            "test/MS_SSIM": msssim_value,
            "test/LPIPS": lpips_value,
            "test/FID": fid_value,
            "test/KID_mean": kid_mean,
            "test/KID_std": kid_std,
            "test/IdentityCosine": identity_value,
            "test/num_samples": float(sample_tensor.detach().cpu().item()),
            "test/num_fid_samples": float(fid_seen),
            "test/epoch": float(epoch),
        }

        self._log_eval_metrics(metrics, total_iters=total_iters)
        self._log_test_visuals(model, epoch=epoch, total_iters=total_iters)

        if _is_main_process():
            msg = ", ".join(f"{k}={v:.6f}" if isinstance(v, float) else f"{k}={v}" for k, v in metrics.items())
            print(f"[Eval @ iter {total_iters}] {msg}")

        model.train()
