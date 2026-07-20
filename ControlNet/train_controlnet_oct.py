import os
import argparse
import datetime
import torch
import numpy as np
import matplotlib.pyplot as plt
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import MLFlowLogger
import mlflow

from oct_controlnet_dataset import OCTControlNetDataset
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict
from cldm.cldm import ControlLDM
import torch.nn.functional as F

# -------------------------------------------------------------------
# Global Seeding for Reproducibility
# -------------------------------------------------------------------
import random
import numpy as np
import torch
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
try:
    import pytorch_lightning as pl
    pl.seed_everything(42, workers=True)
except ImportError:
    pass
# -------------------------------------------------------------------

torch.set_float32_matmul_precision('high')

class MLflowValidationLogger(pl.Callback):
    def __init__(self, val_batches, every_n_epochs=5, max_images=3):
        """
        Args:
            val_batches: List of cached validation batches.
            every_n_epochs: Validation logging frequency.
            max_images: Number of cached samples to log.
        """
        super().__init__()
        self.val_batches = val_batches[:max_images]
        self.every_n_epochs = every_n_epochs

    @torch.no_grad()
    def on_train_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch + 1

        # Log on epoch 1 and then every N epochs
        if epoch != 1 and epoch % self.every_n_epochs != 0:
            return

        print(f"\nLogging validation grids for epoch {epoch}...")

        was_training = pl_module.training
        pl_module.eval()

        for sample_idx, batch in enumerate(self.val_batches):

            # Move tensors to GPU
            batch = {
                k: v.to(pl_module.device) if torch.is_tensor(v) else v
                for k, v in batch.items()
            }

            self.log_validation_grid(
                trainer,
                pl_module,
                batch,
                epoch,
                sample_idx
            )

        if was_training:
            pl_module.train()

    @torch.no_grad()
    def log_validation_grid(
        self,
        trainer,
        pl_module,
        batch,
        epoch,
        sample_idx
    ):
        """
        Creates and logs a Prior | Generated | Ground Truth image grid.
        """

        images = pl_module.log_images(
            batch,
            split="val",
            N=1
        )

        # Prior (hint)
        hint_tensor = batch["hint"][0]

        # Ground truth
        gt_tensor = images.get("reconstruction", batch["jpg"])[0]

        # Generated sample
        sample_keys = [k for k in images.keys() if "samples" in k]

        if len(sample_keys) > 0:
            gen_tensor = images[sample_keys[0]][0]
        else:
            gen_tensor = gt_tensor

        def tensor_to_numpy(img):
            img = img.detach().cpu()
            if img.ndim == 3:
                if img.shape[0] in [1, 3]:
                    img = img.permute(1, 2, 0)
            img = img.numpy()
            # [-1,1] -> [0,1]
            img = np.clip((img + 1.0) / 2.0, 0, 1)
            # Convert grayscale
            if img.shape[-1] == 1:
                img = img.squeeze(-1)
            return img

        hint = tensor_to_numpy(hint_tensor)
        gen = tensor_to_numpy(gen_tensor)
        gt = tensor_to_numpy(gt_tensor)

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        titles = [
            "Prior (Synthetic Hint)",
            "Generated Synthesis",
            "Ground Truth (Real)"
        ]

        imgs = [hint, gen, gt]

        for ax, img, title in zip(axes, imgs, titles):
            ax.imshow(img, cmap="gray")
            ax.set_title(title, fontsize=14, fontweight="bold")
            ax.axis("off")

        plt.tight_layout()

        # mlflow.log_figure(
        #     fig,
        #     f"validation_grids/epoch_{epoch}_sample_{sample_idx}.png"
        # )
        artifact_path = f"validation_grids/epoch_{epoch}_sample_{sample_idx}.png"

        trainer.logger.experiment.log_figure(
            run_id=trainer.logger.run_id,
            figure=fig,
            artifact_file=artifact_path,
        )

        plt.close(fig)

# ==========================================
# Spatial Weighted ControlLDM Subclass
# ==========================================
def default(val, d):
    if val is not None:
        return val
    return d() if callable(d) else d

class SpatialWeightedControlLDM(ControlLDM):
    def training_step(self, batch, batch_idx):
        self._current_bg_mask = batch.get("bg_mask")
        return super().training_step(batch, batch_idx)

    def validation_step(self, batch, batch_idx):
        self._current_bg_mask = batch.get("bg_mask")
        return super().validation_step(batch, batch_idx)

    def p_losses(self, x_start, cond, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        model_output = self.apply_model(x_noisy, t, cond)

        loss_dict = {}
        prefix = 'train' if self.training else 'val'

        if self.parameterization == "x0":
            target = x_start
        elif self.parameterization == "eps":
            target = noise
        elif self.parameterization == "v":
            target = self.get_v(x_start, noise, t)
        else:
            raise NotImplementedError()

        # Calculate elementwise loss (mean=False returns reduction='none' for l2 or abs for l1)
        loss_elementwise = self.get_loss(model_output, target, mean=False) # shape [B, C, H, W]

        # Apply spatial loss weighting if mask is available
        bg_mask = getattr(self, "_current_bg_mask", None)
        if bg_mask is not None:
            if bg_mask.ndim == 3:
                bg_mask = bg_mask.unsqueeze(1) # shape [B, 1, H, W]
            
            latent_h, latent_w = loss_elementwise.shape[2], loss_elementwise.shape[3]
            # Downsample using bilinear interpolation
            bg_mask_down = F.interpolate(bg_mask, size=(latent_h, latent_w), mode='bilinear', align_corners=False)
            
            # w_bg = 0.4 on background (where mask=1.0), 1.0 on layers (where mask=0.0)
            # weight = mask_down * 0.4 + (1 - mask_down) * 1.0 = 1.0 - 0.6 * mask_down
            weight = 1.0 - 0.6 * bg_mask_down
            loss_elementwise = loss_elementwise * weight

        loss_simple = loss_elementwise.mean([1, 2, 3])
        loss_dict.update({f'{prefix}/loss_simple': loss_simple.mean()})

        logvar_t = self.logvar[t].to(self.device)
        loss = loss_simple / torch.exp(logvar_t) + logvar_t
        if self.learn_logvar:
            loss_dict.update({f'{prefix}/loss_gamma': loss.mean()})
            loss_dict.update({'logvar': self.logvar.data.mean()})

        loss = self.l_simple_weight * loss.mean()

        # Weight VLB loss as well
        loss_vlb = loss_elementwise.mean(dim=(1, 2, 3))
        loss_vlb = (self.lvlb_weights[t] * loss_vlb).mean()
        loss_dict.update({f'{prefix}/loss_vlb': loss_vlb})
        loss += (self.original_elbo_weight * loss_vlb)
        loss_dict.update({f'{prefix}/loss': loss})

        return loss, loss_dict

# ==========================================
# Main Training Pipeline
# ==========================================
def main():
    parser = argparse.ArgumentParser(description="ControlNet OCT Training")
    parser.add_argument('--checkpoint', type=str, default='./ControlNet/models/control_sd15_ini.ckpt'),
    parser.add_argument('--train_from_scratch', action='store_true', 
                        help='Flag to indicate training the ControlNet branch entirely from scratch.')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--image_size', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=300)
    args = parser.parse_args()

    # Define Data Directories (From NVMe staging /tmp)
    local_data_dir = os.environ.get("LOCAL_DATA_DIR", "./NR206")

    train_real = os.environ.get("TRAIN_REAL", os.path.join(local_data_dir, "train"))
    train_labels = os.environ.get("TRAIN_LABELS", os.path.join(local_data_dir, "train_labels"))

    val_real = os.environ.get("VAL_REAL", os.path.join(local_data_dir, "test"))
    val_labels = os.environ.get("VAL_LABELS", os.path.join(local_data_dir, "test_labels"))

    train_dataset = OCTControlNetDataset(
        labels_dir=train_labels,
        real_dir=train_real,
        target_size=args.image_size,
        prompt="high-resolution retinal OCT scan, spectral domain, medical imaging"
    )

    val_dataset = OCTControlNetDataset(
        labels_dir=val_labels,
        real_dir=val_real,
        target_size=args.image_size,
        prompt="high-resolution retinal OCT scan, spectral domain, medical imaging"
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    fixed_validation_batches = []

    for i, batch in enumerate(val_loader):
        fixed_validation_batches.append(batch)
        if len(fixed_validation_batches) == 3:
            break

    # MLflow Setup
    mlflow_uri = "http://10.24.38.15:5000"
    experiment_name = "OCT_ControlNet_8BitNorm"
    mode_str = "Scratch" if args.train_from_scratch else "Pretrained"
    run_name = f"ControlNet_{mode_str}_OCT_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')}"
    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment(experiment_name)
    
    with mlflow.start_run(run_name=run_name):
        # Set details tags & run description note
        mlflow.set_tags({
            "model_type": "ControlNet",
            "initialization": mode_str,
            "epochs": str(args.epochs),
            "batch_size": str(args.batch_size),
            "image_size": str(args.image_size),
            "normalization": "8-bit (-1 to 1)",
            "loss_weighting": "Spatial loss weighting (w_bg=0.4, w_layer=1.0)"
        })
        
        mlflow.set_tag("mlflow.note.content", f"""
# OCT ControlNet Retraining with 8-Bit Normalization and Spatial Loss Weighting
- **Experiment:** OCT_ControlNet_8BitNorm
- **Date/Time:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **Base Checkpoint:** {args.checkpoint}
- **Training Type:** {mode_str}
- **Hyperparameters:**
  - Batch Size: {args.batch_size}
  - Image Size: {args.image_size}
  - Epochs: {args.epochs}
  - Learning Rate: 1e-5
- **Key Enhancements:**
  1. 8-Bit Normalization on both Hint (Prior) and Target (Real OCT) to range [-1.0, 1.0] (norm = image / 127.5 - 1.0).
  2. Spatial loss weighting: w_bg = 0.4 on background regions, 1.0 on layers. The background mask is obtained where mask B=G=R=0, and downsampled to 32x32 to match the latent space dimensions before weighting the simple latent diffusion loss.
""")

        mlf_logger = MLFlowLogger(
            experiment_name=experiment_name,
            tracking_uri=mlflow_uri,
            run_id=mlflow.active_run().info.run_id
        )

        # Initialize Base Model Architecture
        model = create_model('./ControlNet/models/cldm_v15.yaml').cpu()
        model.__class__ = SpatialWeightedControlLDM
        
        if args.train_from_scratch:
            print("Initializing ControlNet from scratch. Loading base SD1.5 initial weights...")
            # Fallback to the default initialization checkpoint containing base SD1.5 weights + zeroed ControlNet
            base_ini_path = './ControlNet/models/control_sd15_ini.ckpt'
            model.load_state_dict(load_state_dict(base_ini_path, location='cpu'), strict=False)
        else:
            print(f"Resuming/Finetuning from specified checkpoint: {args.checkpoint}")
            # Load the specific user-provided checkpoint (e.g., a partially trained OCT model)
            model.load_state_dict(load_state_dict(args.checkpoint, location='cpu'), strict=False)
        model.learning_rate = 1e-5
        model.sd_locked = True
        model.only_mid_control = False

        # Log hyperparameters to MLflow
        mlf_logger.log_hyperparams(vars(args))

        # Configure Callbacks
        logger_freq = 300

        local_image_logger = ImageLogger(
            batch_frequency=logger_freq
        )

        mlflow_val_logger = MLflowValidationLogger(
            val_batches=fixed_validation_batches,
            every_n_epochs=5,
            max_images=3
        )

        trainer = pl.Trainer(
            accelerator="gpu",
            devices=1,
            strategy="auto",
            precision=32,
            max_epochs=args.epochs,
            logger=mlf_logger,
            callbacks=[
                local_image_logger,
                mlflow_val_logger,
                pl.callbacks.ModelCheckpoint(
                    dirpath='./ControlNet/checkpoints',
                    every_n_epochs=20, 
                    save_top_k=-1, 
                    filename='controlnet-{epoch:02d}'
                )
            ],
            log_every_n_steps=5
        )

        print(f"Starting {mode_str} ControlNet training.")
        trainer.fit(model, train_loader, val_loader)
        
        # Log only the final trained model to MLflow
        final_ckpt_path = "./ControlNet/checkpoints/controlnet_final.ckpt"
        print(f"Saving final model to {final_ckpt_path}...")
        trainer.save_checkpoint(final_ckpt_path)
        print("Uploading final model to MLflow (this may take a few minutes)...")
        mlflow.log_artifact(final_ckpt_path, artifact_path="checkpoints")
        print("Upload complete!")
        
        # Register the model at the end of the run
        print("Registering model in MLflow Model Registry...")
        try:
            mlflow.pytorch.log_model(
                pytorch_model=model,
                artifact_path="model",
                serialization_format="pickle"
            )
            run_id = mlflow.active_run().info.run_id
            mlflow.register_model(
                model_uri=f"runs:/{run_id}/model",
                name="ControlNet_8BitNorm_Model"
            )
            print("Successfully registered model as 'ControlNet_8BitNorm_Model'")
        except Exception as e:
            print(f"Failed to register model: {e}")

if __name__ == '__main__':
    main()
