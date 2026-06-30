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
        hint_tensor = images.get("control", batch["hint"])[0]

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

            # CHW -> HWC
            img = img.permute(1, 2, 0).numpy()

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
# Main Training Pipeline
# ==========================================
def main():
    parser = argparse.ArgumentParser(description="ControlNet OCT Training")
    parser.add_argument('--checkpoint', type=str, default='./ControlNet/models/control_sd15_ini.ckpt'),
    parser.add_argument('--train_from_scratch', action='store_true', 
                        help='Flag to indicate training the ControlNet branch entirely from scratch.')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--image_size', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=100)
    args = parser.parse_args()

    # Define Data Directories (From NVMe staging /tmp)
    local_data_dir = os.environ.get("LOCAL_DATA_DIR", "./NR206")

    train_real = os.path.join(local_data_dir, "train")
    train_labels = os.path.join(local_data_dir, "train_labels")

    val_real = os.path.join(local_data_dir, "test")
    val_labels = os.path.join(local_data_dir, "test_labels")

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
    experiment_name = "Exp14_ControlNet_OCT"
    mode_str = "Scratch" if args.train_from_scratch else "Pretrained"
    run_name = f"ControlNet_{mode_str}_OCT_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')}"
    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment(experiment_name)
    
    with mlflow.start_run(run_name=run_name):
        mlf_logger = MLFlowLogger(
            experiment_name=experiment_name,
            tracking_uri=mlflow_uri,
            run_id=mlflow.active_run().info.run_id
        )

        # Initialize Base Model Architecture
        model = create_model('./ControlNet/models/cldm_v15.yaml').cpu()
        
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
            every_n_epochs=20,
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
        trainer.fit(model, train_loader)
        
        # Log only the final trained model to MLflow
        final_ckpt_path = "./ControlNet/checkpoints/controlnet_final.ckpt"
        print(f"Saving final model to {final_ckpt_path}...")
        trainer.save_checkpoint(final_ckpt_path)
        print("Uploading final model to MLflow (this may take a few minutes)...")
        mlflow.log_artifact(final_ckpt_path, artifact_path="checkpoints")
        print("Upload complete!")

if __name__ == '__main__':
    main()
