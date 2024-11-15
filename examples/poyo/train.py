import logging

import hydra
import lightning as L
import torch
import torch.nn as nn
from torch_optimizer import Lamb
from lightning.pytorch.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    ModelSummary,
)
from omegaconf import DictConfig, OmegaConf

from torch_brain.registry import MODALITIY_REGISTRY, ModalitySpec
from torch_brain.models.poyo import POYOTokenizer, poyo_mp
from torch_brain.utils import callbacks as tbrain_callbacks
from torch_brain.utils import seed_everything
from torch_brain.utils.stitcher import MultiSessionDecodingStitchEvaluator
from datamodule import DataModule

# higher speed on machines with tensor cores
torch.set_float32_matmul_precision("medium")


class POYOTrainWrapper(L.LightningModule):
    def __init__(
        self,
        cfg: DictConfig,
        model: nn.Module,
        modality_spec: ModalitySpec,
    ):
        super().__init__()

        self.cfg = cfg
        self.model = model
        self.modality_spec = modality_spec
        self.save_hyperparameters(OmegaConf.to_container(cfg))

    def configure_optimizers(self):
        max_lr = self.cfg.optim.base_lr * self.cfg.batch_size  # linear scaling rule

        optimizer = Lamb(
            self.model.parameters(),
            lr=max_lr,
            weight_decay=self.cfg.optim.weight_decay,
        )

        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=max_lr,
            total_steps=self.trainer.estimated_stepping_batches,
            pct_start=self.cfg.optim.lr_decay_start,
            anneal_strategy="cos",
            div_factor=1,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }

    def training_step(self, batch, batch_idx):
        target_values = batch.pop("target_values")
        target_weights = batch.pop("target_weights")
        output_mask = batch.pop("output_mask")

        # forward pass
        output_values = self.model(**batch)

        # compute loss
        output_values = output_values[output_mask]
        target_values = target_values[output_mask]
        target_weights = target_weights[output_mask]

        if self.modality_spec.loss_fn == "mse":
            loss = torch.nn.functional.mse_loss(
                output_values, target_values, reduction="none"
            )
            loss = loss * target_weights[:, None]
            loss = loss.mean()
        else:
            raise NotImplementedError("Only MSE loss is supported for now.")

        self.log("train_loss", loss, prog_bar=True)

        # Log batch statistics
        # for name in target_values.keys():
        #     preds = torch.cat([pred[name] for pred in output if name in pred])
        #     self.log(f"predictions/mean_{name}", preds.mean())
        #     self.log(f"predictions/std_{name}", preds.std())

        #     targets = target_values[name].float()
        #     self.log(f"targets/mean_{name}", targets.mean())
        #     self.log(f"targets/std_{name}", targets.std())

        unit_index = batch["input_unit_index"].float()
        self.log("inputs/mean_unit_index", unit_index.mean())
        self.log("inputs/std_unit_index", unit_index.std())

        return loss

    def validation_step(self, batch, batch_idx):
        target_values = batch.pop("target_values")
        batch.pop("target_weights")
        absolute_starts = batch.pop("absolute_start")
        session_ids = batch.pop("session_id")
        output_subtask_index = batch.pop("output_subtask_index")
        output_mask = batch.pop("output_mask")

        # forward pass
        output_values = self.model(**batch)

        # add removed elements back to batch
        batch["target_values"] = target_values
        batch["absolute_start"] = absolute_starts
        batch["session_id"] = session_ids
        batch["output_subtask_index"] = output_subtask_index
        batch["output_mask"] = output_mask

        return output_values

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)


@hydra.main(version_base="1.3", config_path="./configs", config_name="train.yaml")
def main(cfg: DictConfig):
    # fix random seed, skipped if cfg.seed is None
    seed_everything(cfg.seed)

    # setup loggers
    log = logging.getLogger(__name__)
    wandb_logger = None
    if cfg.wandb.enable:
        wandb_logger = L.pytorch.loggers.WandbLogger(
            save_dir=cfg.log_dir,
            entity=cfg.wandb.entity,
            name=cfg.wandb.run_name,
            project=cfg.wandb.project,
            log_model=cfg.wandb.log_model,
        )

    # get modality details
    modality_spec = MODALITIY_REGISTRY[cfg.modality_name]

    # make model and tokenizer
    model = poyo_mp(dim_out=modality_spec.dim)

    tokenizer = POYOTokenizer(
        unit_tokenizer=model.unit_emb.tokenizer,
        session_tokenizer=model.session_emb.tokenizer,
        latent_step=cfg.latent_step,
        num_latents_per_step=cfg.model.num_latents,
        modality_spec=modality_spec,
        sequence_length=cfg.sequence_length,
    )

    # setup data module
    data_module = DataModule(cfg=cfg, tokenizer=tokenizer)
    data_module.setup()

    # register units and sessions
    model.unit_emb.initialize_vocab(data_module.get_unit_ids())
    model.session_emb.initialize_vocab(data_module.get_session_ids())

    # Lightning train wrapper
    wrapper = POYOTrainWrapper(
        cfg=cfg,
        model=model,
        modality_spec=modality_spec,
    )

    metric_factor = lambda: hydra.utils.instantiate(cfg.metric)
    stitch_evaluator = MultiSessionDecodingStitchEvaluator(
        session_ids=data_module.get_session_ids(),
        metric_factory=metric_factor,
    )

    callbacks = [
        stitch_evaluator,
        ModelSummary(max_depth=2),  # Displays the number of parameters in the model.
        ModelCheckpoint(
            save_last=True,
            save_on_train_epoch_end=True,
            every_n_epochs=cfg.eval_epochs,
        ),
        LearningRateMonitor(logging_interval="step"),
        tbrain_callbacks.MemInfo(),
        tbrain_callbacks.EpochTimeLogger(),
        tbrain_callbacks.ModelWeightStatsLogger(),
    ]

    trainer = L.Trainer(
        logger=wandb_logger,
        default_root_dir=cfg.log_dir,
        check_val_every_n_epoch=cfg.eval_epochs,
        max_epochs=cfg.epochs,
        log_every_n_steps=1,
        strategy=(
            "ddp_find_unused_parameters_true" if torch.cuda.is_available() else "auto"
        ),
        callbacks=callbacks,
        precision=cfg.precision,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=cfg.gpus,
        num_nodes=cfg.nodes,
        num_sanity_val_steps=-1,  # Disable sanity validation
        limit_val_batches=None,  # Ensure no limit on validation batches
    )

    # Train
    trainer.fit(wrapper, data_module, ckpt_path=cfg.ckpt_path)


if __name__ == "__main__":
    main()
