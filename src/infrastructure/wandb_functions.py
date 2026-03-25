import enum
from enum import Enum
from typing import List, Dict
import wandb
from src.infrastructure.configs_general import WANDB_REGISTER

def wandb_initalize(experiment: str, type: str, configs: Dict = None, other_tags: List[str] = [], note = '') -> None:
    if WANDB_REGISTER:
        wandb.init(
            project=experiment,
            config=configs,
            tags=[type, *other_tags],
            notes=note
        )
        wandb.define_metric("epoch")
        wandb.define_metric("*", step_metric="epoch")

def wandb_snapshot_baseline(epoch:int, accuracy: float, test_loss:float, others: Dict = None):
    if WANDB_REGISTER:
        # print("REGISERED DATA")
        wandb.log({"epoch": epoch, "test_loss": test_loss, "accuracy": accuracy, "others": others})

def wandb_snapshot(epoch: int, accuracy: float, test_loss: float, sparsity: float, others: Dict = None):
    if WANDB_REGISTER:
        wandb.log({"epoch": epoch, "test_loss": test_loss, "accuracy": accuracy, "sparsity": sparsity, "others": others})

def wandb_finish():
    if WANDB_REGISTER:
        wandb.finish()
