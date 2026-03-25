import argparse
import sys
from src.infrastructure.constants import INITIAL_LR # Assuming this is correctly defined in your project
from src.infrastructure.others import TrainingConfigsWithResume # Assuming this is correctly defined
from src.mnist_lenet300.train_pruned_lenet300_bottleneck import train_pruned_lenet300_mnist_bottleneck
from src.mnist_lenet300.train_sparsity_curves_adam import run_lenet300_mnist_adam_sparsity_curve
from src.resnet50_cifar10.train_pruned_resnet50_cifar10 import train_resnet50_cifar10_sparse_model
from src.resnet50_cifar100.train_pruned_resnet50_cifar100 import train_resnet50_cifar100_sparse_model
from src.resnet50_imagenet1k.train_pruned_resnet50_imagenet import train_resnet50_imagenet_sparse_model
from src.vgg19_cifar10.train_pruned_vgg19_cifar10 import train_vgg19_cifar10_sparse_model
from src.vgg19_cifar100.train_pruned_vgg19_cifar100 import train_vgg19_cifar100_sparse_model

# --- Experiment Functions (from your new code) ---
def lenet300_bottleneck_experiment():
    print("Running LeNet-300 Bottleneck Experiment...")
    train_pruned_lenet300_mnist_bottleneck()

def lenet300_convergence_experiments():
    print("Running LeNet-300 Convergence Experiments...")
    run_lenet300_mnist_adam_sparsity_curve(2, -2, 2) # Parameters as in your example

def resnet50_imagenet_965_sparsity_experiment():
    print("Running ResNet50 ImageNet 96.5% Sparsity Experiment...")
    defaults: TrainingConfigsWithResume = {
        "pruning_end": 80,
        "regrowing_end": 100,
        "target_sparsity": 100 - 3.5, # Adjusted from 2.5 as per typical interpretation (remaining_params = 100 - sparsity)
        "resume": "resnet50_imagenet_baseline.pth",
        "lr_flow_params_decay_regrowing": 0.55,
        "start_lr_pruning": 0.01,
        "end_lr_pruning": 0.01 / 3,
        "reset_lr_pruning": 0.01 / 10,
        "end_lr_regrowth": 0.0001,
        "reset_lr_flow_params_scaler": 3,
        "weight_decay": 1e-4,
        "notes": "Running imagenet1k aiming for ~96.5% sparsity (3.5% remaining)"
    }
    train_resnet50_imagenet_sparse_model(defaults)

def resnet50_imagenet_95_sparsity_experiment():
    print("Running ResNet50 ImageNet 95% Sparsity Experiment...")
    defaults: TrainingConfigsWithResume = {
        "pruning_end": 80,
        "regrowing_end": 100,
        "target_sparsity": 100 - 5, # Adjusted from 4 (remaining_params = 100 - sparsity)
        "resume": "resnet50_imagenet_baseline.pth",
        "lr_flow_params_decay_regrowing": 0.55,
        "start_lr_pruning": 0.01,
        "end_lr_pruning": 0.01 / 3,
        "reset_lr_pruning": 0.01 / 10,
        "end_lr_regrowth": 0.0001,
        "reset_lr_flow_params_scaler": 3,
        "weight_decay": 1e-4,
        "notes": "Running imagenet1k aiming for ~95% sparsity (5% remaining)"
    }
    train_resnet50_imagenet_sparse_model(defaults)

def resnet50_imagenet_90_sparsity_experiment():
    print("Running ResNet50 ImageNet 90% Sparsity Experiment...")
    defaults: TrainingConfigsWithResume = {
        "pruning_end": 80,
        "regrowing_end": 100,
        "target_sparsity": 100 - 10, # Adjusted from 8.5 (remaining_params = 100 - sparsity)
        "resume": "resnet50_imagenet_baseline.pth",
        "lr_flow_params_decay_regrowing": 0.55,
        "start_lr_pruning": 0.01,
        "end_lr_pruning": 0.01 / 3,
        "reset_lr_pruning": 0.01 / 10,
        "end_lr_regrowth": 0.0001,
        "reset_lr_flow_params_scaler": 3,
        "weight_decay": 1e-4,
        "notes": "Running imagenet1k aiming for ~90% sparsity (10% remaining)"
    }
    train_resnet50_imagenet_sparse_model(defaults)

def resnet50_cifar100_sparsity_experiment(final_sparsity: float):
    defaults: TrainingConfigsWithResume = {
        "pruning_end": 100,
        "regrowing_end": 160,
        "lr_flow_params_decay_regrowing": 0.75,
        "start_lr_pruning": INITIAL_LR,
        "end_lr_pruning": INITIAL_LR / 30,
        "reset_lr_pruning": INITIAL_LR / 100,
        "end_lr_regrowth": INITIAL_LR / 1000,
        "reset_lr_flow_params_scaler": 1,
        "weight_decay": 5e-4,
        "target_sparsity": final_sparsity,
        "resume": "resnet50_cifar100_accuracy78.54%", # Ensure this baseline file exists
        "notes": f"ResNet50 CIFAR-100 {final_sparsity}% final sparsity"
    }
    train_resnet50_cifar100_sparse_model(defaults)

def resnet50_cifar10_sparsity_experiment(final_sparsity: float):
    defaults: TrainingConfigsWithResume = {
        "pruning_end": 100,
        "regrowing_end": 160,
        "lr_flow_params_decay_regrowing": 0.75,
        "start_lr_pruning": INITIAL_LR,
        "end_lr_pruning": INITIAL_LR / 30,
        "reset_lr_pruning": INITIAL_LR / 100,
        "end_lr_regrowth": INITIAL_LR / 1000,
        "reset_lr_flow_params_scaler": 1,
        "weight_decay": 5e-4,
        "target_sparsity": final_sparsity,
        "resume": "resnet50_cifar10_accuracy94.64%", # Ensure this baseline file exists
        "notes": f"ResNet50 CIFAR-10 {final_sparsity}% final sparsity"
    }
    train_resnet50_cifar10_sparse_model(defaults)

def vgg19_cifar10_sparsity_experiment(final_sparsity: float):
    defaults: TrainingConfigsWithResume = {
        "pruning_end": 100,
        "regrowing_end": 160,
        "lr_flow_params_decay_regrowing": 0.75,
        "start_lr_pruning": INITIAL_LR,
        "end_lr_pruning": INITIAL_LR / 30,
        "reset_lr_pruning": INITIAL_LR / 100,
        "end_lr_regrowth": INITIAL_LR / 1000,
        "reset_lr_flow_params_scaler": 1,
        "weight_decay": 5e-4,
        "target_sparsity": final_sparsity,
        "resume": "vgg19_cifar10_accuracy93.5%", # Ensure this baseline file exists
        "notes": f"VGG19 CIFAR-10 {final_sparsity}% final sparsity"
    }
    train_vgg19_cifar10_sparse_model(defaults)

def vgg19_cifar100_sparsity_experiment(final_sparsity: float):
    defaults: TrainingConfigsWithResume = {
        "pruning_end": 100,
        "regrowing_end": 160,
        "lr_flow_params_decay_regrowing": 0.75,
        "start_lr_pruning": INITIAL_LR,
        "end_lr_pruning": INITIAL_LR / 30,
        "reset_lr_pruning": INITIAL_LR / 100,
        "end_lr_regrowth": INITIAL_LR / 1000,
        "reset_lr_flow_params_scaler": 5,
        "weight_decay": 5e-4,
        "target_sparsity": final_sparsity,
        "resume": "vgg19_cifar100_accuracy73.51%", # Ensure this baseline file exists
        "notes": f"VGG19 CIFAR-100 {final_sparsity}% final sparsity"
    }
    train_vgg19_cifar100_sparse_model(defaults)

def main():
    parser = argparse.ArgumentParser(description="Run Hyperflux pruning experiments.")
    parser.add_argument(
        "--experiment",
        type=str,
        required=True,
        choices=[
            "lenet300_bottleneck", "lenet300_convergence",
            "resnet50_imagenet_96.5", "resnet50_imagenet_95", "resnet50_imagenet_90",
            "resnet50_cifar100", "resnet50_cifar10",
            "vgg19_cifar10", "vgg19_cifar100"
        ],
        help="Name of the experiment to run."
    )
    parser.add_argument(
        "--sparsity",
        type=float,
        default=None,
        help="Target final sparsity percentage (e.g., 98 for 98%%). Required for experiments: "
             "resnet50_cifar100, resnet50_cifar10, vgg19_cifar10, vgg19_cifar100."
    )

    args = parser.parse_args()

    experiments_requiring_sparsity = [
        "resnet50_cifar100", "resnet50_cifar10",
        "vgg19_cifar10", "vgg19_cifar100"
    ]

    if args.experiment in experiments_requiring_sparsity and args.sparsity is None:
        parser.error(f"Experiment '{args.experiment}' requires the --sparsity argument.")

    if args.sparsity is not None and not (0 <= args.sparsity <= 100):
        parser.error(f"--sparsity must be between 0 and 100. Got: {args.sparsity}")

    if args.experiment == "lenet300_bottleneck":
        lenet300_bottleneck_experiment()
    elif args.experiment == "lenet300_convergence":
        lenet300_convergence_experiments()
    elif args.experiment == "resnet50_imagenet_96.5":
        resnet50_imagenet_965_sparsity_experiment()
    elif args.experiment == "resnet50_imagenet_95":
        resnet50_imagenet_95_sparsity_experiment()
    elif args.experiment == "resnet50_imagenet_90":
        resnet50_imagenet_90_sparsity_experiment()
    elif args.experiment == "resnet50_cifar100":
        resnet50_cifar100_sparsity_experiment(args.sparsity)
    elif args.experiment == "resnet50_cifar10":
        resnet50_cifar10_sparsity_experiment(args.sparsity)
    elif args.experiment == "vgg19_cifar10":
        vgg19_cifar10_sparsity_experiment(args.sparsity)
    elif args.experiment == "vgg19_cifar100":
        vgg19_cifar100_sparsity_experiment(args.sparsity)
    else:
        print(f"Unknown experiment: {args.experiment}")
        sys.exit(1)

    print(f"Experiment '{args.experiment}' finished.")

if __name__ == "__main__":
    # Example: To run the main function from your original new code, you would call these directly.
    # For CLI usage, the main() function above will handle dispatch.
    # Original main calls:
    resnet50_cifar100_sparsity_experiment(98)
    # resnet50_cifar10_sparsity_experiment(98)
    # vgg19_cifar10_sparsity_experiment(98)
    # vgg19_cifar100_sparsity_experiment(98)

    # main()