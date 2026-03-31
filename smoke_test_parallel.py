import multiprocessing as mp
import os
import torch
import time

# Import training functions
from src.experiments.resnet50_variable_cifar100_train_dense import train_dense_resnet50_cifar100
from src.experiments.vgg19_variable_cifar100_train_dense import train_dense_vgg19_cifar100
from src.experiments.resnet50_variable_cifar10_train_dense import train_dense_resnet50_cifar10
from src.experiments.vgg19_variable_cifar10_train_dense import train_dense_vgg19_cifar10

# Import model classes
from src.model_resnet50_cifars.model_resnet50_variable_class import ModelResnet50Variable
from src.model_vgg19_cifars.model_vgg19_variable_class import ModelVGG19Variable

from src.infrastructure.layers import ConfigsNetworkMask
# Import pre-processing functions for pre-downloading
from src.infrastructure.dataset_context.data_preprocessing import cifar10_preprocess, cifar100_preprocess

def worker(gpu_id, exp_name, train_func, model_class, num_classes):
    """
    Worker function to run a single experiment on a specific GPU.
    """
    print(f"\n>>> [GPU {gpu_id}] Starting experiment: {exp_name}")
    
    # Set environment variable to isolate the GPU for this process
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    # Ensure torch uses the correct device (it will see only one GPU as index 0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f">>> [GPU {gpu_id}] Process {os.getpid()} using device: {device}")

    # Initialize model
    cfg: ConfigsNetworkMask = {
        "mask_apply_enabled": False,
        "mask_training_enabled": False,
        "weights_training_enabled": True
    }
    model = model_class(alpha=1.0, config_network_mask=cfg, num_classes=num_classes).to(device)
    
    # Run the training
    try:
        train_func(model)
        print(f"\n>>> [GPU {gpu_id}] COMPLETED experiment: {exp_name}")
    except Exception as e:
        print(f"\n>>> [GPU {gpu_id}] FAILED experiment: {exp_name}")
        print(f"Error: {e}")

def main():
    # Use 'spawn' to avoid issues with CUDA in multiprocessing
    mp.set_start_method('spawn', force=True)
    
    print("Main: Pre-downloading datasets to avoid race conditions...")
    # Trigger downloads sequentially in the main process
    cifar10_preprocess()
    cifar100_preprocess()
    print("Main: Datasets ready.")
    
    # List of experiments to run
    experiments = [
        ("ResNet50 CIFAR-100", train_dense_resnet50_cifar100, ModelResnet50Variable, 100),
        ("VGG19 CIFAR-100", train_dense_vgg19_cifar100, ModelVGG19Variable, 100),
        ("ResNet50 CIFAR-10", train_dense_resnet50_cifar10, ModelResnet50Variable, 10),
        ("VGG19 CIFAR-10", train_dense_vgg19_cifar10, ModelVGG19Variable, 10),
    ]
    
    num_gpus = 3
    active_processes = []
    
    # Queue of available GPU IDs
    gpu_pool = [i for i in range(num_gpus)]
    
    exp_idx = 0
    while exp_idx < len(experiments) or active_processes:
        # 1. Start new processes if GPUs are available
        while gpu_pool and exp_idx < len(experiments):
            gpu_id = gpu_pool.pop(0)
            exp_info = experiments[exp_idx]
            exp_idx += 1
            
            p = mp.Process(
                target=worker, 
                args=(gpu_id, *exp_info)
            )
            p.start()
            active_processes.append((p, gpu_id))
            print(f"Main: Dispatched '{exp_info[0]}' to GPU {gpu_id}")
            
        # 2. Check for finished processes
        finished_any = False
        for i in range(len(active_processes) - 1, -1, -1):
            p, gpu_id = active_processes[i]
            if not p.is_alive():
                print(f"Main: Process on GPU {gpu_id} finished (Exit Code: {p.exitcode})")
                p.join()
                gpu_pool.append(gpu_id) # Return GPU to pool
                active_processes.pop(i)
                finished_any = True
                
        if not finished_any:
            time.sleep(10) # Wait before checking again

    print("\nAll experiments finished.")

if __name__ == "__main__":
    main()
