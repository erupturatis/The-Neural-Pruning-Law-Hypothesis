
def get_model_sparsity(model) -> float:
    total, remaining = model.get_parameters_pruning_statistics()
    return round(remaining / total * 100, 4)

