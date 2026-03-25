from src.infrastructure.layers import get_total_and_remaining_params

def get_model_density(model) -> float:
    total, remaining = get_total_and_remaining_params(model)
    return round(remaining / total * 100, 4)

