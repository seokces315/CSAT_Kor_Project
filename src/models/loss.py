import torch


# Function to define Huber loss
def compute_huber_loss(preds, labels, delta):
    # Assertion
    assert preds.shape == labels.shape

    # Define the values of abs_error, delta
    delta = torch.tensor(delta, dtype=preds.dtype, device=preds.device)
    abs_error = torch.abs(preds - labels)

    # Flag
    is_small_error = abs_error <= delta

    # Compute & Select loss
    small_error_loss = 0.5 * (abs_error**2)
    large_error_loss = delta * (abs_error - 0.5 * delta)
    loss = torch.where(is_small_error, small_error_loss, large_error_loss)

    return loss.mean()
