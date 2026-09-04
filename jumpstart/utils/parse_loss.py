def parse_loss(log_dict):
    """Parse the loss from a log dictionary.

    Args:
        log_dict (dict): A dictionary containing training logs.
    Returns:
        float: The parsed loss value.
    """

    if isinstance(log_dict, tuple):
        log_dict = log_dict[-1]

    # sum of all keys with "loss" in their name
    sum_loss = 0.0

    for key, value in log_dict.items():
        if "loss" in key and isinstance(value, (int, float)):
            sum_loss += value

    return sum_loss
