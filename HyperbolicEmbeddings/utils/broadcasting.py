import torch


def _mul_broadcast_shape(*shapes, error_msg=None):
    """
    Compute dimension suggested by multiple tensor indices (supports broadcasting)
    This function was copied from the previous version of gpytorch.utils.broadcasting.py
    """

    # Pad each shape so they have the same number of dimensions
    num_dims = max(len(shape) for shape in shapes)
    shapes = tuple([1] * (num_dims - len(shape)) + list(shape) for shape in shapes)

    # Make sure that each dimension agrees in size
    final_size = []
    for size_by_dim in zip(*shapes):
        non_singleton_sizes = tuple(size for size in size_by_dim if size != 1)
        if len(non_singleton_sizes):
            if any(size != non_singleton_sizes[0] for size in non_singleton_sizes):
                if error_msg is None:
                    raise RuntimeError("Shapes are not broadcastable for mul operation")
                else:
                    raise RuntimeError(error_msg)
            final_size.append(non_singleton_sizes[0])
        # In this case - all dimensions are singleton sizes
        else:
            final_size.append(1)

    return torch.Size(final_size)

