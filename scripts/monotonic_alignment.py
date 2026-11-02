"""
Monotonic alignment search with optional Numba acceleration.

The alignment algorithm finds which text token corresponds to each frame.

It works on a matrix for scores for audio frames against text tokens. It then finds the 
highest scoring path through the matrix in a monotonic fashion. 

Each frame can stay on the current token, or advance to the next token. It never moves backward.

This is needed because the dataset has no information such as timestamps which match up
text and audio frames.

The process is as follows:
1. Compute a score matrix for each batch item, where each element represents the alignment score between a frame and a token.
2. Use dynamic programming to find the highest scoring monotonic path through the matrix.
3. Backtrack from the end of the path to determine the alignment of frames to tokens.

Note that although VITS uses a stochatistic duration predictor, this alignment algorithm 
is purely deterministic based on the score matrix.

The individual scores are log-likelihoods of the alignment between frames and tokens. Therefore,
accumulated scores represent the log-likelihood of the entire alignment path.

"""

import logging

import numpy as np
import torch

logger = logging.getLogger(__name__)


def maximum_path_python(paths, values, t_ys, t_xs):
    """
    This is a pure Python implementation of the maximum path algorithm for monotonic alignment.

    Numba optionally compiles this same algorithm to accelerate computation.

    Args:
        paths (np.ndarray): Array to store the resulting alignment paths.
        values (np.ndarray): Array containing the alignment scores.
        t_ys (np.ndarray): Array containing the lengths of the frames for each batch item.
        t_xs (np.ndarray): Array containing the lengths of the tokens for each batch item.

    Returns:
        None: The function modifies the `paths` array in place and updates the `values` array with accumulated scores.
    """

    # Iterate over each batch item to compute the maximum path.
    for i in range(paths.shape[0]):
        path = paths[i]
        value = values[i]
        t_y = t_ys[i]
        t_x = t_xs[i]
        index = t_x - 1

        # Compute the accumulated scores for the dynamic programming matrix.
        for y in range(t_y):
            # Iterate over each token position within the valid range for this frame.
            for x in range(max(0, t_x + y - t_y), min(t_x, y + 1)):
                # Compute the current and previous accumulated scores for this position.
                v_cur = -1e9 if x == y else value[y - 1, x]
                # If x equals y, set the current score to a very low value to enforce monotonicity.
                if x == 0:
                    v_prev = 0.0 if y == 0 else -1e9
                # If x is 0, the previous score comes from the top cell; otherwise, it comes from the top-left cell.
                # This ensures that the path remains monotonic and only moves diagonally or vertically.
                else:
                    v_prev = value[y - 1, x - 1]
                # Update the accumulated score for this cell with the maximum of the previous and current scores.
                value[y, x] += max(v_prev, v_cur)

        # Backtrack to find the maximum path through the accumulated scores.
        for y in range(t_y - 1, -1, -1):
            path[y, index] = 1
            if index != 0 and y != 0 and (
                index == y or value[y - 1, index] < value[y - 1, index - 1]
            ):
                index -= 1




class MonotonicAlignment:
    """
    Select an alignment kernel once per instance, depending on the availability of 
    Numba for performance optimization.
    """

    def __init__(self):
        self._kernel = self._select_maximum_path()

    def _select_maximum_path(self):
        """
        Choose either numba or pure python implementation for the maximum path computation.
        Ideally numba is used for performance, but falls back to pure Python if unavailable.
        """
        try:
            from numba import njit

            kernel = njit(nogil=True)(maximum_path_python)
            # Compile and test some sample training data
            kernel(
                np.zeros((1, 2, 1), dtype=np.int32),
                np.zeros((1, 2, 1), dtype=np.float32),
                np.array([2], dtype=np.int32),
                np.array([1], dtype=np.int32),
            )
        except Exception:
            logger.warning(
                "Numba alignment startup check failed; using Python alignment.",
                exc_info=True,
            )
            return maximum_path_python
        logger.info("Using Numba alignment.")
        return kernel


    def maximum_path(self, neg_cent, mask):
        """
        Get the maximum path alignment for the input scores and mask.

        Inputs have shape [batch, frames, tokens]. The mask must describe a
        nonempty rectangular valid region at the start of each batch item.

        Args:
            neg_cent (torch.Tensor): Input scores with shape [batch, frames, tokens].
            mask (torch.Tensor): Boolean mask indicating valid regions with the same shape as neg_cent.
        """

        # perform some basic input validation
        if neg_cent.ndim != 3 or mask.shape != neg_cent.shape:
            raise ValueError("Scores and mask must have matching [batch, frames, tokens] shapes.")
        if neg_cent.shape[1] == 0 or neg_cent.shape[2] == 0:
            raise ValueError("Alignment requires nonempty frame and token dimensions.")

        # put the input tensors on CPU and convert to numpy arrays for processing
        values = neg_cent.detach().to(device="cpu", dtype=torch.float32).numpy().copy()
        paths = np.zeros(values.shape, dtype=np.int32)
        # compute the lengths of valid frames and tokens for each batch item
        t_ys = mask.sum(1)[:, 0].detach().cpu().numpy().astype(np.int32)
        t_xs = mask.sum(2)[:, 0].detach().cpu().numpy().astype(np.int32)

        # check that the computed lengths of valid frames and tokens are within acceptable bounds
        if np.any((t_xs < 1) | (t_ys < t_xs) | (t_ys > values.shape[1]) | (t_xs > values.shape[2])):
            raise ValueError("Alignment requires 1 <= tokens <= frames within the array bounds.")
        self._kernel(paths, values, t_ys, t_xs)

        return torch.from_numpy(paths).to(device=neg_cent.device, dtype=neg_cent.dtype)
