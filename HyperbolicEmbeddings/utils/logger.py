import numpy as np
from typing import Protocol, TypeVar, Union
import torch
import logging
import copy
import os
from pathlib import Path
from datetime import datetime
import gpytorch
from HyperbolicEmbeddings.losses.graph_based_loss import stress, StressLossTermExactMLL

class Logger(Protocol):
    def log(self, step: int, n_steps: int, loss: float, parameters: dict[str, np.ndarray], state_dict) -> None:
        ...


T = TypeVar('T')


class CompositeLogger():
    def __init__(self, loggers: T) -> None:
        self.loggers = loggers

    def __getattr__(self, attr):
        for logger in self.loggers:
            if hasattr(logger, attr):
                return getattr(logger, attr)
        raise AttributeError(f"'CompositeLogger' object contains no 'Logger' object with attribute '{attr}'")

    def log(self, step: int, n_steps: int, loss: float, X: np.ndarray, parameters: dict[str, np.ndarray] = None,
            state_dict: dict[str, torch.Tensor] = None, curvature: float = None, mll: gpytorch.mlls.ExactMarginalLogLikelihood = None) -> None:
        for logger in self.loggers:
            logger.log(step=step, n_steps=n_steps, loss=loss, X=X, parameters=parameters, state_dict=state_dict,
                       curvature=curvature, mll=mll)


def compose_loggers(*loggers: T) -> Union[CompositeLogger, T]:
    return CompositeLogger(loggers)


class ConsoleLogger():
    def __init__(self, n_logs=100) -> None:
        """
        Parameters
        n_logs: int   number of print calls to the console
        """
        self.n_logs = n_logs

    def log(self, step: int, n_steps: int, loss: float, parameters: dict[str, np.ndarray], **_) -> None:
        if n_steps < self.n_logs or step % (n_steps / self.n_logs) == 0:
            self.print_loss(step, n_steps, loss, parameters)

    def print_loss(self, step: int, n_steps: int, loss: float, parameters: dict[str, np.ndarray]):
        width = str(int(np.log10(n_steps) + 1)) if n_steps > 0 else '1'
        # cuda = 'Cuda' if base_config.cuda else ''
        curvature_key = [key for key, val in parameters.items() if 'manifold.k' in key]
        if curvature_key:
            curvature_key = curvature_key[0]
            k = parameters[curvature_key]
            logging.info(f'{step:{width}d}/{n_steps:{width}d} - Loss: {loss:.5f} - Curvature: {k:.5f}')
        else:
            logging.info(f'{step:{width}d}/{n_steps:{width}d} - Loss: {loss:.5f}')
        # logging.info(f'{step:{width}d}/{n_steps:{width}d} - Loss: {loss:.5f} - {cuda} Memory: {memory_analyser.memory()}')

    def print_full_iteration(self, title: str, step: int, loss: int, parameters: dict[str, np.ndarray]):
        logging.info(f'============== {title}: {step} - Loss: {loss:.5f} ==============')
        max_name_length = max([len(name) for name in parameters.keys()])
        for name, params in parameters.items():
            np.set_printoptions(precision=5)
            logging.info(f'{name.ljust(max_name_length)} {params}')
        logging.info('')


class MemoryLogger():
    """
    Stores the parameters and loss value for all iterations in memory
    """

    def __init__(self, path: str, only_store_best_step: bool = False) -> None:
        """
        Initialisation of the memory logger.

        Parameters
        ----------
        path : str
            string denoting the path to the directory where all the logged variables should be stored in
        only_store_best_step : Optional[bool]
            whether only data pertaining to the best step should be stored

        """
        self.only_store_best_step = only_store_best_step

        self.steps = set()
        self.losses = []
        self.stess_loss = []
        self.reconstruction_error = []
        self.curvature = []
        self.parameters = {}
        self.X_during_training = np.array([])
        self.state_dict = None

        self.best_step = None
        self.best_loss = 1e10
        self.best_X = None
        self.best_parameters = None
        self.best_curvature = None
        self.best_state_dict = None
        self.path = path

    @property
    def last_parameters(self) -> dict[str, np.ndarray]:
        return {k: v[-1] for k, v in self.parameters.items()}

    def log(self, step: int, loss: float, X: np.ndarray, parameters: dict[str, np.ndarray] = None,
            state_dict: dict[str, torch.Tensor] = None, mll: gpytorch.mlls.ExactMarginalLogLikelihood = None, **_) -> None:

        # if step in self.steps:
        #     return

        if loss < self.best_loss:
            self.best_step = step
            self.best_loss = loss
            self.best_X = X
            self.best_parameters = parameters
            self.best_state_dict = copy.deepcopy(state_dict) if state_dict is not None else None

        self.steps.add(step)
        self.losses.append(loss)
        self.state_dict = copy.deepcopy(state_dict) if state_dict is not None else None

        # Compute the stress loss to log it (Only when the added loss is stress!)
        latent_embeddings = torch.from_numpy(X)
        if isinstance(mll.model.added_loss, StressLossTermExactMLL):
            stress_loss_value = mll.model.added_loss.loss(latent_embeddings)
            self.stess_loss.append(stress_loss_value)

        # Compute the reconstruction error to log it
        reconstructed_training_data = mll.model(latent_embeddings).mean           # This T kinda sus
        reconstruction_error = np.mean(np.abs((reconstructed_training_data - mll.model.train_targets).detach().numpy()))
        self.reconstruction_error.append(reconstruction_error)

        if self.only_store_best_step or state_dict is None or parameters is None:
            return

        self.X_during_training = np.stack([*self.X_during_training, X])
        if len(self.parameters) == 0:
            self.parameters = {k: [] for k, _ in parameters.items()}
        for k, _ in self.parameters.items():
            self.parameters[k] = np.stack([*self.parameters[k], parameters[k]])

        # Save the variables to the pre-defined directory. This should overwrite the previously saved variables in the directory
        # 1: Variables that are overwritten
        torch.save(self.best_state_dict, os.path.join(self.path, "model_best_state_dict.pth"))
        torch.save(self.state_dict, os.path.join(self.path, "model_state_dict.pth"))

        # 2: Variables that are appended to at each training iteration
        torch.save(self.steps, os.path.join(self.path, "steps.pt"))
        # torch.save(self.parameters, os.path.join(self.path, "parameters.pt"))
        torch.save(self.losses, os.path.join(self.path, "losses.pt"))
        torch.save(self.reconstruction_error, os.path.join(self.path, "recon_errors.pt"))
        torch.save(self.curvature, os.path.join(self.path, "curvature.pt"))

        if isinstance(mll.model.added_loss, StressLossTermExactMLL):
            torch.save(self.stess_loss, os.path.join(self.path, "stress_losses.pt"))

