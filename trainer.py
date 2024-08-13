import torch
import time
import copy
from abc import ABC, abstractmethod
from typing import List, Dict, Union, Optional, Callable, Tuple
from torch.cuda.amp import GradScaler, autocast


def multiclass_accuracy_score(labels: torch.Tensor, raw_predictions: torch.Tensor) -> float:
    predictions = torch.argmax(raw_predictions, dim=-1)
    correct = (predictions == labels).sum().item()
    return correct / labels.size(0)


def binary_accuracy_score(labels: torch.Tensor, raw_predictions: torch.Tensor) -> float:
    predictions = torch.round(torch.sigmoid(raw_predictions))  # Assuming threshold of 0.5
    correct = (predictions == labels).sum().item()
    return correct / labels.size(0)


def multiclass_precision_score(labels: torch.Tensor, raw_predictions: torch.Tensor,
                               average: str = 'macro') -> Optional[Union[float, torch.tensor]]:
    y_pred = torch.argmax(raw_predictions, dim=1)
    classes = torch.unique(labels)
    precision_per_class = []

    for cls in classes:
        # True Positives (TP): correctly predicted instances of class `cls`
        tp = torch.sum((labels == cls) & (y_pred == cls)).item()

        # False Positives (FP): instances predicted as class `cls` but are actually some other class
        fp = torch.sum((labels != cls) & (y_pred == cls)).item()

        # Precision calculation for class `cls`
        precision = tp / (tp + fp) if tp + fp > 0 else 0.0  # Modified conditional
        precision_per_class.append(precision)

    precision_per_class = torch.tensor(precision_per_class)

    if average == 'macro':
        return precision_per_class.mean().item()
    elif average == 'micro':
        # Global counts
        tp_total = torch.sum(labels == y_pred).item()
        fp_total = torch.sum((labels != y_pred) & (y_pred != -1)).item()  # Note: Assuming -1 is not a valid class label
        return tp_total / (tp_total + fp_total)
    elif average == 'weighted':
        # Calculate support for each class
        support = torch.tensor([torch.sum(labels == cls).item() for cls in classes])
        return (precision_per_class * support / support.sum()).sum().item()
    else:
        return precision_per_class


def binary_precision_score(labels: torch.Tensor, raw_predictions: torch.Tensor) -> float:
    predictions = torch.round(torch.sigmoid(raw_predictions))
    tp = torch.sum((labels == 1) & (predictions == 1)).item()
    fp = torch.sum((labels == 0) & (predictions == 1)).item()
    return tp / (tp + fp) if tp + fp > 0 else 0.0  # Modified conditional


def multiclass_recall_score(labels: torch.Tensor, raw_predictions: torch.Tensor,
                            average: str = 'macro') -> Optional[Union[float, torch.tensor]]:
    # Convert logits to predicted class labels
    predictions = torch.argmax(raw_predictions, dim=1)

    # Get the unique classes
    classes = torch.unique(labels)

    # Initialize variables
    recall_per_class = []

    for cls in classes:
        # True Positives (TP): correctly predicted instances of class `cls`
        tp = torch.sum((labels == cls) & (predictions == cls)).item()

        # False Negatives (FN): instances of class `cls` that were predicted as some other class
        fn = torch.sum((labels == cls) & (predictions != cls)).item()

        # Recall calculation for class `cls`
        recall = tp / (tp + fn) if tp + fn > 0 else 0.0  # Modified conditional

        recall_per_class.append(recall)

    recall_per_class = torch.tensor(recall_per_class)

    if average == 'macro':
        return recall_per_class.mean().item()
    elif average == 'micro':
        # Global counts
        tp_total = torch.sum(labels == predictions).item()
        fn_total = torch.sum((labels != predictions) & (labels == labels)).item()  # Fixed condition
        return tp_total / (tp_total + fn_total)
    elif average == 'weighted':
        # Calculate support for each class
        support = torch.tensor([torch.sum(labels == cls).item() for cls in classes])
        return (recall_per_class * support / support.sum()).sum().item()
    else:
        return recall_per_class


def binary_recall_score(labels: torch.Tensor, raw_predictions: torch.Tensor) -> float:
    # Convert logits to predicted class labels (assuming a threshold of 0.5)
    predictions = torch.round(torch.sigmoid(raw_predictions))

    # True Positives (TP): correctly predicted positive instances
    tp = torch.sum((labels == 1) & (predictions == 1)).item()

    # False Negatives (FN): instances of class `1` predicted as `0`
    fn = torch.sum((labels == 1) & (predictions == 0)).item()

    # Recall calculation
    return tp / (tp + fn) if tp + fn > 0 else 0.0  # Modified conditional


def r2_score(labels: torch.Tensor, predictions: torch.Tensor) -> float:
    labels_mean = torch.mean(labels, dim=0)
    tss = torch.sum((labels - labels_mean) ** 2)
    rss = torch.sum((labels - predictions) ** 2)
    delta = 1.e-10
    tss += delta
    return (1 - rss / tss).item()


class Callback(ABC):
    """
    Abstract base class for callbacks.
    """

    def __init__(self, pos: int = 1):
        """
        The position at which the callback will run
        Options:
            0 - After training 1 batch (loss.backward() and optimizer.step(), before evaluation/validation (runs for each batch in each epoch) --> return values are not used.
            1 (default) - After validation step. Runs once per epoch. (best for metric tracking, logging, etc.) --> return values (str) are used to print at the end of the epoch
            2 - After training step and before validation step. Runs once per epoch. --> return values are not used.
            Note - for some callbacks, returned value is not used, you can still print within it.
        """
        self.pos = pos

    @abstractmethod
    def runner(self, trainer: 'Trainer') -> Optional[str]:
        """
        Carry out any operation at desired epoch number
        Args :
            trainer: object of Trainer class on which Callback is applied
        Return:
            (Optional) A message to print in the end of epoch while training
        """
        pass

    # Do not overwrite this method (run)
    def run(self, trainer: 'Trainer', pos: int) -> Optional[str]:
        try:
            if pos == self.pos:
                return self.runner(trainer)
            return None
        except Exception as e:
            print(f"Callback Error! : {e}")
            return None


class Trainer:
    def __init__(self, model: torch.nn.Module,
                 epochs: int,
                 criterion: torch.nn.Module,
                 optimizer: torch.optim.Optimizer,
                 metrics: Optional[Union[str, List[str]]] = None,
                 metric_func_dict: Optional[Dict[str, Callable]] = None,
                 binary_output: bool = False, callbacks: Optional[List[Callback]] = None,
                 display_time_elapsed: bool = False,
                 device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                 roff: int = 5, report_in_one_line: bool = True,
                 clear_cuda_cache: bool = True,
                 mixed_precision_training: bool = True,
                 grad_clip: Optional[int] = None):

        """
        Initializes the pytorch training loop class.

        Args:
            model (torch.nn.Module): The model to be trained.
            epochs (int): The number of epochs to train the model.
            criterion (torch.nn.Module): The loss function to use during training.
            optimizer (torch.optim.Optimizer): The optimizer to use during training.
            metrics (list, optional): A list of metric names to track during training. Defaults to None.
            metric_func_dict (dict, optional): A dictionary mapping metric names to their corresponding functions. Defaults to None.
            binary_output (bool, optional): If True, assumes the model has a single output. Defaults to False.
            callbacks (list, optional): A list of callback instances to be executed during training. Defaults to None.
            display_time_elapsed (bool, optional): If True, displays the time elapsed after each epoch. Defaults to False.
            device (torch.device, optional): The device to train the model on. Defaults to GPU (if available, else CPU).
            roff (int, optional): The number of decimal places to round off the results to. Defaults to 5.
            report_in_one_line (bool, optional): If True, reports the training progress at each epoch in a single line. Defaults to True.
            clear_cuda_cache (bool, optional) : If True, clears the cuda cache at the beginning of each epoch, only if device is cuda.
            mixed_precision_training (bool, optional) : If true implements mixed precision training in Pytorch
        """
        self.b = None
        self.batch_size = None
        self.messages_joiner = "  ||  " if report_in_one_line else "\n"
        self.num_batches = None
        self.epoch_message = None
        self.epochs = epochs

        callbacks = callbacks if isinstance(callbacks, list) else []
        self.callbacks = [cb for cb in callbacks if isinstance(cb, Callback)]

        metrics = metrics if isinstance(metrics, list) else [metrics]
        prior_metric_fn_dict = {
            'accuracy': binary_accuracy_score if binary_output else multiclass_accuracy_score,
            'precision': binary_precision_score if binary_output else multiclass_precision_score,
            'recall': binary_recall_score if binary_output else multiclass_recall_score,
            'r2_score': r2_score,
            None: lambda a, b: 0.}
        if metric_func_dict is not None:
            prior_metric_fn_dict.update(metric_func_dict)
        metric_fns = []
        filtered_metrics = []
        for metric in metrics:
            if metric in prior_metric_fn_dict:
                filtered_metrics.append(metric)
                metric_fns.append(prior_metric_fn_dict[metric])
            else:
                error_message = f"Please provide a scoring function for your metric : {metric} using metric_fn_dict argument\n" + \
                                f"Example \n  >>> trainer = Trainer(other_args,metric_fn_dict = {'{'}{metric} : {metric}_fn{'}'})\n" + \
                                f"Or use add_metric method\n" + \
                                f"  >>> trainer.add_metric({metric},{metric}_fn)\n" + \
                                f"Note : {metric}_fn expected to return float. Convert tensors to float using T.item() for single output for better logging "
                raise AttributeError(error_message)

        self.metrics = filtered_metrics
        self.metric_fns = metric_fns

        self.model = model
        self.model.to(device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.clear_cuda_cache = clear_cuda_cache
        self.scaler = GradScaler() if (self.device.type == 'cuda' and mixed_precision_training) else None

        self.grad_clip = grad_clip

        self.roff = roff
        self.STOPPER = False
        self.running_loss = 0.
        self.running_metrics = torch.zeros(len(self.metrics))

        self.current_epoch = 0
        self.best_model_weights = None
        self.display_time_elapsed = display_time_elapsed
        self.History = {'loss': [], "vloss": [], "epochs": []}
        if self.metrics[0]:
            for metric in self.metrics:
                self.History.update({metric: [], f"v{metric}": []})

    def add_metric(self, metric: str, metric_fn: Callable) -> None:
        """
        Adds a new metric to display and monitor while training
        Args:
            metric: Name of the metric
            metric_fn: Score function for the metric
        Return: None
        """
        self.metrics.append(metric)
        self.metric_fns.append(metric_fn)
        return None

    @classmethod
    def add_method(cls) -> Callable:
        """
        A decorator to add a new method to the Trainer class.

        Use case:
            @Trainer.add_method()
            def method(self):
                # CODE #
                return value

        This allows dynamically adding methods to the Trainer class.

        Returns:
            function: The decorator function that sets the new method to the class.
        """

        def decorator(func: Callable) -> Callable:
            setattr(cls, func.__name__, func)
            return func

        return decorator

    def add_callback(self, callback: Callback) -> None:
        """
        Adds a callback to the Trainer.

        Note:
            If you're adding a custom callback function, make sure it's inherited
            from the `Callback` abstract base class and overwrites the `run` method,
            otherwise the callback will not run!

        Args:
            callback (Callback): Callback object to add. Must be an instance of
                                 a class inherited from the `Callback` base class.

        """
        if (callback not in self.callbacks) and isinstance(callback, Callback):
            self.callbacks.append(callback)

    def remove_callback(self, callback: Callback) -> None:
        """
        Removes a callback from the Trainer.

        Args:
            callback: Callback object to remove.
        """
        if callback in self.callbacks:
            self.callbacks.remove(callback)

    def __run_callbacks(self, pos: int) -> List[Optional[str]]:
        responses = [callback.run(self, pos) for callback in self.callbacks]
        return [response for response in responses if response]

    def __train_fn(self, train_loader: torch.utils.data.DataLoader) -> Tuple[float, torch.Tensor]:

        # Initializing Metrics
        batch_loss = 0.
        batch_metrics = torch.zeros(len(self.metrics))
        self.running_loss = 0.
        self.running_metrics.zero_()

        # Set to training mode
        self.model.train()
        for self.b, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            # One Batch Training
            self.optimizer.zero_grad()
            if self.scaler:
                # Mixed precision training
                with autocast():
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, labels)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # Normal training
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

            batch_loss += loss.item()
            self.running_loss += loss.item()
            with torch.no_grad():
                for i, metric_fn in enumerate(self.metric_fns):
                    metric_value = metric_fn(labels, outputs)
                    batch_metrics[i] += metric_value
                    self.running_metrics[i] += metric_value
            self.__run_callbacks(pos=0)

        avg_batch_loss = batch_loss / self.num_batches
        avg_batch_metrics = batch_metrics / self.num_batches
        return avg_batch_loss, avg_batch_metrics

    @torch.no_grad()
    def __validation_fn(self, val_loader: torch.utils.data.DataLoader) -> Tuple[float, torch.Tensor]:
        running_vloss = 0.
        running_vmetrics = torch.zeros(len(self.metrics))
        # Set to the evaluation mode
        self.model.eval()
        # Disable gradient computation and reduce memory consumption.
        for vinputs, vlabels in val_loader:
            vinputs, vlabels = vinputs.to(self.device), vlabels.to(self.device)
            if self.scaler:
                with autocast():
                    voutputs = self.model(vinputs)
                    vloss = self.criterion(voutputs, vlabels)
            else:
                voutputs = self.model(vinputs)
                vloss = self.criterion(voutputs, vlabels)
            running_vloss += vloss.item()
            for i, metric_fn in enumerate(self.metric_fns):
                running_vmetrics[i] += metric_fn(vlabels, voutputs)

        avg_vloss = running_vloss / len(val_loader)
        avg_vmetrics = running_vmetrics / len(val_loader)
        return avg_vloss, avg_vmetrics

    def fit(self, train_loader: torch.utils.data.DataLoader, val_loader: torch.utils.data.DataLoader,
            train_fn: Optional[Callable] = None, validation_fn: Optional[Callable] = None) -> None:
        """
        Trains the model for the specified number of epochs.

        Args:
            train_loader (torch.utils.data.DataLoader): DataLoader for training data.
            val_loader (torch.utils.data.DataLoader): DataLoader for validation data.
            train_fn (function, optional): Custom training function for one epoch. If None, uses the default training function. Defaults to None.
            validation_fn (function, optional): Custom validation function for one epoch. If None, uses the default validation function. Defaults to None.

        Returns:
            None
        """
        self.num_batches = len(train_loader)
        if (train_fn is None) and (validation_fn is None):
            train_fn = self.__train_fn
            validation_fn = self.__validation_fn
        self.batch_size = train_loader.batch_size
        start_time = time.time()
        on_gpu = True if self.device.type == 'cuda' else False
        # The main Training loop
        for epoch in range(self.epochs):

            if on_gpu and self.clear_cuda_cache:
                torch.cuda.empty_cache()

            self.epoch_message = f'EPOCH {self.current_epoch + 1}:'

            # Train model
            current_loss, current_metrics = train_fn(train_loader)
            self.History['loss'].append(current_loss)
            if self.metrics[0]:
                for i, metric in enumerate(self.metrics):
                    self.History[metric].append(current_metrics[i].item())

            self.__run_callbacks(pos=2)

            # Validate model
            current_vloss, current_vmetrics = validation_fn(val_loader)
            self.History['vloss'].append(current_vloss)
            if self.metrics[0]:
                for i, metric in enumerate(self.metrics):
                    self.History[f'v{metric}'].append(current_vmetrics[i].item())

            # Run callbacks
            responses = self.__run_callbacks(pos=1)

            print(self.epoch_message)
            long_message = f"LOSS : train {round(current_loss, self.roff)} validation {round(current_vloss, self.roff)}"
            if self.metrics[0]:
                for i, metric in enumerate(self.metrics):
                    message = f"{metric} : train {round(current_metrics[i].item(), self.roff)} validation {round(current_vmetrics[i].item(), self.roff)}"
                    long_message += (self.messages_joiner + message)
            print(long_message)

            if self.display_time_elapsed:
                end_time = time.time()
                print(f"Time elapsed : {end_time - start_time} s")

            for response in responses:
                print(response)

            self.current_epoch += 1

            if self.STOPPER:
                break
        self.History['epochs'] = [epoch for epoch in range(self.current_epoch)]
        if self.best_model_weights is None:
            self.best_model_weights = copy.deepcopy(self.model.state_dict())

    def run_per_epoch(self, pos=1) -> Callable:
        class Custom(Callback):
            def __init__(self):
                super().__init__(pos=pos)

            def runner(self, trainer: Trainer) -> Optional[str]:
                pass

        def decorator(func: Callable) -> Callable:
            Custom.runner = func
            return func

        self.callbacks.append(Custom())
        return decorator


class IntraEpochReport(Callback):

    def __init__(self, reports_per_epoch: int, report_in_one_line: bool = True):
        """
        Initializes the IntraEpochReport callback.

        Args:
            reports_per_epoch (int): The number of reports to generate per epoch.
            report_in_one_line (bool, optional): If True, reports will be displayed in one line. Defaults to True.
        """
        super().__init__(pos=0)
        self.reports_per_epoch = reports_per_epoch
        self.log_batches = 0
        self.messages_joiner = " " if report_in_one_line else "\n"

    def runner(self, trainer: Trainer) -> None:
        if trainer.current_epoch == 0:
            self.log_batches = max(1, trainer.num_batches // self.reports_per_epoch)

        if trainer.b % self.log_batches == self.log_batches - 1:
            last_loss = trainer.running_loss / self.log_batches  # loss per batch
            long_message = f"  E-{trainer.current_epoch + 1} batch {trainer.b + 1} loss: {round(last_loss, trainer.roff)}"
            if trainer.metrics[0]:
                last_metrics = trainer.running_metrics / self.log_batches  # loss per batch
                for i, metric in enumerate(trainer.metrics):
                    message = f"{metric}: {round(last_metrics[i].item(), trainer.roff)}"
                    long_message += (self.messages_joiner + message)
            print(long_message)
            trainer.running_loss = 0.
            trainer.running_metrics.zero_()


class EarlyStopping(Callback):
    """
    Callback to stop training early based on a specified condition.

    Attributes:
        basis: Metric to monitor for early stopping.
        metric_minimize: Whether to minimize the metric.
        patience: Number of epochs with no improvement after which training will be stopped.
        threshold: Threshold for the monitored metric, without crossing this, model cannot stop.
        restore_best_weights: Whether to restore the best model weights.
    """

    def __init__(self, basis: str, metric_minimize: bool = True, patience: int = 5,
                 threshold: Optional[float] = None, restore_best_weights: bool = True):
        super().__init__()

        self.best_epoch = 0
        self.basis = basis
        self.metric_minimize = metric_minimize
        self.patience = patience
        self.threshold = threshold
        self.best_value = float('inf') if metric_minimize else float('-inf')
        self.restore_best_weights = restore_best_weights
        self.instance = 0
        self.multi_instances = False
        self.called = False

    def runner(self, trainer: Trainer) -> Optional[str]:
        """
        Runs the early stopping check and updates trainer state if needed.

        Args:
            trainer: Trainer object.

         Returns:
            Optional message if restoring best weights.
        """
        if trainer.current_epoch == 0:
            for callback in trainer.callbacks:
                if isinstance(callback, EarlyStopping) and callback != self:
                    self.multi_instances = True
                    self.instance = callback.instance + 1 if callback.instance >= self.instance else self.instance
            if not self.multi_instances:
                del self.called
                del self.instance
        History = trainer.History
        metric = History[self.basis][trainer.current_epoch]

        if ((self.metric_minimize and (metric < self.best_value))
                or ((not self.metric_minimize) and (metric > self.best_value))):  # Minimize Loss
            self.best_value = metric
            trainer.best_model_weights = copy.deepcopy(trainer.model.state_dict())
            self.best_epoch = trainer.current_epoch
        else:
            if (self.threshold is None) or ((self.metric_minimize
                                             and (self.best_value < self.threshold))
                                            or ((not self.metric_minimize)
                                                and (self.best_value > self.threshold))):
                self.patience -= 1

            if self.multi_instances:
                trainer.epoch_message += f" <es{self.instance}-{self.basis}-p-{self.patience}>"
            else:
                trainer.epoch_message += f" <es-p-{self.patience}>"
            last_epoch = (trainer.current_epoch + 1 == trainer.epochs) and trainer.best_model_weights

            if self.patience == 0 or last_epoch:

                if self.multi_instances:
                    for callback in trainer.callbacks:
                        if isinstance(callback, EarlyStopping) and callback != self:
                            if callback.called:
                                return

                if last_epoch:
                    print(f"Stopping at last epoch {trainer.current_epoch + 1}")
                else:
                    print(f"Early-stopping at epoch {trainer.current_epoch + 1}, basis : {self.basis}")
                trainer.STOPPER = True  # This will break the loop
                if self.restore_best_weights:
                    final_message = "restoring best weights..." + \
                                    f" {trainer.model.load_state_dict(trainer.best_model_weights)}" + \
                                    f"\n\tBest epoch : {self.best_epoch + 1} ," + \
                                    f"\n\ttraining loss : {History['loss'][self.best_epoch]}," + \
                                    f"\n\tvalidation loss : {History['vloss'][self.best_epoch]},"

                    if trainer.metrics[0]:
                        for metric in trainer.metrics:
                            message = f"\n\ttraining {metric} : {History[metric][self.best_epoch]}," + \
                                      f"""\n\tvalidation {metric} :{History[f"v{metric}"][self.best_epoch]}"""
                            final_message += message

                    if self.multi_instances:
                        self.called = True

                    return final_message
        return None


class GradientClipping(Callback):
    def __init__(self, clipping_value):
        super().__init__(pos=0)
        self.clipping_value = clipping_value

    def runner(self, trainer: Trainer):
        torch.nn.utils.clip_grad_value_(trainer.model.parameters(), self.clipping_value)


class LRScheduler(Callback):
    def __init__(self, scheduler):
        super().__init__(pos=0)
        self.scheduler = scheduler

    def runner(self, trainer: Trainer):
        self.scheduler.step()
        return None


class LRTracker(Callback):
    def __init__(self):
        super().__init__(pos=1)

    @staticmethod
    def get_lr(optimizer):
        for param_group in optimizer.param_groups:
            return param_group['lr']

    def runner(self, trainer: Trainer):
        try:
            trainer.History['lr'].append(self.get_lr(trainer.optimizer))
        except Exception:
            trainer.History['lr'] = [self.get_lr(trainer.optimizer)]
        return None
