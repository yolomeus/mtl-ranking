import torch
from torch.nn import Module, ModuleDict


class MultiModule(Module):
    """A set of pytorch modules with a `forward()` method that allows to specify which module to use by passing a
    string key. This class is needed in order to dynamically instantiate an arbitrary number of modules using hydra.
    (not possible with ModuleDict only)
    """

    def __init__(self, **input_modules):
        super().__init__()
        self.in_modules = ModuleDict(input_modules)

    def forward(self, inputs, module_name: str):
        return self.in_modules[module_name](inputs)

    @property
    def module_names(self):
        return [name for name in self.in_modules.keys()]


class MultiTaskModel(Module):
    """A model consisting of multiple input and head modules, each grouped into a `MultiModel` and a single body module.
    """

    def __init__(self, input_module: MultiModule, body: Module, head_module: MultiModule):
        super().__init__()

        self.input_module = input_module
        self.body = body
        self.head_module = head_module

    def forward(self, name_to_batch: dict):
        """
        :param name_to_batch: a mapping from input module name to the batch of inputs to be passed to the corresponding
        model input.

        :return: a dict mapping from output module names to corresponding predictions.
        """
        input_reps = [self.input_module(batch, name)
                      for name, batch in name_to_batch.items()]

        stacked_batch = torch.cat(input_reps, dim=0)
        hidden_reps = self.body(stacked_batch)

        batch_sizes = [len(x) for x in name_to_batch.values()]
        hidden_batches = torch.split(hidden_reps, batch_sizes)

        preds = {name: self.head_module(x, name)
                 for name, x in zip(name_to_batch, hidden_batches)}

        return preds

    @property
    def output_names(self):
        return self.head_module.module_names
