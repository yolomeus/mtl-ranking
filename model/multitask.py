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
    """A model consisting of a body and multiple head modules with the latter being grouped into a `MultiModule`.
    """

    def __init__(self, body: Module, head: MultiModule):
        super().__init__()
        self.body = body
        self.head = head

    def forward(self, name_to_batch: dict):
        """
        :param name_to_batch: a mapping from input module name to the batch of inputs to be passed to the corresponding
        model input.

        :return: a dict mapping from output module names to corresponding predictions.
        """
        preds = {name: self.head(self.body(batch), name)
                 for name, batch in name_to_batch.items()}
        return preds

    @property
    def output_names(self):
        return self.head.module_names
