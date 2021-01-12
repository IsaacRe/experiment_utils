import torch.nn as nn
from experiment_utils.utils.hook_management import HookManager
from experiment_utils.utils.helpers import find_network_modules_by_name


class LayerInjector:

    def __init__(self, model: nn.Module):
        """
        Allows injection of torch.nn.Module's after a specified layer in a network by registering forward hooks
        on that layer. Injected modules will have their forward function called following calls to the hooked module's
        forward function and those of all previously inserted modules.
        :param model: the model in which modules will be injected. Should have valid named_modules() function.
        """
        self.hook_manager = HookManager(wrap_calls=False)
        self.model = model
        self.inserted_modules = {}
        model.inserted_modules = self.inserted_modules

    def _wrap_forward(self, module: nn.Module):
        def forward_hook(hooked_module, inp, out):
            return module(out)
        return forward_hook

    def insert_at_layer(self, module: nn.Module, layer_name: str):
        """
        Appends module forward function as a forward hook of the module specified by layer_name
        :param module: the module to insert
        :param layer_name: the layer after which to insert the module
        """
        hooked_module, = find_network_modules_by_name(self.model, [layer_name])
        self.inserted_modules[layer_name] = module
        if not hasattr(hooked_module, 'appended_modules'):
            hooked_module.appended_modules = []
        hooked_module.appended_modules += [module]
        self.hook_manager.register_forward_hook(self._wrap_forward(module),**{layer_name: hooked_module})
