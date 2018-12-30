from copy import deepcopy

from codes.net.base_net import Net


class BaseComponent(Net):
    """
    Base class to implement all the modules needed in the message passing neural networks.
    All the components must extend this class
    """

    def __init__(self, model_config):
        super(BaseComponent, self).__init__(model_config)
        self.name = "base component for the message passing networks"
        self._parameter_list = None
        self._module_list = None
        self.config = None

        self.init_config(model_config)
        self.init_parameters_and_modules()

    def init_config(self, model_config):
        self.config = deepcopy(model_config)

    def init_parameters_and_modules(self):
        pass

    def forward(self, data):
        return None

    def get_name(self):
        return self.name

    def get_config(self):
        return self.config
