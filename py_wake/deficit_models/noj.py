from py_wake.utils.model_utils import DeprecatedModel
from py_wake.deficit_models import deprecated
from py_wake import wind_farm_models


class NOJ(wind_farm_models.predefined.NOJ, DeprecatedModel):
    def __init__(self, *args, **kwargs):
        wind_farm_models.predefined.NOJ.__init__(self, *args, **kwargs)
        DeprecatedModel.__init__(self, "py_wake.wind_farm_models.predefined.NOJ")


class NOJLocal(wind_farm_models.predefined.NOJLocal, DeprecatedModel):
    def __init__(self, *args, **kwargs):
        wind_farm_models.predefined.NOJLocal.__init__(self, *args, **kwargs)
        DeprecatedModel.__init__(self, "py_wake.wind_farm_models.predefined.NOJLocal")


class NOJDeficit(deprecated.noj.NOJDeficit, DeprecatedModel):
    def __init__(self, *args, **kwargs):
        deprecated.noj.NOJDeficit.__init__(self, *args, **kwargs)
        DeprecatedModel.__init__(self, "py_wake.deficit_models.deprecated.noj.NOJDeficit")


class NOJLocalDeficit(deprecated.noj.NOJLocalDeficit, DeprecatedModel):
    def __init__(self, *args, **kwargs):
        deprecated.noj.NOJLocalDeficit.__init__(self, *args, **kwargs)
        DeprecatedModel.__init__(self, "py_wake.deficit_models.deprecated.noj.NOJLocalDeficit")


class TurboNOJDeficit(deprecated.noj.TurboNOJDeficit, DeprecatedModel):
    def __init__(self, *args, **kwargs):
        deprecated.noj.TurboNOJDeficit.__init__(self, *args, **kwargs)
        DeprecatedModel.__init__(self, "py_wake.deficit_models.deprecated.noj.TurboNOJDeficit")
