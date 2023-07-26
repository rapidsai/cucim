import lazy_loader as lazy

submodules = [
    'color',
    'data',
    'exposure',
    'feature',
    'filters',
    'measure',
    'metrics',
    'morphology',
    'registration',
    'restoration',
    'segmentation',
    'transform',
    'util',
]

__all__ = submodules

from . import (color, data, exposure, feature, filters, measure, metrics,
               morphology, registration, restoration, segmentation, transform,
               util)
