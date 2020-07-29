"""Top-level package for sd-nn."""
from sdNN.keras_version.main import KerasSDNN
from sdNN.pytorch_version.main import PytorchSDNN

__author__ = """Nidhal Baccouri"""
__email__ = 'nidhalbacc@gmail.com'
__version__ = '0.1.1'

__all__ = [KerasSDNN, PytorchSDNN]
