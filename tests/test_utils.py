import pytest

import sys
sys.path.append("..")

from GeostrophicVelocityDataset import GeostrophicVelocityDataset


def test_init_without_date():
    GeostrophicVelocityDataset()
