import pytest

import sys
sys.path.append("..")

from GeostrophicVelocityDataReader import GeostrophicVelocityDataReader


def test_init_without_date():
    GeostrophicVelocityDataReader()
