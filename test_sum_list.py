import pytest
from sum_list import sum_list


@pytest.mark.parametrize("numbers, expected", [
    ([1, 2, 3], 6),
    ([10, 20, 30], 60),
    ([-1, -2, -3], -6),
    ([0, 0, 0], 0),
    ([1.5, 2.5], 4.0),
    ([], 0),
    ([42], 42),
])
def test_sum_list(numbers, expected):
    assert sum_list(numbers) == expected
