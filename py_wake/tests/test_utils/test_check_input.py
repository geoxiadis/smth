from pathlib import Path

import pytest

from py_wake import np
from py_wake.tests import clear_ptf, ptf
from py_wake.utils.check_input import check_input, is_list_like
import time


def test_check_input():
    input_space = [(0, 1), (100, 200)]

    with pytest.raises(ValueError, match="Input, index_0, with value, 2 outside range 0-1"):
        check_input(input_space, np.array([(2, 150)]).T)

    with pytest.raises(ValueError, match="Input, index_1, with value, 50 outside range 100-200"):
        check_input(input_space, np.array([(1, 50)]).T)

    with pytest.raises(ValueError, match="Input, wd, with value, 250 outside range 100-200"):
        check_input(input_space, np.array([(1, 250)]).T, ['ws', 'wd'])

    check_input(input_space, np.array([(1, 200)]).T, ['ws', 'wd'])


def test_ptf():
    f = Path(ptf('test.txt', 'ecd71870d1963316a97e3ac3408c9835ad8cf0f3c1bc703527c30265534f75ae'))
    assert f.read_text() == 'test123'
    for _ in range(3):
        try:
            clear_ptf()
            break
        except Exception:
            time.sleep(0.5)
    assert f.exists() is False
    assert f.parent.exists() is False


def test_is_list_like():
    assert is_list_like([1, 2])
    assert is_list_like((1, 2))
    assert is_list_like({1, 2})
    assert not is_list_like("abc")
    assert not is_list_like(bytes("abc", "utf-8"))
    assert not is_list_like({"a": 1})
