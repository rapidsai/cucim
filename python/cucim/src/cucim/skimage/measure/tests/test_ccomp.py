# Note: These test cases originated in skimage/morphology/tests/test_ccomp.py

import cupy as cp
# import numpy as np
from cupy.testing import assert_array_equal

from cucim.skimage.measure import label

# import pytest

# import cucim.skimage.measure._ccomp as ccomp


BG = 0  # background value


class TestConnectedComponents:
    def setup(self):
        # fmt: off
        self.x = cp.array([
            [0, 0, 3, 2, 1, 9],
            [0, 1, 1, 9, 2, 9],
            [0, 0, 1, 9, 9, 9],
            [3, 1, 1, 5, 3, 0]])

        self.labels = cp.array([
            [0, 0, 1, 2, 3, 4],
            [0, 5, 5, 4, 2, 4],
            [0, 0, 5, 4, 4, 4],
            [6, 5, 5, 7, 8, 0]])
        # fmt: on

        # No background - there is no label 0, instead, labelling starts with 1
        # and all labels are incremented by 1.
        self.labels_nobg = self.labels + 1
        # The 0 at lower right corner is isolated, so it should get a new label
        self.labels_nobg[-1, -1] = 10

        # We say that background value is 9 (and bg label is 0)
        self.labels_bg_9 = self.labels_nobg.copy()
        self.labels_bg_9[self.x == 9] = 0
        # Then, where there was the label 5, we now expect 4 etc.
        # (we assume that the label of value 9 would normally be 5)
        self.labels_bg_9[self.labels_bg_9 > 5] -= 1

    def test_basic(self):
        assert_array_equal(label(self.x), self.labels)

        # Make sure data wasn't modified
        assert self.x[0, 2] == 3

        # Check that everything works if there is no background
        assert_array_equal(label(self.x, background=99), self.labels_nobg)
        # Check that everything works if background value != 0
        assert_array_equal(label(self.x, background=9), self.labels_bg_9)

    def test_random(self):
        x = (cp.random.rand(20, 30) * 5).astype(int)
        labels = label(x)

        n = int(labels.max())
        for i in range(n):
            values = x[labels == i]
            assert cp.all(values == values[0])

    def test_diag(self):
        # fmt: off
        x = cp.array([[0, 0, 1],
                      [0, 1, 0],
                      [1, 0, 0]])
        assert_array_equal(label(x), x)
        # fmt: on

    def test_4_vs_8(self):
        # fmt: off
        x = cp.array([[0, 1],
                      [1, 0]], dtype=int)

        assert_array_equal(label(x, connectivity=1),
                           [[0, 1],
                            [2, 0]])
        assert_array_equal(label(x, connectivity=2),
                           [[0, 1],
                            [1, 0]])
        # fmt: on

    def test_background(self):
        # fmt: off
        x = cp.array([[1, 0, 0],
                      [1, 1, 5],
                      [0, 0, 0]])

        assert_array_equal(label(x), [[1, 0, 0],
                                      [1, 1, 2],
                                      [0, 0, 0]])

        assert_array_equal(label(x, background=0),
                           [[1, 0, 0],
                            [1, 1, 2],
                            [0, 0, 0]])
        # fmt: on

    def test_background_two_regions(self):
        # fmt: off
        x = cp.array([[0, 0, 6],
                      [0, 0, 6],
                      [5, 5, 5]])

        res = label(x, background=0)
        assert_array_equal(res,
                           [[0, 0, 1],
                            [0, 0, 1],
                            [2, 2, 2]])
        # fmt: on

    def test_background_one_region_center(self):
        # fmt: off
        x = cp.array([[0, 0, 0],
                      [0, 1, 0],
                      [0, 0, 0]])

        assert_array_equal(label(x, connectivity=1, background=0),
                           [[0, 0, 0],
                            [0, 1, 0],
                            [0, 0, 0]])
        # fmt: on

    def test_return_num(self):
        # fmt: off
        x = cp.array([[1, 0, 6],
                      [0, 0, 6],
                      [5, 5, 5]])
        # fmt: on
        assert_array_equal(label(x, return_num=True)[1], 3)

        assert_array_equal(label(x, background=-1, return_num=True)[1], 4)


class TestConnectedComponents3d:
    def setup(self):
        self.x = cp.zeros((3, 4, 5), int)
        # fmt: off
        self.x[0] = cp.array([[0, 3, 2, 1, 9],
                              [0, 1, 9, 2, 9],
                              [0, 1, 9, 9, 9],
                              [3, 1, 5, 3, 0]])

        self.x[1] = cp.array([[3, 3, 2, 1, 9],
                              [0, 3, 9, 2, 1],
                              [0, 3, 3, 1, 1],
                              [3, 1, 3, 3, 0]])

        self.x[2] = cp.array([[3, 3, 8, 8, 0],
                              [2, 3, 9, 8, 8],
                              [2, 3, 0, 8, 0],
                              [2, 1, 0, 0, 0]])

        self.labels = cp.zeros((3, 4, 5), int)

        self.labels[0] = cp.array([[0, 1, 2, 3, 4],
                                   [0, 5, 4, 2, 4],
                                   [0, 5, 4, 4, 4],
                                   [1, 5, 6, 1, 0]])

        self.labels[1] = cp.array([[1, 1, 2, 3, 4],
                                   [0, 1, 4, 2, 3],
                                   [0, 1, 1, 3, 3],
                                   [1, 5, 1, 1, 0]])

        self.labels[2] = cp.array([[1, 1, 7, 7, 0],
                                   [8, 1, 4, 7, 7],
                                   [8, 1, 0, 7, 0],
                                   [8, 5, 0, 0, 0]])
        # fmt: on

    def test_basic(self):
        labels = label(self.x)
        assert_array_equal(labels, self.labels)

        assert self.x[0, 0, 2] == 2, "Data was modified!"

    def test_random(self):
        x = (cp.random.rand(20, 30) * 5).astype(int)
        labels = label(x)

        n = int(labels.max())
        for i in range(n):
            values = x[labels == i]
            assert cp.all(values == values[0])

    def test_diag(self):
        x = cp.zeros((3, 3, 3), int)
        x[0, 2, 2] = 1
        x[1, 1, 1] = 1
        x[2, 0, 0] = 1
        assert_array_equal(label(x), x)

    def test_4_vs_8(self):
        x = cp.zeros((2, 2, 2), int)
        x[0, 1, 1] = 1
        x[1, 0, 0] = 1
        label4 = x.copy()
        label4[1, 0, 0] = 2
        assert_array_equal(label(x, connectivity=1), label4)
        assert_array_equal(label(x, connectivity=3), x)

    def test_connectivity_1_vs_2(self):
        x = cp.zeros((2, 2, 2), int)
        x[0, 1, 1] = 1
        x[1, 0, 0] = 1
        label1 = x.copy()
        label1[1, 0, 0] = 2
        assert_array_equal(label(x, connectivity=1), label1)
        assert_array_equal(label(x, connectivity=3), x)

    def test_background(self):
        x = cp.zeros((2, 3, 3), int)
        # fmt: off
        x[0] = cp.array([[1, 0, 0],
                         [1, 0, 0],
                         [0, 0, 0]])
        x[1] = cp.array([[0, 0, 0],
                         [0, 1, 5],
                         [0, 0, 0]])

        lnb = x.copy()
        lnb[0] = cp.array([[1, 2, 2],
                           [1, 2, 2],
                           [2, 2, 2]])
        lnb[1] = cp.array([[2, 2, 2],
                           [2, 1, 3],
                           [2, 2, 2]])
        lb = x.copy()
        lb[0] = cp.array([[1,  BG, BG],  # noqa
                          [1,  BG, BG],  # noqa
                          [BG, BG, BG]])
        lb[1] = cp.array([[BG, BG, BG],
                          [BG, 1,   2],  # noqa
                          [BG, BG, BG]])
        # fmt: on
        assert_array_equal(label(x), lb)
        assert_array_equal(label(x, background=-1), lnb)

    def test_background_two_regions(self):
        x = cp.zeros((2, 3, 3), int)
        # fmt: off
        x[0] = cp.array([[0, 0, 6],
                         [0, 0, 6],
                         [5, 5, 5]])
        x[1] = cp.array([[6, 6, 0],
                         [5, 0, 0],
                         [0, 0, 0]])
        lb = x.copy()
        lb[0] = cp.array([[BG, BG, 1],
                          [BG, BG, 1],
                          [2,  2,  2]])  # noqa
        lb[1] = cp.array([[1,  1,  BG],  # noqa
                          [2,  BG, BG],  # noqa
                          [BG, BG, BG]])
        # fmt: on
        res = label(x, background=0)
        assert_array_equal(res, lb)

    def test_background_one_region_center(self):
        x = cp.zeros((3, 3, 3), int)
        x[1, 1, 1] = 1

        lb = cp.ones_like(x) * BG
        lb[1, 1, 1] = 1

        assert_array_equal(label(x, connectivity=1, background=0), lb)

    def test_return_num(self):
        # fmt: off
        x = cp.array([[1, 0, 6],
                      [0, 0, 6],
                      [5, 5, 5]])
        # fmt: on
        assert_array_equal(label(x, return_num=True)[1], 3)
        assert_array_equal(label(x, background=-1, return_num=True)[1], 4)

    def test_1D(self):
        x = cp.array((0, 1, 2, 2, 1, 1, 0, 0))
        xlen = len(x)
        y = cp.array((0, 1, 2, 2, 3, 3, 0, 0))
        reshapes = ((xlen,),
                    (1, xlen), (xlen, 1),
                    (1, xlen, 1), (xlen, 1, 1), (1, 1, xlen))
        for reshape in reshapes:
            x2 = x.reshape(reshape)
            labelled = label(x2)
            assert_array_equal(y, labelled.flatten())

# CuPy Backend: unlike scikit-image, the CUDA implementation is nD
#    def test_nd(self):
#        x = cp.ones((1, 2, 3, 4))
#        with testing.raises(NotImplementedError):
#            label(x)


# @pytest.mark.skip("ccomp not yet implemented")
# class TestSupport:
#     def test_reshape(self):
#         shapes_in = ((3, 1, 2), (1, 4, 5), (3, 1, 1), (2, 1), (1,))
#         for shape in shapes_in:
#             shape = np.array(shape)
#             numones = sum(shape == 1)
#             inp = np.random.random(shape)
#             inp = cp.asarray(inp)

#             fixed, swaps = ccomp.reshape_array(inp)
#             shape2 = fixed.shape
#             # now check that all ones are at the beginning
#             for i in range(numones):
#                 assert shape2[i] == 1

#             back = ccomp.undo_reshape_array(fixed, swaps)
#             # check that the undo works as expected
#             assert_array_equal(inp, back)
