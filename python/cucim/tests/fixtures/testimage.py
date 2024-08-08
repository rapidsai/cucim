#
# Copyright (c) 2021-2024, NVIDIA CORPORATION.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import shutil

import pytest
from pytest_lazy_fixtures import lf as lazy_fixture

from ..util.gen_image import ImageGenerator


def gen_image(tmpdir_factory, recipe, resolution=None):
    dataset_path = tmpdir_factory.mktemp("datasets").strpath
    dataset_gen = ImageGenerator(dataset_path, [recipe], [resolution])
    image_path = dataset_gen.gen()
    return (dataset_path, image_path[0])


# tiff_stripe_32x24_16
@pytest.fixture(scope="session")
def testimg_tiff_stripe_32x24_16_jpeg(tmpdir_factory):
    dataset_path, image_path = gen_image(tmpdir_factory, "tiff")
    yield image_path
    # Clean up fake dataset folder
    shutil.rmtree(dataset_path)


@pytest.fixture(scope="session")
def testimg_tiff_stripe_32x24_16_deflate(tmpdir_factory):
    dataset_path, image_path = gen_image(
        tmpdir_factory, "tiff::stripe:32x24:16:deflate"
    )
    yield image_path
    # Clean up fake dataset folder
    shutil.rmtree(dataset_path)


@pytest.fixture(scope="session")
def testimg_tiff_stripe_32x24_16_raw(tmpdir_factory):
    dataset_path, image_path = gen_image(
        tmpdir_factory, "tiff::stripe:32x24:16:raw"
    )
    yield image_path
    # Clean up fake dataset folder
    shutil.rmtree(dataset_path)


@pytest.fixture(
    scope="session",
    params=[
        lazy_fixture("testimg_tiff_stripe_32x24_16_jpeg"),
        lazy_fixture("testimg_tiff_stripe_32x24_16_deflate"),
        lazy_fixture("testimg_tiff_stripe_32x24_16_raw"),
    ],
)
def testimg_tiff_stripe_32x24_16(request):
    return request.param


# tiff_stripe_4096x4096_256
@pytest.fixture(scope="session")
def testimg_tiff_stripe_4096x4096_256_jpeg(tmpdir_factory):
    dataset_path, image_path = gen_image(
        tmpdir_factory, "tiff::stripe:4096x4096:256:jpeg"
    )
    yield image_path
    # Clean up fake dataset folder
    shutil.rmtree(dataset_path)


@pytest.fixture(scope="session")
def testimg_tiff_stripe_4096x4096_256_deflate(tmpdir_factory):
    dataset_path, image_path = gen_image(
        tmpdir_factory, "tiff::stripe:4096x4096:256:deflate"
    )
    yield image_path
    # Clean up fake dataset folder
    shutil.rmtree(dataset_path)


@pytest.fixture(scope="session")
def testimg_tiff_stripe_4096x4096_256_raw(tmpdir_factory):
    dataset_path, image_path = gen_image(
        tmpdir_factory, "tiff::stripe:4096x4096:256:raw"
    )
    yield image_path
    # Clean up fake dataset folder
    shutil.rmtree(dataset_path)


@pytest.fixture(
    scope="session",
    params=[
        lazy_fixture("testimg_tiff_stripe_4096x4096_256_jpeg"),
        lazy_fixture("testimg_tiff_stripe_4096x4096_256_deflate"),
        lazy_fixture("testimg_tiff_stripe_4096x4096_256_raw"),
    ],
)
def testimg_tiff_stripe_4096x4096_256(request):
    return request.param


# tiff_stripe_100000x100000_256
@pytest.fixture(scope="session")
def testimg_tiff_stripe_100000x100000_256_jpeg(tmpdir_factory):
    dataset_path, image_path = gen_image(
        tmpdir_factory, "tiff::stripe:100000x100000:256:jpeg"
    )
    yield image_path
    # Clean up fake dataset folder
    shutil.rmtree(dataset_path)


@pytest.fixture(scope="session")
def testimg_tiff_stripe_100000x100000_256_deflate(tmpdir_factory):
    dataset_path, image_path = gen_image(
        tmpdir_factory, "tiff::stripe:100000x100000:256:deflate"
    )
    yield image_path
    # Clean up fake dataset folder
    shutil.rmtree(dataset_path)


@pytest.fixture(scope="session")
def testimg_tiff_stripe_100000x100000_256_raw(tmpdir_factory):
    dataset_path, image_path = gen_image(
        tmpdir_factory, "tiff::stripe:100000x100000:256:raw"
    )
    yield image_path
    # Clean up fake dataset folder
    shutil.rmtree(dataset_path)


@pytest.fixture(
    scope="session",
    params=[
        lazy_fixture("testimg_tiff_stripe_100000x100000_256_jpeg"),
        lazy_fixture("testimg_tiff_stripe_100000x100000_256_deflate"),
        lazy_fixture("testimg_tiff_stripe_100000x100000_256_raw"),
    ],
)
def testimg_tiff_stripe_100000x100000_256(request):
    return request.param


# testimg_tiff_stripe_4096_4096_256_jpeg_resolution
@pytest.fixture(scope="session")
def testimg_tiff_stripe_4096_4096_256_jpeg_resolution_3_5_centimeter(
    tmpdir_factory,
):
    resolution = (0.3, 0.5, "CENTIMETER")
    dataset_path, image_path = gen_image(
        tmpdir_factory, "tiff::stripe:4096x4096:256:jpeg", resolution
    )
    yield image_path, resolution
    # Clean up fake dataset folder
    shutil.rmtree(dataset_path)


@pytest.fixture(scope="session")
def testimg_tiff_stripe_4096_4096_256_jpeg_resolution_4_7_inch(tmpdir_factory):
    resolution = (0.4, 0.7, "INCH")
    dataset_path, image_path = gen_image(
        tmpdir_factory, "tiff::stripe:4096x4096:256:jpeg", resolution
    )
    yield image_path, resolution
    # Clean up fake dataset folder
    shutil.rmtree(dataset_path)


@pytest.fixture(scope="session")
def testimg_tiff_stripe_4096_4096_256_jpeg_resolution_9_1_none(tmpdir_factory):
    resolution = (9, 1, "NONE")
    dataset_path, image_path = gen_image(
        tmpdir_factory, "tiff::stripe:4096x4096:256:jpeg", resolution
    )
    yield image_path, resolution
    # Clean up fake dataset folder
    shutil.rmtree(dataset_path)


@pytest.fixture(
    scope="session",
    params=[
        lazy_fixture(
            "testimg_tiff_stripe_4096_4096_256_jpeg_resolution_3_5_centimeter"
        ),
        lazy_fixture(
            "testimg_tiff_stripe_4096_4096_256_jpeg_resolution_4_7_inch"
        ),
        lazy_fixture(
            "testimg_tiff_stripe_4096_4096_256_jpeg_resolution_9_1_none"
        ),
    ],
)
def testimg_tiff_stripe_4096_4096_256_jpeg_resolution(request):
    return request.param
