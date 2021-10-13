#
# Copyright (c) 2021, NVIDIA CORPORATION.
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

import argparse
import logging
import os

try:
    from .gen_tiff import TiffGenerator
except ImportError:
    from gen_tiff import TiffGenerator

GENERATOR_MAP = {
    'tiff': TiffGenerator()
}


class ImageGenerator:
    def __init__(self, dest, recipes, logger=None):
        self.logger = logger or logging.getLogger(__name__)
        self.dest = dest
        self.recipes = recipes

    def gen(self):

        results = []

        for recipe in self.recipes:
            items = recipe.split(':')
            item_len = len(items)
            if not (1 <= item_len <= 6):
                raise RuntimeError(
                    'Value should be '
                    + 'type[:subpath:pattern:image_size:tile_size:compression]'
                    + ' format')

            kind = items[0]
            subpath = '' if item_len == 1 else items[1]
            pattern = 'stripe' if item_len <= 2 else items[2]
            image_size_str = '32x24' if item_len <= 3 else items[3]
            image_size = list(map(lambda x: int(x), image_size_str.split('x')))
            tile_size = 16 if item_len <= 4 else int(items[4])
            compression = 'jpeg' if item_len <= 5 else items[5]

            dest_folder = os.path.join(self.dest, subpath)
            os.makedirs(dest_folder, exist_ok=True)

            generator_obj = GENERATOR_MAP.get(kind)
            if generator_obj is None:
                raise RuntimeError(
                    "There is no generator for '{}'".format(kind))

            image_data = generator_obj.get_image(
                pattern=pattern, image_size=image_size)

            if image_data is None:
                raise RuntimeError(
                    f'No data generated from [pattern={pattern},'
                    + f' image_size={image_size}, tile_size={tile_size},'
                    + f' compression={compression}].')

            file_name = f'{kind}_{pattern}_{image_size_str}_{tile_size}'

            image_path = generator_obj.save_image(image_data,
                                                  dest_folder,
                                                  file_name=file_name,
                                                  kind=kind,
                                                  subpath=subpath,
                                                  pattern=pattern,
                                                  image_size=image_size,
                                                  tile_size=tile_size,
                                                  compression=compression)
            self.logger.info('  Generated %s...', image_path)
            results.append(image_path)

        self.logger.info('[Finished] Dataset generation')
        return results


def main():
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(
        description='Generate Image Data')
    parser.add_argument(
        'recipes',
        metavar='type[:subpath:pattern:image_size:tile_size:compression]',
        default=['tiff::stripe:32x24:16'], nargs='+',
        help='data set type with pattern to write '
             + '(default: tiff::stripe:32x24:16:jpeg')

    parser.add_argument('--dest', '-d', default='.', help='destination folder')
    args = parser.parse_args()
    generator = ImageGenerator(args.dest, args.recipes)
    generator.gen()


if __name__ == '__main__':
    main()
