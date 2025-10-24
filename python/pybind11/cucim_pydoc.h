/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2021, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef PYCUCIM_CUCIM_PYDOC_H
#define PYCUCIM_CUCIM_PYDOC_H

#include <string>

#include "macros.h"

namespace cucim::doc
{

namespace DLDataType
{

//  Constructor
PYDOC(DLDataType, R"doc(
Constructor for `DLDataType`.
)doc")

//  uint8_t code;
PYDOC(code, R"doc(
Type code of base types.
)doc")

//  uint8_t bits;
PYDOC(bits, R"doc(
Number of bits, common choices are 8, 16, 32.
)doc")

//  uint16_t lanes;
PYDOC(lanes, R"doc(
Number of lanes in the type, used for vector types.
)doc")

} // namespace DLDataType


namespace CuImage
{

// CuImage(const filesystem::Path& path);
PYDOC(CuImage, R"doc(
Constructor of CuImage.
)doc")

// CuImage(const filesystem::Path& path, const std::string& plugin_name);
// CuImage(const CuImage& cuimg) = delete;
// CuImage(CuImage&& cuimg);
//
// ~CuImage();

// std::shared_ptr<cache::ImageCache> CuImage::cache()
PYDOC(cache, R"doc(
Get cache object.
)doc")

// std::shared_ptr<profiler::Profiler> CuImage::profiler()
PYDOC(profiler, R"doc(
Get profiler object.
)doc")

// bool CuImage::is_trace_enabled()
PYDOC(is_trace_enabled, R"doc(
Return whether if the tracing is enabled or not.
)doc")

// filesystem::Path path() const;
PYDOC(path, R"doc(
Underlying file path for this object.
)doc")

// bool is_loaded() const;
PYDOC(is_loaded, R"doc(
True if image data is loaded & available.
)doc")

// io::Device device() const;
PYDOC(device, R"doc(
A device type.

By default t is `cpu` (It will be changed since v0.19.0).
)doc")

// Metadata metadata() const;
PYDOC(raw_metadata, R"doc(
A raw metadata string.
)doc")

// Metadata metadata() const;
PYDOC(metadata, R"doc(
A metadata object as `dict`.

It would be a dictionary(key-value pair) in general but can be a complex object (e.g., OME-TIFF metadata).
)doc")

// uint16_t ndim() const;
PYDOC(ndim, R"doc(
The number of dimensions.
)doc")

// std::string dims() const;
PYDOC(dims, R"doc(
A string containing a list of dimensions being requested.

The default is to return the six standard dims ('STCZYX') unless it is a DP multi-resolution image.
  [sites, time, channel(or wavelength), z, y, x]. S - Sites or multiposition locations.

NOTE: in OME-TIFF's metadata, dimension order would be specified as 'XYZCTS' (first one is fast-iterating dimension).
)doc")

// Shape shape() const;
PYDOC(shape, R"doc(
A tuple of dimension sizes (in the order of `dims`)
)doc")

// std::vector<int64_t> size(std::string dim_order) const;
PYDOC(size, R"doc(
Returns size as a tuple for the given dimension order.
)doc")

// DLDataType dtype() const;
PYDOC(dtype, R"doc(
The data type of the image.
)doc")

// DLDataType typestr() const;
PYDOC(typestr, R"doc(
The data type of the image in string format.

The value can be converted to NumPy's dtype using `numpy.dtype()`.
(e.g., `numpy.dtype(img.typestr)`).
)doc")

// std::vector<std::string> channel_names() const;
PYDOC(channel_names, R"doc(
A channel name list.
)doc")

// std::vector<float> spacing(std::string dim_order = std::string{}) const;
PYDOC(spacing, R"doc(
Returns physical size in tuple.

If `dim_order` is specified, it returns physical size for the dimensions.
If a dimension given by the `dim_order` doesn't exist, it returns 1.0 by default for the missing dimension.

Args:
  dim_order: A dimension string (e.g., 'XYZ')

Returns:
  A tuple with physical size for each dimension
)doc")

// std::vector<std::string> spacing_units(std::string dim_order = std::string{}) const;
PYDOC(spacing_units, R"doc(
Units for each spacing element (size is same with `ndim`).
)doc")

// std::array<float, 3> origin() const;
PYDOC(origin, R"doc(
Physical location of (0, 0, 0) (size is always 3).
)doc")

// std::array<std::array<float, 3>, 3> direction() const;
PYDOC(direction, R"doc(
Direction cosines (size is always 3x3).
)doc")

// std::string coord_sys() const;
PYDOC(coord_sys, R"doc(
Coordinate frame in which the direction cosines are measured.

Available Coordinate frame names are not finalized yet.
)doc") //  (either `LPS`(ITK/DICOM) or `RAS`(NIfTI/3D Slicer)). (either `LPS`(ITK/DICOM) or `RAS`(NIfTI/3D Slicer)).

// ResolutionInfo resolutions() const;
PYDOC(resolutions, R"doc(
Returns a dict that includes resolution information.

- level_count: The number of levels
- level_dimensions: A tuple of dimension tuples (width, height)
- level_downsamples: A tuple of down-sample factors
- level_tile_sizes: A tuple of tile size tuple (tile width, tile_height)
)doc")

// dlpack::DLTContainer container() const;

// CuImage read_region(std::vector<int64_t> location,
//                     std::vector<int64_t> size,
//                     uint16_t level=0,
//                     DimIndices region_dim_indices={},
//                     io::Device device="cpu",
//                     DLTensor* buf=nullptr,
//                     std::string shm_name="");
PYDOC(read_region, R"doc(
Returns a subresolution image.

- `location` and `size`'s dimension order is reverse of image's dimension order.
- Need to specify (X,Y) and (Width, Height) instead of (Y,X) and (Height, Width).
- If location is not specified, location would be (0, 0) if Z=0. Otherwise, location would be (0, 0, 0)
- Like OpenSlide, location is level-0 based coordinates (using the level-0 reference frame)
- If `size` is not specified, size would be (width, height) of the image at the specified `level`.
- `<not supported yet>` Additional parameters (S,T,C,Z) are similar to <https://allencellmodeling.github.io/aicsimageio/aicsimageio.html#aicsimageio.aics_image.AICSImage.get_image_data>
- Do not yet support indices/ranges for (S,T,C,Z).
- Default value for level, S, T, Z are zero.
- Default value for C is -1 (whole channels)
- `<not supported yet>` `device` could be one of the following strings or Device object: e.g., `'cpu'`, `'cuda'`, `'cuda:0'` (use index 0), `cucim.clara.io.Device(cucim.clara.io.CUDA,0)`.
- `<not supported yet>` If `buf` is specified (buf's type can be either numpy object that implements `__array_interface__`, or cupy-compatible object that implements `__cuda_array_interface__`), the read image would be saved into buf object without creating CPU/GPU memory.
- `<not supported yet>` If `shm_name` is specified, shared memory would be created and data would be read in the shared memory.

)doc")

// std::set<std::string> associated_images() const;
PYDOC(associated_images, R"doc(
Returns a set of associated image names.

Digital Pathology image usually has a label/thumbnail or a macro image(low-power snapshot of the entire glass slide).
Names of those images (such as 'macro' and 'label') are in `associated_images`.
)doc")

// CuImage associated_image(const std::string& name) const;
PYDOC(associated_image, R"doc(
Returns an associated image for the given name, as a CuImage object.
)doc")

// void save(std::string file_path) const;
PYDOC(save, R"doc(
Saves image data to the file path.

Currently it supports only .ppm file format that can be viewed by `eog` command in Ubuntu.
)doc")

// void close();
PYDOC(close, R"doc(
Closes the file handle.

Once the file handle is closed, the image object (if loaded before) still exists but cannot read additional images
from the file.
)doc")


// void _set_array_interface(const CuImage& cuimg);
PYDOC(_set_array_interface, R"doc(
Add `__array_interface__` or `__cuda_array_interface__` depending on the memory type.

Args:
  cuimg: CuImage object

Returns:
  None
)doc")

}; // namespace CuImage

namespace CuImageIterator
{

// CuImageIterator(std::shared_ptr<DataType> cuimg, bool ending = false);
PYDOC(CuImageIterator, R"doc(
Constructor of CuImageIterator.
)doc")

} // namespace CuImageIterator

} // namespace cucim::doc

#endif // PYCUCIM_CUCIM_PYDOC_H
