Example scripts used for benchmarking reads of uncompressed TIFF images using
existing CPU-based tools (openslide-python, tifffile) as well as reads
accelerated using `kvikio` with GPUDirect Storage enabled or disabled.

### GPUDirect setup/install

GPUDirect Storage (GDS) is only supported on certain linux systems

(e.g. at the time of writing:
 Ubuntu 18.04, Ubuntu 20.04, RHEL 8.3, RHEL 8.4, DGX workstations)

Also, not all CUDA-capable hardware supports GDS. For example, gaming-focused
GPUs are currently not GDS-enabled. GDS is supported on T4, V100, A100 and
RTX A6000 GPUs, for example:
https://docs.nvidia.com/gpudirect-storage/troubleshooting-guide/index.html#api-error-2

GPUDirect must be installed as described in
https://docs.nvidia.com/gpudirect-storage/troubleshooting-guide/index.html

As part of this, the Mellanox OFED drivers (MLNX_OFED) must be installed:
https://docs.nvidia.com/gpudirect-storage/troubleshooting-guide/index.html#mofed-req-install
https://network.nvidia.com/support/mlnx-ofed-matrix/


### Obtaining test data

Due to the large data size needed for benchmarking, no data is included in this
repository. Many suitable images are publicly available. Assume we download a single image named `TUPAC-TR-467.svs` which is available in the training data from this challenge:

https://tupac.grand-challenge.org/Dataset/

which corresponds to the following publication:

Veta, Mitko, et al. "Predicting breast tumor proliferation from whole-slide images: the TUPAC16 challenge." Medical image analysis 54 (2019): 111-121.

Because the demo only supports uncompressed data, it is necessary to first convert the data a raw (uncompressed) TIFF image. This can be done using cuCIM >= 2022.12.00 via the following command line call:
```sh
cucim convert --tile-size 512 --overlap 0 --num-workers 12  --output-filename resize.tiff --compression RAW TUPAC-TR-467.svs
```

The scripts have the filename `resize.tiff` hardcoded, so you will have to modify the file name in near the top of the scripts if a different image is to be used.


### Summary of demo files

- **benchmark_read.py** : Benchmark reads of a full image at the specified
resolution level from an uncompressed multi-resolution TIFF image.

- **benchmark_round_trip.py** : Read from uncompressed TIFF while writing to an
uncompressed Zarr file with a tile size matching the TIFF image.

- **benchmark_zarr_write.py** : Benchmark writing of a CuPy array to an
uncompressed Zarr file of the specified chunk size. 

- **benchmark_zarr_write_lz4_via_dask.py** : Use Dask and
`kvikio.zarr.GDSStore` to write LZ4 lossless compressed Zarr array with the
specified storage level.

- **lz4_nvcomp.py** : this is a LZ4 compressor for use with
`kvikio.zarr.GDSStore`

- **demo_implementation.py** : Implementations of tiled read/write that are
used by the benchmarking scripts described above.


### some commands from GDSDirect guide for checking system status

Below are some useful diagnostic commands extracted from the
[GPUDirect Storage docs](https://docs.nvidia.com/gpudirect-storage/index.html).

To check MLNX_OFED version
```sh
ofed_info -s
```

To check GDS status run:
```sh
/usr/local/cuda/gds/tools/gdscheck -p
```
want to see at least NVMe supported in the driver configuration, e.g.
```
 =====================
 DRIVER CONFIGURATION:
 =====================
 NVMe               : Supported
```
and under the GPU INFO, the device should be listed as "supports GDS", e.g.
```
 =========
 GPU INFO:
 =========
 GPU index 0 NVIDIA RTX A6000 bar:1 bar size (MiB):256 supports GDS, IOMMU State: Disable
 ```

Check nvidia-fs
```sh
cat /proc/driver/nvidia-fs/stats
```
e.g.
```
GDS Version: 1.4.0.29 
NVFS statistics(ver: 4.0)
NVFS Driver(version: 2.13.5)
Mellanox PeerDirect Supported: False
IO stats: Disabled, peer IO stats: Disabled
Logging level: info
```
