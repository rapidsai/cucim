# Security Policy

cuCIM is a GPU-accelerated computer-vision and image-processing library for
N-dimensional images, with a focus on biomedical whole-slide imaging (WSI)
and other large microscopy / geospatial / remote-sensing formats. It is a
library, not a service: it runs in-process inside a Python interpreter or a
native C++ application, and inherits the caller's privilege.

Because cuCIM's role is to ingest large, format-rich image files —
multi-resolution TIFF (SVS, Philips), JPEG, JPEG2000, LZW/Deflate streams —
the file-format readers and the bundled native dependencies they call into
are the dominant attack surface. The threat model below reflects that.

## Reporting a Vulnerability

Please report security vulnerabilities privately through one of the channels
below. **Do not open a public GitHub issue, PR, or discussion** for a
suspected vulnerability.

1. **NVIDIA Vulnerability Disclosure Program (preferred)**
   <https://www.nvidia.com/en-us/security/>
   Submit through the NVIDIA PSIRT web form. This is the fastest path to
   triage and tracking.

2. **Email NVIDIA PSIRT**
   psirt@nvidia.com — encrypt sensitive reports with the
   [NVIDIA PSIRT PGP key](https://www.nvidia.com/en-us/security/pgp-key).

3. **GitHub Private Vulnerability Reporting**
   Use the **Security** tab on this repository → *Report a vulnerability*.

Please include, where possible:

- Affected component (e.g. the `cuslide` plugin, the JPEG2000 decoder,
  the shared-memory image cache, the GDS file handle, the pybind11 surface)
- cuCIM version, CUDA version, GPU model, and OS
- Reproduction steps and a minimal proof-of-concept image (sanitize PHI
  before sending — synthetic samples are preferred)
- Impact assessment (memory corruption, code execution, DoS, info disclosure)
- Any relevant CWE / CVE identifiers

NVIDIA PSIRT will acknowledge receipt and coordinate triage, fix
development, and coordinated disclosure. More on NVIDIA's response
process: <https://www.nvidia.com/en-us/security/psirt-policies/>.

## Security Architecture & Context

**Classification:** Library (C++ core with Python bindings via pybind11; a
plugin architecture loads format-specific decoders at runtime).

**Primary security responsibility:** Safely ingest large, attacker-shaped
image files — TIFF / SVS / Philips TIFF, with embedded JPEG, JPEG2000, LZW,
and Deflate streams — and produce GPU-resident tiles for downstream
processing without crashing the host process, corrupting memory, or
executing attacker-supplied code.

**Components and trust boundaries:**

- **C++ core** (`cpp/src/`) — image-cache management (`cache/`), codec
  dispatch (`codec/`), concurrent primitives (`concurrent/`), in-process
  configuration (`config/`), filesystem and GDS integration
  (`filesystem/` — `cufile_driver.cpp`, `file_handle.cpp`), runtime
  plugin loader (`plugin/`).
- **Format plugins** (`cpp/plugins/`) — `cucim.kit.cuslide` and
  `cucim.kit.cuslide2` for slide formats (SVS, Philips TIFF, generic
  multi-resolution TIFF), `cucim.kit.cumed` for medical formats. Loaded
  dynamically by the runtime plugin loader.
- **Bundled native dependencies** (`3rdparty/`) — cuCIM links against or
  vendors libtiff, libopenjpeg, libjpeg-turbo, libspng, libdeflate,
  pugixml, openslide, abseil, boost, folly, fmt, libcuckoo, pybind11,
  taskflow. Vulnerabilities in any of these are reachable from cuCIM
  inputs.
- **GDS / cufile** (`gds/`, `cpp/src/filesystem/cufile_driver.cpp`) —
  GPU Direct Storage paths that read image bytes directly into GPU
  memory. Bypasses the host page cache.
- **Shared-memory image cache** (`cpp/src/cache/image_cache_shared_memory.{h,cpp}`)
  — POSIX shared memory used to share decoded tiles between processes
  (e.g. multi-worker dataloaders).
- **pybind11 surface** (`python/cucim/`, `python/pybind11_*.cpp`) —
  exposes C++ objects, pointers, and arrays to Python. Type conversions
  here have historically allowed Python-side construction of pointers
  consumed by the C++ layer.
- **Python API** (`python/cucim/`) — Pythonic image-processing API
  matching scikit-image and OpenSlide; convenience layer over the C++
  plugins.

**Out of scope for this policy:** vulnerabilities in CUDA, the NVIDIA
driver, GDS / cufile itself, cupy, scikit-image, scipy, numpy, or the
upstream image-codec libraries cuCIM bundles (libtiff, libopenjpeg,
libjpeg-turbo, libspng, libdeflate). Vulnerabilities in cuCIM's *integration*
with those libraries — wrappers, type conversions, error handling, lifetime
management — are in scope.

## Threat Model

The threats below trace to specific components in this repository. Several
have already been observed and remediated through the
[RAPIDS Security Audit](https://github.com/orgs/rapidsai/projects/207); they
are listed so that callers and integrators understand the classes of bugs
the library defends against.

1. **TIFF / WSI parser memory corruption.**
   The slide-format plugins (`cpp/plugins/cucim.kit.cuslide*`) parse
   complex multi-resolution TIFF structures with vendor-specific
   metadata. Historical findings include integer overflow in tile / image
   buffer size calculations, heap buffer overflow in the tile
   decompression dispatch from unvalidated offsets and sizes, and heap
   OOB reads / uncontrolled recursion in vendor metadata parsing.
   A hostile `.svs`, `.tif`, or `.tiff` file is the canonical exploit
   vector.

2. **JPEG2000 decoder memory corruption.**
   The bundled libopenjpeg path historically produced a heap buffer
   overflow in color conversion when decoded image dimensions exceed
   tile dimensions, plus integer-overflow and uninitialized-memory
   issues in stream callbacks, plus memory leaks on error paths. A
   hostile JPEG2000 stream embedded inside a TIFF tile or supplied
   directly drives these.

3. **pybind11 type-conversion as a pointer-construction primitive.**
   Some pybind11 bindings accept Python integers and convert them into
   C++ pointers without validating that the integer references a live,
   owned, type-correct object. A hostile Python caller can pass an
   arbitrary integer and trigger a wild dereference inside the C++
   layer.

4. **Shared-memory image cache: world-accessible IPC with predictable name.**
   `image_cache_shared_memory.cpp` creates a POSIX shared-memory segment
   with a predictable name and default permissions. A co-tenant on the
   same host can attach to the segment, read decoded tile contents
   (potentially containing PHI), or corrupt them and cause downstream
   crashes in cuCIM workers.

5. **Plugin loading.**
   The runtime plugin loader (`cpp/src/plugin/`) loads format plugins
   from disk and executes their initialization. A writeable plugin
   directory or a `LD_LIBRARY_PATH` / `RPATH` an attacker controls
   yields arbitrary code execution at plugin-load time.

6. **GDS / cufile direct-to-GPU reads.**
   The GDS path (`cpp/src/filesystem/cufile_driver.cpp`,
   `file_handle.cpp`) writes file bytes into GPU memory without
   intermediate host validation. Length / offset bugs in this path
   produce GPU-side OOB writes that are harder to observe than
   host-side bugs.

7. **Decompression bombs (DoS).**
   The Deflate, LZW, and JPEG2000 streams cuCIM decodes do not have an
   output-size ceiling enforced by cuCIM's dispatch layer. A small TIFF
   tile holding a high-ratio compressed payload can expand to exhaust
   host or GPU memory.

8. **Upstream codec CVEs.**
   Vulnerabilities in the bundled / linked native codecs (libtiff,
   libopenjpeg, libjpeg-turbo, libspng, libdeflate) are reachable from
   any caller-supplied image. Dependency-update lag is itself a
   security risk for cuCIM specifically because of how broad and
   frequently-CVE'd these codecs are.

## Critical Security Assumptions

cuCIM is a library and inherits the caller's privilege; the following are
assumed of the caller / deployer.

- **Inputs may be hostile, but the caller decides whether to trust them.**
  cuCIM endeavors to fail safely on malformed image inputs, but callers
  parsing files from untrusted sources (web uploads, multi-tenant
  pipelines) should run cuCIM in a process with the smallest viable
  blast radius — separate process, container, cgroup memory limits, no
  network egress, no PHI on the host beyond what the call needs.

- **Resource limits are imposed externally.**
  cuCIM does not cap memory or time per decode. WSI files routinely
  decode to tens of gigabytes; hostile compression ratios can push
  this further. Callers should impose memory and time limits at the
  process / container boundary.

- **The plugin directory and library load path are trusted.**
  The plugin loader executes code from any shared library it finds at
  startup. The plugin directory and `LD_LIBRARY_PATH` / `RPATH` /
  `LD_PRELOAD` must be writable only by trusted principals.

- **The shared-memory cache is not a confidentiality boundary.**
  When cuCIM's shared-memory image cache is enabled, decoded tiles
  (which may contain PHI in clinical pipelines) are accessible to any
  process on the host that can guess or enumerate the segment name.
  Disable the shared-memory cache on multi-tenant hosts, or restrict it
  via the host's IPC namespace / user separation.

- **Python callers respect pybind11 type contracts.**
  pybind11 conversions trust the Python-side type. A caller that passes
  raw integers where pointers or typed handles are expected can drive
  arbitrary dereferences in the C++ layer. Treat the cuCIM Python API
  as an in-process FFI, not a sandbox.

- **GDS / cufile inputs are trusted.**
  GDS bypasses the host page cache, so length / offset mistakes manifest
  as GPU-side OOB writes that are harder to attribute. Only use GDS
  with file sources whose contents are trusted.

- **GPU memory is not a confidentiality boundary.**
  Multiple processes sharing a GPU may observe each other's GPU memory
  through driver-level side channels. cuCIM assumes the caller has
  provisioned the GPU appropriately (MIG, exclusive process, container
  isolation) when confidentiality matters — especially relevant for
  PHI-bearing biomedical pipelines.

- **Upstream codec security is the caller's problem too.**
  cuCIM bundles or links libtiff, libopenjpeg, libjpeg-turbo, libspng,
  libdeflate, and others. CVE-driven updates to these libraries may
  require rebuilding cuCIM. Callers operating on untrusted images
  should track upstream advisories for these libraries and avoid
  long-pinned cuCIM builds on outdated codec versions.

## Supported Versions

Security fixes are issued against the current release line published on the
[RAPIDS release schedule](https://docs.rapids.ai/releases/). Older minor
releases are generally not backported; upgrade to the latest supported
version to receive fixes — and to pick up codec-dependency updates.

## Dependency Security

cuCIM vendors or links a long list of native dependencies, several of which
have an active CVE history: libtiff, libopenjpeg, libjpeg-turbo, libspng,
libdeflate, openslide, abseil, boost, folly, pybind11. Dependency updates
ship with regular releases; high-severity upstream CVEs in any of those
libraries may trigger out-of-band patch releases. The full third-party
inventory is enumerated in [`LICENSE-3rdparty.md`](LICENSE-3rdparty.md).
