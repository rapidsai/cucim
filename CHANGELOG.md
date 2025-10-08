# cucim 25.10.00 (8 Oct 2025)

### 🐛 Bug Fixes
* Update fmt library to version 11.2.0 in CMake configuration by [@gigony in https://github.com/rapidsai/cucim/pull/917](https://github.com/gigony in https://github.com/rapidsai/cucim/pull/917)
* Fix indentation by [@jakirkham in https://github.com/rapidsai/cucim/pull/931](https://github.com/jakirkham in https://github.com/rapidsai/cucim/pull/931)
* Relax `libcufile` dependency for old CTK 12 on ARM by [@jakirkham in https://github.com/rapidsai/cucim/pull/930](https://github.com/jakirkham in https://github.com/rapidsai/cucim/pull/930)
* Add `pip` to `cucim`'s `requirements/host` by [@jakirkham in https://github.com/rapidsai/cucim/pull/941](https://github.com/jakirkham in https://github.com/rapidsai/cucim/pull/941)
### 📖 Documentation
* update package installation commands in notebooks by [@grlee77 in https://github.com/rapidsai/cucim/pull/927](https://github.com/grlee77 in https://github.com/rapidsai/cucim/pull/927)
### 🛠️ Improvements
* fix(docker): use versioned `-latest` tag for all `rapidsai` images by [@gforsyth in https://github.com/rapidsai/cucim/pull/908](https://github.com/gforsyth in https://github.com/rapidsai/cucim/pull/908)
* remove cuspatial references by [@jameslamb in https://github.com/rapidsai/cucim/pull/906](https://github.com/jameslamb in https://github.com/rapidsai/cucim/pull/906)
* run: fix compatibility with PEP-0632 for supporting Python >=3.12 by [@ElieDeBrauwer in https://github.com/rapidsai/cucim/pull/902](https://github.com/ElieDeBrauwer in https://github.com/rapidsai/cucim/pull/902)
* Enhance GPU memory leak test with unified memory check and increased thresholds by [@gigony in https://github.com/rapidsai/cucim/pull/898](https://github.com/gigony in https://github.com/rapidsai/cucim/pull/898)
* Drop CUDA 11 references from docs by [@jakirkham in https://github.com/rapidsai/cucim/pull/914](https://github.com/jakirkham in https://github.com/rapidsai/cucim/pull/914)
* add a copy of rapids-configure-conda-channels to the repo by [@jameslamb in https://github.com/rapidsai/cucim/pull/923](https://github.com/jameslamb in https://github.com/rapidsai/cucim/pull/923)
* Update rapids-build-backend to 0.4.1 by [@KyleFromNVIDIA in https://github.com/rapidsai/cucim/pull/922](https://github.com/KyleFromNVIDIA in https://github.com/rapidsai/cucim/pull/922)
* Remove CUDA 11 references by [@jakirkham in https://github.com/rapidsai/cucim/pull/905](https://github.com/jakirkham in https://github.com/rapidsai/cucim/pull/905)
* Build and test with CUDA 13.0.0, use GCC 14 for conda builds by [@jameslamb in https://github.com/rapidsai/cucim/pull/926](https://github.com/jameslamb in https://github.com/rapidsai/cucim/pull/926)
* Update rapids-dependency-file-generator by [@KyleFromNVIDIA in https://github.com/rapidsai/cucim/pull/934](https://github.com/KyleFromNVIDIA in https://github.com/rapidsai/cucim/pull/934)
* Use branch-25.10 again by [@jameslamb in https://github.com/rapidsai/cucim/pull/936](https://github.com/jameslamb in https://github.com/rapidsai/cucim/pull/936)
* Configure repo for automatic release notes generation by [@AyodeAwe in https://github.com/rapidsai/cucim/pull/939](https://github.com/AyodeAwe in https://github.com/rapidsai/cucim/pull/939)
* remove unused RMM config and docs, update pre-commit hooks by [@jameslamb in https://github.com/rapidsai/cucim/pull/946](https://github.com/jameslamb in https://github.com/rapidsai/cucim/pull/946)
* add montage and compare_images utility functions by [@grlee77 in https://github.com/rapidsai/cucim/pull/935](https://github.com/grlee77 in https://github.com/rapidsai/cucim/pull/935)

## New Contributors
* [@ElieDeBrauwer made their first contribution in https://github.com/rapidsai/cucim/pull/902](https://github.com/ElieDeBrauwer made their first contribution in https://github.com/rapidsai/cucim/pull/902)

**Full Changelog**: https://github.com/rapidsai/cucim/compare/v25.10.00a...branch-25.10

# cucim 25.08.00 (6 Aug 2025)

## 🚨 Breaking Changes

- Remove CUDA 11 from dependencies.yaml ([#887](https://github.com/rapidsai/cucim/pull/887)) [@KyleFromNVIDIA](https://github.com/KyleFromNVIDIA)
- stop uploading packages to downloads.rapids.ai ([#883](https://github.com/rapidsai/cucim/pull/883)) [@jameslamb](https://github.com/jameslamb)

## 🐛 Bug Fixes

- Drop unused variable `CUDA_MAJOR_VERSION` ([#907](https://github.com/rapidsai/cucim/pull/907)) [@jakirkham](https://github.com/jakirkham)
- Fix test failures due to deprecated Pillow Image.fromarray &#39;mode&#39; parameter ([#901](https://github.com/rapidsai/cucim/pull/901)) [@grlee77](https://github.com/grlee77)
- Adding GH_TOKEN pass-through to summarize job ([#891](https://github.com/rapidsai/cucim/pull/891)) [@msarahan](https://github.com/msarahan)

## 📖 Documentation

- add docs on CI workflow inputs ([#897](https://github.com/rapidsai/cucim/pull/897)) [@jameslamb](https://github.com/jameslamb)

## 🛠️ Improvements

- Drop CUDA 11 references from docs ([#914](https://github.com/rapidsai/cucim/pull/914)) [@jakirkham](https://github.com/jakirkham)
- fix(docker): use versioned `-latest` tag for all `rapidsai` images ([#908](https://github.com/rapidsai/cucim/pull/908)) [@gforsyth](https://github.com/gforsyth)
- remove cuspatial references ([#906](https://github.com/rapidsai/cucim/pull/906)) [@jameslamb](https://github.com/jameslamb)
- Drop CUDA 11 from CI Scripts ([#903](https://github.com/rapidsai/cucim/pull/903)) [@AyodeAwe](https://github.com/AyodeAwe)
- run: fix compatibility with PEP-0632 for supporting Python &gt;=3.12 ([#902](https://github.com/rapidsai/cucim/pull/902)) [@ElieDeBrauwer](https://github.com/ElieDeBrauwer)
- Enhance GPU memory leak test with unified memory check and increased thresholds ([#898](https://github.com/rapidsai/cucim/pull/898)) [@gigony](https://github.com/gigony)
- Use CUDA 12.9 in Conda, Devcontainers, Spark, GHA, etc. ([#894](https://github.com/rapidsai/cucim/pull/894)) [@jakirkham](https://github.com/jakirkham)
- Remove nvidia and dask channels ([#893](https://github.com/rapidsai/cucim/pull/893)) [@vyasr](https://github.com/vyasr)
- refactor(shellcheck): fix all shellcheck warnings/errors ([#890](https://github.com/rapidsai/cucim/pull/890)) [@gforsyth](https://github.com/gforsyth)
- refactor(conda): remove cuda11 conditionals from conda recipes ([#889](https://github.com/rapidsai/cucim/pull/889)) [@gforsyth](https://github.com/gforsyth)
- Remove CUDA 11 from dependencies.yaml ([#887](https://github.com/rapidsai/cucim/pull/887)) [@KyleFromNVIDIA](https://github.com/KyleFromNVIDIA)
- stop uploading packages to downloads.rapids.ai ([#883](https://github.com/rapidsai/cucim/pull/883)) [@jameslamb](https://github.com/jameslamb)
- Forward-merge branch-25.06 into branch-25.08 ([#880](https://github.com/rapidsai/cucim/pull/880)) [@gforsyth](https://github.com/gforsyth)
- Forward-merge branch-25.06 into branch-25.08 ([#876](https://github.com/rapidsai/cucim/pull/876)) [@gforsyth](https://github.com/gforsyth)

# cucim 25.06.00 (5 Jun 2025)

## 🛠️ Improvements

- use &#39;rapids-init-pip&#39; in wheel CI, other CI changes ([#881](https://github.com/rapidsai/cucim/pull/881)) [@jameslamb](https://github.com/jameslamb)
- Finish CUDA 12.9 migration and use branch-25.06 workflows ([#879](https://github.com/rapidsai/cucim/pull/879)) [@bdice](https://github.com/bdice)
- Build and test with CUDA 12.9.0 ([#877](https://github.com/rapidsai/cucim/pull/877)) [@bdice](https://github.com/bdice)
- Refactor: Enhance GPU Memory Leak Test for `read_region` ([#874](https://github.com/rapidsai/cucim/pull/874)) [@gigony](https://github.com/gigony)
- Add support for Python 3.13 ([#873](https://github.com/rapidsai/cucim/pull/873)) [@gforsyth](https://github.com/gforsyth)
- Download build artifacts from GitHub for CI ([#870](https://github.com/rapidsai/cucim/pull/870)) [@VenkateshJaya](https://github.com/VenkateshJaya)
- Add ARM conda environments ([#867](https://github.com/rapidsai/cucim/pull/867)) [@bdice](https://github.com/bdice)
- Moving wheel builds to specified location and uploading build artifacts to Github ([#854](https://github.com/rapidsai/cucim/pull/854)) [@VenkateshJaya](https://github.com/VenkateshJaya)
- Region Properties Performance Overhaul - Part 6: Public API (regionprops_table) ([#848](https://github.com/rapidsai/cucim/pull/848)) [@grlee77](https://github.com/grlee77)
- Region Properties Performance Overhaul - Part 5: Perimeter and Euler Characteristic ([#847](https://github.com/rapidsai/cucim/pull/847)) [@grlee77](https://github.com/grlee77)
- Region Properties Performance Overhaul - Part 4: Moment-Based Properties ([#846](https://github.com/rapidsai/cucim/pull/846)) [@grlee77](https://github.com/grlee77)
- Region Properties Performance Overhaul - Part 3: Convex Image Properties ([#845](https://github.com/rapidsai/cucim/pull/845)) [@grlee77](https://github.com/grlee77)
- Region Properties Performance Overhaul - Part 2: Intensity Image Properties ([#844](https://github.com/rapidsai/cucim/pull/844)) [@grlee77](https://github.com/grlee77)
- Region Properties Performance Overhaul - Part 1: Basic Properties ([#843](https://github.com/rapidsai/cucim/pull/843)) [@grlee77](https://github.com/grlee77)

# cucim 25.04.00 (9 Apr 2025)

## 🐛 Bug Fixes

- CuPy 13.4.1 compatibility: Fix dtype handling in fused chan-vese kernels ([#856](https://github.com/rapidsai/cucim/pull/856)) [@grlee77](https://github.com/grlee77)
- Fix path in update-version ([#852](https://github.com/rapidsai/cucim/pull/852)) [@raydouglass](https://github.com/raydouglass)
- avoid potential CUDA out of bounds memory access in test case ([#851](https://github.com/rapidsai/cucim/pull/851)) [@grlee77](https://github.com/grlee77)
- Consistently raise error on non-CuPy input to regionprops functions ([#849](https://github.com/rapidsai/cucim/pull/849)) [@grlee77](https://github.com/grlee77)
- update vendored binary_fill_holes ([#842](https://github.com/rapidsai/cucim/pull/842)) [@grlee77](https://github.com/grlee77)

## 📖 Documentation

- Add cuCIM pronunciation to README ([#823](https://github.com/rapidsai/cucim/pull/823)) [@jakirkham](https://github.com/jakirkham)

## 🛠️ Improvements

- Update various library versions to meet the minimum required CMake version (&gt;= 3.5) for compatibility with CMake 3.30.4. ([#858](https://github.com/rapidsai/cucim/pull/858)) [@gigony](https://github.com/gigony)
- Use conda-build instead of conda-mambabuild ([#841](https://github.com/rapidsai/cucim/pull/841)) [@bdice](https://github.com/bdice)
- require sphinx&lt;8.2.0 ([#840](https://github.com/rapidsai/cucim/pull/840)) [@jameslamb](https://github.com/jameslamb)
- Improve performance of `label2rgb` ([#839](https://github.com/rapidsai/cucim/pull/839)) [@grlee77](https://github.com/grlee77)
- add utility for memory efficient maximum pairwise distance computation with GPU support ([#838](https://github.com/rapidsai/cucim/pull/838)) [@grlee77](https://github.com/grlee77)
- vendor CUDA-accelerated `find objects` ([#837](https://github.com/rapidsai/cucim/pull/837)) [@grlee77](https://github.com/grlee77)
- update vendored binary morphology code ([#836](https://github.com/rapidsai/cucim/pull/836)) [@grlee77](https://github.com/grlee77)
- Consolidate more Conda solves in CI ([#835](https://github.com/rapidsai/cucim/pull/835)) [@KyleFromNVIDIA](https://github.com/KyleFromNVIDIA)
- Require CMake 3.30.4 ([#834](https://github.com/rapidsai/cucim/pull/834)) [@robertmaynard](https://github.com/robertmaynard)
- Create Conda CI test env in one step ([#833](https://github.com/rapidsai/cucim/pull/833)) [@KyleFromNVIDIA](https://github.com/KyleFromNVIDIA)
- Add build_type input to `test.yaml` ([#831](https://github.com/rapidsai/cucim/pull/831)) [@gforsyth](https://github.com/gforsyth)
- Implement `convex_hull_image` and `convex_hull_object` ([#828](https://github.com/rapidsai/cucim/pull/828)) [@grlee77](https://github.com/grlee77)
- Use shared-workflows branch-25.04 ([#826](https://github.com/rapidsai/cucim/pull/826)) [@bdice](https://github.com/bdice)
- add telemetry ([#822](https://github.com/rapidsai/cucim/pull/822)) [@msarahan](https://github.com/msarahan)
- raise setuptools floor to 61.0.0, sphinx floor to 8.0.0, other small dependencies cleanup ([#820](https://github.com/rapidsai/cucim/pull/820)) [@jameslamb](https://github.com/jameslamb)
- Forward-merge branch-25.02 to branch-25.04 ([#819](https://github.com/rapidsai/cucim/pull/819)) [@bdice](https://github.com/bdice)
- Migrate to NVKS for amd64 CI runners ([#817](https://github.com/rapidsai/cucim/pull/817)) [@bdice](https://github.com/bdice)

# cucim 25.02.00 (13 Feb 2025)

## 🐛 Bug Fixes

- Fix primitives benchmark code ([#812](https://github.com/rapidsai/cucim/pull/812)) [@gigony](https://github.com/gigony)
- CuPy 14.0 compatibility ([#808](https://github.com/rapidsai/cucim/pull/808)) [@grlee77](https://github.com/grlee77)

## 🛠️ Improvements

- Use `rapids-pip-retry` in CI jobs that might need retries ([#824](https://github.com/rapidsai/cucim/pull/824)) [@gforsyth](https://github.com/gforsyth)
- Revert CUDA 12.8 shared workflow branch changes ([#818](https://github.com/rapidsai/cucim/pull/818)) [@vyasr](https://github.com/vyasr)
- Build and test with CUDA 12.8.0 ([#815](https://github.com/rapidsai/cucim/pull/815)) [@bdice](https://github.com/bdice)
- Add shellcheck to pre-commit and fix warnings ([#814](https://github.com/rapidsai/cucim/pull/814)) [@gforsyth](https://github.com/gforsyth)
- Update vendored ndimage code with axes support ([#813](https://github.com/rapidsai/cucim/pull/813)) [@grlee77](https://github.com/grlee77)
- Use GCC 13 in CUDA 12 conda builds. ([#811](https://github.com/rapidsai/cucim/pull/811)) [@bdice](https://github.com/bdice)
- Improve performance of color distance calculations by kernel fusion ([#809](https://github.com/rapidsai/cucim/pull/809)) [@grlee77](https://github.com/grlee77)
- Incorporate upstream changes from scikit-image 0.25 ([#806](https://github.com/rapidsai/cucim/pull/806)) [@grlee77](https://github.com/grlee77)
- Update version references in workflow ([#803](https://github.com/rapidsai/cucim/pull/803)) [@AyodeAwe](https://github.com/AyodeAwe)
- Require approval to run CI on draft PRs ([#798](https://github.com/rapidsai/cucim/pull/798)) [@bdice](https://github.com/bdice)
- Add breaking change workflow trigger ([#795](https://github.com/rapidsai/cucim/pull/795)) [@AyodeAwe](https://github.com/AyodeAwe)

# cucim 24.12.00 (11 Dec 2024)

## 🚨 Breaking Changes

- Deprecations: carry out removals scheduled for release 24.12 ([#786](https://github.com/rapidsai/cucim/pull/786)) [@grlee77](https://github.com/grlee77)

## 🛠️ Improvements

- allow path conflicts in conda builds ([#801](https://github.com/rapidsai/cucim/pull/801)) [@jameslamb](https://github.com/jameslamb)
- enforce wheel size limits, README formatting in CI ([#797](https://github.com/rapidsai/cucim/pull/797)) [@jameslamb](https://github.com/jameslamb)
- build wheels without build isolation, print sccache stats in builds ([#792](https://github.com/rapidsai/cucim/pull/792)) [@jameslamb](https://github.com/jameslamb)
- make conda installs in CI stricter ([#791](https://github.com/rapidsai/cucim/pull/791)) [@jameslamb](https://github.com/jameslamb)
- Deprecations: carry out removals scheduled for release 24.12 ([#786](https://github.com/rapidsai/cucim/pull/786)) [@grlee77](https://github.com/grlee77)

# cucim 24.10.00 (9 Oct 2024)

## 🐛 Bug Fixes

- Use cupy to measure memory leak ([#777](https://github.com/rapidsai/cucim/pull/777)) [@bdice](https://github.com/bdice)
- Fix wheel tests for Rocky Linux 8. ([#774](https://github.com/rapidsai/cucim/pull/774)) [@bdice](https://github.com/bdice)
- Disable custom 2D separable filtering kernels on windows ([#770](https://github.com/rapidsai/cucim/pull/770)) [@grlee77](https://github.com/grlee77)
- chan_vese: pass all constants to `_fused_variance_kernel2` as device scalars ([#764](https://github.com/rapidsai/cucim/pull/764)) [@grlee77](https://github.com/grlee77)
- Fix &quot;compatibility&quot; spelling in CHANGELOG ([#759](https://github.com/rapidsai/cucim/pull/759)) [@jakirkham](https://github.com/jakirkham)
- Fix error in dependencies.yaml causing incomplete pyproject.toml generation ([#757](https://github.com/rapidsai/cucim/pull/757)) [@grlee77](https://github.com/grlee77)

## 🛠️ Improvements

- Do not depends on unused libraries for libtiff ([#785](https://github.com/rapidsai/cucim/pull/785)) [@gigony](https://github.com/gigony)
- Fix a couple of performance issues in `peak_local_max` (improve performance of blob detectors and `corner_peaks`) ([#782](https://github.com/rapidsai/cucim/pull/782)) [@grlee77](https://github.com/grlee77)
- update vendored CUDA includes to match CuPy &gt;= 13.3 ([#781](https://github.com/rapidsai/cucim/pull/781)) [@grlee77](https://github.com/grlee77)
- Use CI workflow branch &#39;branch-24.10&#39; again ([#780](https://github.com/rapidsai/cucim/pull/780)) [@jameslamb](https://github.com/jameslamb)
- Add support for Python 3.12 ([#773](https://github.com/rapidsai/cucim/pull/773)) [@jameslamb](https://github.com/jameslamb)
- Update rapidsai/pre-commit-hooks ([#772](https://github.com/rapidsai/cucim/pull/772)) [@KyleFromNVIDIA](https://github.com/KyleFromNVIDIA)
- bump scikit-image upper bound (allow 0.24.x) ([#769](https://github.com/rapidsai/cucim/pull/769)) [@grlee77](https://github.com/grlee77)
- Drop Python 3.9 support ([#766](https://github.com/rapidsai/cucim/pull/766)) [@jameslamb](https://github.com/jameslamb)
- Remove NumPy &lt;2 pin ([#762](https://github.com/rapidsai/cucim/pull/762)) [@seberg](https://github.com/seberg)
- Update pre-commit hooks ([#760](https://github.com/rapidsai/cucim/pull/760)) [@KyleFromNVIDIA](https://github.com/KyleFromNVIDIA)
- Switch to pytest-lazy-fixtures ([#756](https://github.com/rapidsai/cucim/pull/756)) [@gigony](https://github.com/gigony)

# cucim 24.08.00 (7 Aug 2024)

## 🛠️ Improvements

- Drop NumPy build dependency ([#751](https://github.com/rapidsai/cucim/pull/751)) [@jakirkham](https://github.com/jakirkham)
- Use workflow branch 24.08 again ([#749](https://github.com/rapidsai/cucim/pull/749)) [@KyleFromNVIDIA](https://github.com/KyleFromNVIDIA)
- Build and test with CUDA 12.5.1 ([#747](https://github.com/rapidsai/cucim/pull/747)) [@KyleFromNVIDIA](https://github.com/KyleFromNVIDIA)
- Minor fixes for NumPy 2.0 compatibility ([#746](https://github.com/rapidsai/cucim/pull/746)) [@grlee77](https://github.com/grlee77)
- skip CMake 3.30.0, require CMake &gt;=3.26.4 ([#745](https://github.com/rapidsai/cucim/pull/745)) [@jameslamb](https://github.com/jameslamb)
- Use verify-alpha-spec hook ([#744](https://github.com/rapidsai/cucim/pull/744)) [@KyleFromNVIDIA](https://github.com/KyleFromNVIDIA)
- remove .gitattributes ([#740](https://github.com/rapidsai/cucim/pull/740)) [@jameslamb](https://github.com/jameslamb)
- Adopt CI/packaging codeowners ([#739](https://github.com/rapidsai/cucim/pull/739)) [@bdice](https://github.com/bdice)
- Remove text builds of documentation ([#738](https://github.com/rapidsai/cucim/pull/738)) [@vyasr](https://github.com/vyasr)
- use rapids-build-backend ([#736](https://github.com/rapidsai/cucim/pull/736)) [@jameslamb](https://github.com/jameslamb)

# cucim 24.06.00 (5 Jun 2024)

## 🚨 Breaking Changes

- The `output` argument of `cucim.skimage.filters.gaussian` has been renamed to `out`. The old name is deprecated and will be removed in release 25.02 (#727)
- Renamed `get_xyz_coords` function is now removed (use `skimage.color.xyz_tristimulus_values` instead) (#724)
- Removed deprecated `return_error` kwarg from `phase_cross_correlation` (the error is now always returned) (#724)
- Removed deprecated `random_state` kwarg from `medial_axis` (it was renamed to `rng` previously) (#724)

## 🐛 Bug Fixes

- Use SciPy&#39;s KDTree instead of deprecated cKDTree ([#733](https://github.com/rapidsai/cucim/pull/733)) [@grlee77](https://github.com/grlee77)
- Binary and grayscale morphology functions have bug fixes in the case of even-sized/non-symmetric footprints (for details see upstream MR: https://github.com/scikit-image/scikit-image/pull/6695) (#728)

## 🚀 New Features

- `cucim.skimage.measure.regionprops` (and `regionprops_table`) support one new region property: `intensity_std` (#727)
- `cucim.skimage.segmentation.expand_labels` now supports a `spacing` keyword argument to take a pixel's physical dimensions into account (#727)
- binary morphology functions have a new `mode` argument that controls how values outside the image boundaries are interpreted (#728)
- grayscale morphology functions have new `mode` and `cval` arguments that control how boundaries are extended (#728)

## 🛠️ Improvements

- Enable FutureWarnings/DeprecationWarnings as errors ([#734](https://github.com/rapidsai/cucim/pull/734)) [@mroeschke](https://github.com/mroeschke)
- Migrate to `{{ stdlib(&quot;c&quot;) }}` ([#731](https://github.com/rapidsai/cucim/pull/731)) [@hcho3](https://github.com/hcho3)
- Implement upstream changes from scikit-image 0.23 (part 2 of 2: morphology) ([#728](https://github.com/rapidsai/cucim/pull/728)) [@grlee77](https://github.com/grlee77)
- Implement upstream changes from scikit-image 0.23 (part 1 of 2) ([#727](https://github.com/rapidsai/cucim/pull/727)) [@grlee77](https://github.com/grlee77)
- Update the test criteria for test_read_random_region_cpu_memleak ([#726](https://github.com/rapidsai/cucim/pull/726)) [@gigony](https://github.com/gigony)
- Remove code needed to support Python &lt; 3.9 and apply ruff&#39;s pyupgrade rules ([#725](https://github.com/rapidsai/cucim/pull/725)) [@grlee77](https://github.com/grlee77)
- removal of deprecated functions/kwargs scheduled for release 24.06 ([#724](https://github.com/rapidsai/cucim/pull/724)) [@grlee77](https://github.com/grlee77)
- Enable all tests for `arm` jobs ([#717](https://github.com/rapidsai/cucim/pull/717)) [@galipremsagar](https://github.com/galipremsagar)
- prevent path conflict ([#713](https://github.com/rapidsai/cucim/pull/713)) [@AyodeAwe](https://github.com/AyodeAwe)
- Updated cuCIM APIs for consistency with scikit-image 0.23.2 (#727 and #728)
- Additional modules use `__init__.pyi` instead of just `__init__.py` (#727)
- Some grayscale tests now compare directly to `skimage` CPU outputs instead fetching previously saved values (#728)
- Refactored some test cases to better use `pytest.mark.parametrize` (#728)
- Bumped version pinning for scikit-image to allow 0.23.x to be installed (#728)

## 📖 Documentation
- Various fixes to documentation strings (consistent shape notation, etc.) (#727)

# cuCIM 24.04.00 (10 Apr 2024)

## 🐛 Bug Fixes

- Require `click` as a wheel dependency ([#719](https://github.com/rapidsai/cucim/pull/719)) [@jakirkham](https://github.com/jakirkham)
- Fix docs upload directory ([#714](https://github.com/rapidsai/cucim/pull/714)) [@raydouglass](https://github.com/raydouglass)
- Fix `popd` indent in `run` ([#693](https://github.com/rapidsai/cucim/pull/693)) [@jakirkham](https://github.com/jakirkham)
- Re-run `ci/release/update-version.sh 24.04.00` ([#690](https://github.com/rapidsai/cucim/pull/690)) [@jakirkham](https://github.com/jakirkham)

## 🚀 New Features

- Support CUDA 12.2 ([#672](https://github.com/rapidsai/cucim/pull/672)) [@jameslamb](https://github.com/jameslamb)

## 🛠️ Improvements

- Use `conda env create --yes` instead of `--force` ([#716](https://github.com/rapidsai/cucim/pull/716)) [@bdice](https://github.com/bdice)
- Add upper bound to prevent usage of NumPy 2 ([#712](https://github.com/rapidsai/cucim/pull/712)) [@bdice](https://github.com/bdice)
- Remove hard-coding of RAPIDS version ([#711](https://github.com/rapidsai/cucim/pull/711)) [@KyleFromNVIDIA](https://github.com/KyleFromNVIDIA)
- Switch `pytest-xdist` algorithm to `worksteal` ([#708](https://github.com/rapidsai/cucim/pull/708)) [@bdice](https://github.com/bdice)
- Simplify version update script ([#705](https://github.com/rapidsai/cucim/pull/705)) [@jakirkham](https://github.com/jakirkham)
- Add support for Python 3.11, require NumPy 1.23+ ([#704](https://github.com/rapidsai/cucim/pull/704)) [@jameslamb](https://github.com/jameslamb)
- target branch-24.04 for GitHub Actions workflows ([#702](https://github.com/rapidsai/cucim/pull/702)) [@jameslamb](https://github.com/jameslamb)
- Refactor CUDA libraries in dependencies.yaml ([#699](https://github.com/rapidsai/cucim/pull/699)) [@bdice](https://github.com/bdice)
- Update ops-bot.yaml ([#694](https://github.com/rapidsai/cucim/pull/694)) [@AyodeAwe](https://github.com/AyodeAwe)
- add rapids-dependency-file-generator pre-commmit hook ([#682](https://github.com/rapidsai/cucim/pull/682)) [@jameslamb](https://github.com/jameslamb)

# cuCIM 24.02.00 (12 Feb 2024)

## 🐛 Bug Fixes

- Fix CUDA trove classifiers &amp; update README install instructions ([#695](https://github.com/rapidsai/cucim/pull/695)) [@jakirkham](https://github.com/jakirkham)
- Exclude PyTest 8 ([#689](https://github.com/rapidsai/cucim/pull/689)) [@jakirkham](https://github.com/jakirkham)
- Update OpenJPEG to 2.5.0 ([#685](https://github.com/rapidsai/cucim/pull/685)) [@jakirkham](https://github.com/jakirkham)
- Fix CI (pt. 2) ([#680](https://github.com/rapidsai/cucim/pull/680)) [@jakirkham](https://github.com/jakirkham)
- Fix CI issues ([#676](https://github.com/rapidsai/cucim/pull/676)) [@jakirkham](https://github.com/jakirkham)
- Remove update to symlink ([#674](https://github.com/rapidsai/cucim/pull/674)) [@raydouglass](https://github.com/raydouglass)
- Add 3rd party license file in Conda packages ([#654](https://github.com/rapidsai/cucim/pull/654)) [@jakirkham](https://github.com/jakirkham)
- Fix style issue in `docs/source/conf.py` ([#648](https://github.com/rapidsai/cucim/pull/648)) [@jakirkham](https://github.com/jakirkham)

## 🛠️ Improvements

- Consolidate test requirements in `dependencies.yaml` ([#683](https://github.com/rapidsai/cucim/pull/683)) [@jakirkham](https://github.com/jakirkham)
- Remove usages of rapids-env-update ([#673](https://github.com/rapidsai/cucim/pull/673)) [@KyleFromNVIDIA](https://github.com/KyleFromNVIDIA)
- refactor CUDA versions in dependencies.yaml ([#671](https://github.com/rapidsai/cucim/pull/671)) [@jameslamb](https://github.com/jameslamb)
- minor updates/fixes for consistency with scikit-image 0.22 ([#670](https://github.com/rapidsai/cucim/pull/670)) [@grlee77](https://github.com/grlee77)
- Update CODEOWNERS ([#669](https://github.com/rapidsai/cucim/pull/669)) [@ajschmidt8](https://github.com/ajschmidt8)
- remove redundant notebook ([#668](https://github.com/rapidsai/cucim/pull/668)) [@grlee77](https://github.com/grlee77)
- remove .idea folder (CLion IDE configuration) ([#667](https://github.com/rapidsai/cucim/pull/667)) [@grlee77](https://github.com/grlee77)
- Cleanup old ci and docs subfolders and related files under python/cucim ([#666](https://github.com/rapidsai/cucim/pull/666)) [@grlee77](https://github.com/grlee77)
- remove various files related to old wheel building mechanism ([#665](https://github.com/rapidsai/cucim/pull/665)) [@grlee77](https://github.com/grlee77)
- Relax `openslide` pin ([#653](https://github.com/rapidsai/cucim/pull/653)) [@jakirkham](https://github.com/jakirkham)
- install imagecodecs and openslide-python dependencies so additional tests will run ([#634](https://github.com/rapidsai/cucim/pull/634)) [@grlee77](https://github.com/grlee77)

# cuCIM 23.12.00 (6 Dec 2023)

## 🐛 Bug Fixes

- Retag wheels to be `cpXY` ([#644](https://github.com/rapidsai/cucim/pull/644)) [@jakirkham](https://github.com/jakirkham)
- remove leftover pyproject_.toml file ([#632](https://github.com/rapidsai/cucim/pull/632)) [@grlee77](https://github.com/grlee77)
- update version string (e.g. for CHANGELOG link) in pyproject.toml ([#630](https://github.com/rapidsai/cucim/pull/630)) [@grlee77](https://github.com/grlee77)
- fix import order in test case ([#624](https://github.com/rapidsai/cucim/pull/624)) [@grlee77](https://github.com/grlee77)
- Standardize on `rng` over `seed` and fix miscellaneous deprecation warnings ([#621](https://github.com/rapidsai/cucim/pull/621)) [@grlee77](https://github.com/grlee77)
- Fix iterator-related memory issues ([#620](https://github.com/rapidsai/cucim/pull/620)) [@gigony](https://github.com/gigony)

## 🚀 New Features

- build wheels on CI ([#619](https://github.com/rapidsai/cucim/pull/619)) [@grlee77](https://github.com/grlee77)

## 🛠️ Improvements

- carry out removals of previously deprecated functions/kwargs ([#631](https://github.com/rapidsai/cucim/pull/631)) [@grlee77](https://github.com/grlee77)
- Improve memory leak check stability in test_read_region_cpu_memleak test ([#623](https://github.com/rapidsai/cucim/pull/623)) [@gigony](https://github.com/gigony)
- Update linting tool versions and replace isort with ruff ([#622](https://github.com/rapidsai/cucim/pull/622)) [@grlee77](https://github.com/grlee77)
- Update packages (pybind11 and catch2) and do not use nvidia-docker command ([#618](https://github.com/rapidsai/cucim/pull/618)) [@gigony](https://github.com/gigony)
- Replace setup.py with pyproject toml ([#617](https://github.com/rapidsai/cucim/pull/617)) [@grlee77](https://github.com/grlee77)
- update linters and move their configurations from setup.cfg to pyproject.toml ([#616](https://github.com/rapidsai/cucim/pull/616)) [@grlee77](https://github.com/grlee77)
- remove versioneer ([#615](https://github.com/rapidsai/cucim/pull/615)) [@grlee77](https://github.com/grlee77)
- Update `shared-action-workflows` references ([#614](https://github.com/rapidsai/cucim/pull/614)) [@AyodeAwe](https://github.com/AyodeAwe)
- Use branch-23.12 workflows. ([#613](https://github.com/rapidsai/cucim/pull/613)) [@bdice](https://github.com/bdice)
- cucim: Build CUDA 12.0 ARM conda packages. ([#610](https://github.com/rapidsai/cucim/pull/610)) [@bdice](https://github.com/bdice)

# cuCIM 23.10.00 (11 Oct 2023)

## 🐛 Bug Fixes

- Use `conda mambabuild` not `mamba mambabuild` ([#607](https://github.com/rapidsai/cucim/pull/607)) [@bdice](https://github.com/bdice)

## 📖 Documentation

- minor updates to release 23.08 changelog ([#605](https://github.com/rapidsai/cucim/pull/605)) [@grlee77](https://github.com/grlee77)

## 🛠️ Improvements

- Update image names ([#609](https://github.com/rapidsai/cucim/pull/609)) [@AyodeAwe](https://github.com/AyodeAwe)
- Use `copy-pr-bot` ([#606](https://github.com/rapidsai/cucim/pull/606)) [@ajschmidt8](https://github.com/ajschmidt8)

# cuCIM 23.08.00 (9 Aug 2023)

## 🚨 Breaking Changes

- Sync cuCIM API with scikit-image 0.21 ([#573](https://github.com/rapidsai/cucim/pull/573)) [@grlee77](https://github.com/grlee77)
- The `random_state` argument of medial_axis and unsupervised_wiener is now deprecated and will be removed in the future. The new argument name, `seed`, should be used instead. ([#573](https://github.com/rapidsai/cucim/pull/573)) [@grlee77](https://github.com/grlee77)
- The existing function `cucim.skimage.color.get_xyz_coords` has been renamed `cucim.skimage.color.xyz_tristimulus_values`. The former function name is deprecated and will be removed in the future. ([#573](https://github.com/rapidsai/cucim/pull/573)) [@grlee77](https://github.com/grlee77)

## 🐛 Bug Fixes

- Remove libwebp-base upper bound. ([#599](https://github.com/rapidsai/cucim/pull/599)) [@bdice](https://github.com/bdice)
- Fix ignore_run_exports_from. ([#596](https://github.com/rapidsai/cucim/pull/596)) [@bdice](https://github.com/bdice)
- Add ignore_run_exports for CUDA 11 ([#593](https://github.com/rapidsai/cucim/pull/593)) [@raydouglass](https://github.com/raydouglass)
- Use linalg &amp; inline `_get_manders_overlap_coeff` ([#578](https://github.com/rapidsai/cucim/pull/578)) [@jakirkham](https://github.com/jakirkham)
- Fix canny and butterworth (recent CuPy and NumPy compatibility) ([#574](https://github.com/rapidsai/cucim/pull/574)) [@grlee77](https://github.com/grlee77)
- A bug was fixed in 2D shear calculations for AffineTransform. ([#573](https://github.com/rapidsai/cucim/pull/573)) [@grlee77](https://github.com/grlee77)
- A bug was fixed in the energy calculation of `cucim.skimage.segmentation.chan_vese`. This fix may result in different output from previous versions. ([#573](https://github.com/rapidsai/cucim/pull/573)) [@grlee77](https://github.com/grlee77)

## 🛠️ Improvements

- The formerly private function `_invariant_denoise` has been renamed `denoise_invariant` and is now part of the public `cucim.skimage.restoration` API ([#573](https://github.com/rapidsai/cucim/pull/573)) [@grlee77](https://github.com/grlee77)
- A new `return_mapping` option was added to `cucim.skimage.segmentation.join_segmentations`. This provides an additional output with a mapping between the labels in the joined segmentation and the original ones. ([#573](https://github.com/rapidsai/cucim/pull/573)) [@grlee77](https://github.com/grlee77)
- Added support for y-axis shear to the 2D AffineTransform. ([#573](https://github.com/rapidsai/cucim/pull/573)) [@grlee77](https://github.com/grlee77)
- Postponed the assessment of GPU memory for testing ([#601](https://github.com/rapidsai/cucim/pull/601)) [@gigony](https://github.com/gigony)
- Do not use x86_64 GDS binaries for aarch64 ([#590](https://github.com/rapidsai/cucim/pull/590)) [@gigony](https://github.com/gigony)
- remove checks for versions of scikit-image that are no longer supported ([#587](https://github.com/rapidsai/cucim/pull/587)) [@grlee77](https://github.com/grlee77)
- Allow scikit-image 0.21.0 ([#580](https://github.com/rapidsai/cucim/pull/580)) [@jakirkham](https://github.com/jakirkham)
- Drop unneeded selector on `libwebp-base` ([#579](https://github.com/rapidsai/cucim/pull/579)) [@jakirkham](https://github.com/jakirkham)
- switch from bundled lazy loading code to the public lazy_loader package ([#575](https://github.com/rapidsai/cucim/pull/575)) [@grlee77](https://github.com/grlee77)
- Sync cuCIM API with scikit-image 0.21 ([#573](https://github.com/rapidsai/cucim/pull/573)) [@grlee77](https://github.com/grlee77)
- cuCIM: Build CUDA 12 packages ([#572](https://github.com/rapidsai/cucim/pull/572)) [@jakirkham](https://github.com/jakirkham)
- use rapids-upload-docs script ([#570](https://github.com/rapidsai/cucim/pull/570)) [@AyodeAwe](https://github.com/AyodeAwe)
- Remove documentation build scripts for Jenkins ([#567](https://github.com/rapidsai/cucim/pull/567)) [@ajschmidt8](https://github.com/ajschmidt8)

# cuCIM 23.06.00 (7 Jun 2023)

## 🚨 Breaking Changes

- Support Python 3.9 build/tests ([#547](https://github.com/rapidsai/cucim/pull/547)) [@shwina](https://github.com/shwina)

## 🐛 Bug Fixes

- Fix SHA256 check failure in test suite ([#564](https://github.com/rapidsai/cucim/pull/564)) [@grlee77](https://github.com/grlee77)
- Handle space character in ./run download_testdata ([#556](https://github.com/rapidsai/cucim/pull/556)) [@gigony](https://github.com/gigony)
- Fix `return_error=&#39;always&#39;` behavior in phase_cross_correlation ([#549](https://github.com/rapidsai/cucim/pull/549)) [@grlee77](https://github.com/grlee77)
- Only load versioned `libcufile` ([#548](https://github.com/rapidsai/cucim/pull/548)) [@jakirkham](https://github.com/jakirkham)
- add a 20 minute timeout for pytest runs on CI ([#545](https://github.com/rapidsai/cucim/pull/545)) [@grlee77](https://github.com/grlee77)
- protect against possible out of bounds memory access in 2D distance transform ([#540](https://github.com/rapidsai/cucim/pull/540)) [@grlee77](https://github.com/grlee77)

## 📖 Documentation

- Fix doc building via `run build_package` ([#553](https://github.com/rapidsai/cucim/pull/553)) [@grlee77](https://github.com/grlee77)
- update changelog for release 23.04.00 and 23.04.01 ([#552](https://github.com/rapidsai/cucim/pull/552)) [@grlee77](https://github.com/grlee77)

## 🛠️ Improvements

- Allow numpy 1.24. ([#563](https://github.com/rapidsai/cucim/pull/563)) [@bdice](https://github.com/bdice)
- run docs nightly too ([#560](https://github.com/rapidsai/cucim/pull/560)) [@AyodeAwe](https://github.com/AyodeAwe)
- Update cupy dependency ([#558](https://github.com/rapidsai/cucim/pull/558)) [@vyasr](https://github.com/vyasr)
- Remove libjpeg dependency ([#557](https://github.com/rapidsai/cucim/pull/557)) [@gigony](https://github.com/gigony)
- Enable sccache hits from local builds ([#551](https://github.com/rapidsai/cucim/pull/551)) [@AyodeAwe](https://github.com/AyodeAwe)
- Revert shared workflows branch ([#550](https://github.com/rapidsai/cucim/pull/550)) [@ajschmidt8](https://github.com/ajschmidt8)
- Support Python 3.9 build/tests ([#547](https://github.com/rapidsai/cucim/pull/547)) [@shwina](https://github.com/shwina)
- Remove usage of rapids-get-rapids-version-from-git ([#546](https://github.com/rapidsai/cucim/pull/546)) [@jjacobelli](https://github.com/jjacobelli)
- Use ARC V2 self-hosted runners for GPU jobs ([#538](https://github.com/rapidsai/cucim/pull/538)) [@jjacobelli](https://github.com/jjacobelli)
- Remove underscore in build string. ([#528](https://github.com/rapidsai/cucim/pull/528)) [@bdice](https://github.com/bdice)

# cuCIM 23.04.01 (14 Apr 2023)

## 🛠️ Improvements

- Pin libwebp-base ([#541](https://github.com/rapidsai/cucim/pull/541)) [@ajschmidt8](https://github.com/ajschmidt8)

# cuCIM 23.04.00 (6 Apr 2023)

## 🚨 Breaking Changes

- Fix inefficiency in handling clipping of image range in `resize` and other transforms ([#516](https://github.com/rapidsai/cucim/pull/516)) [@grlee77](https://github.com/grlee77)

## 🐛 Bug Fixes

- Fix bug in median filter with non-uniform footprint ([#521](https://github.com/rapidsai/cucim/pull/521)) [@grlee77](https://github.com/grlee77)
- use cp.around instead of cp.round for CuPy 10.x compatibility ([#508](https://github.com/rapidsai/cucim/pull/508)) [@grlee77](https://github.com/grlee77)
- Fix error in LZ4-compressed Zarr writing demo ([#506](https://github.com/rapidsai/cucim/pull/506)) [@grlee77](https://github.com/grlee77)
- Normalize whitespace. ([#474](https://github.com/rapidsai/cucim/pull/474)) [@bdice](https://github.com/bdice)

## 🛠️ Improvements

- allow scikit-image 0.20 as well ([#536](https://github.com/rapidsai/cucim/pull/536)) [@grlee77](https://github.com/grlee77)
- Pass `AWS_SESSION_TOKEN` and `SCCACHE_S3_USE_SSL` vars to conda build ([#525](https://github.com/rapidsai/cucim/pull/525)) [@ajschmidt8](https://github.com/ajschmidt8)
- Update aarch64 to GCC 11 ([#524](https://github.com/rapidsai/cucim/pull/524)) [@bdice](https://github.com/bdice)
- Update to GCC 11 ([#522](https://github.com/rapidsai/cucim/pull/522)) [@bdice](https://github.com/bdice)
- Upgrade dockcross and pybind11 ([#519](https://github.com/rapidsai/cucim/pull/519)) [@gigony](https://github.com/gigony)
- Binary morphology: omit weights array when possible ([#517](https://github.com/rapidsai/cucim/pull/517)) [@grlee77](https://github.com/grlee77)
- Fix inefficiency in handling clipping of image range in `resize` and other transforms ([#516](https://github.com/rapidsai/cucim/pull/516)) [@grlee77](https://github.com/grlee77)
- Fix GHA build workflow ([#515](https://github.com/rapidsai/cucim/pull/515)) [@AjayThorve](https://github.com/AjayThorve)
- Reduce error handling verbosity in CI tests scripts ([#511](https://github.com/rapidsai/cucim/pull/511)) [@AjayThorve](https://github.com/AjayThorve)
- Update shared workflow branches ([#510](https://github.com/rapidsai/cucim/pull/510)) [@ajschmidt8](https://github.com/ajschmidt8)
- Remove gpuCI scripts. ([#505](https://github.com/rapidsai/cucim/pull/505)) [@bdice](https://github.com/bdice)
- Move date to build string in `conda` recipe ([#497](https://github.com/rapidsai/cucim/pull/497)) [@ajschmidt8](https://github.com/ajschmidt8)

# cuCIM 23.02.00 (9 Feb 2023)

## 🚨 Breaking Changes

- Add disambiguation option to phase_cross_correlation (skimage 0.20 feature) ([#486](https://github.com/rapidsai/cucim/pull/486)) [@grlee77](https://github.com/grlee77)

## 🐛 Bug Fixes

- apply bug fix to vendored ndimage code ([#494](https://github.com/rapidsai/cucim/pull/494)) [@grlee77](https://github.com/grlee77)
- Closes #490 -- fixes bug in hue jitter ([#491](https://github.com/rapidsai/cucim/pull/491)) [@benlansdell](https://github.com/benlansdell)
- Fix random seed used in test_3d_similarity_estimation ([#472](https://github.com/rapidsai/cucim/pull/472)) [@grlee77](https://github.com/grlee77)

## 📖 Documentation

- Fix documentation author ([#475](https://github.com/rapidsai/cucim/pull/475)) [@bdice](https://github.com/bdice)

## 🚀 New Features

- Add colocalization measures ([#488](https://github.com/rapidsai/cucim/pull/488)) [@grlee77](https://github.com/grlee77)
- Add disambiguation option to phase_cross_correlation (skimage 0.20 feature) ([#486](https://github.com/rapidsai/cucim/pull/486)) [@grlee77](https://github.com/grlee77)

## 🛠️ Improvements

- Update shared workflow branches ([#501](https://github.com/rapidsai/cucim/pull/501)) [@ajschmidt8](https://github.com/ajschmidt8)
- Update `isort` version to 5.12.0 ([#492](https://github.com/rapidsai/cucim/pull/492)) [@ajschmidt8](https://github.com/ajschmidt8)
- Improve rank filtering performance by removing use of footprint kernel when possible ([#485](https://github.com/rapidsai/cucim/pull/485)) [@grlee77](https://github.com/grlee77)
- use vendored version of cupy.pad with added performance optimizations ([#482](https://github.com/rapidsai/cucim/pull/482)) [@grlee77](https://github.com/grlee77)
- add docs builds to Github Actions ([#481](https://github.com/rapidsai/cucim/pull/481)) [@AjayThorve](https://github.com/AjayThorve)
- Update `numpy` version specifier ([#480](https://github.com/rapidsai/cucim/pull/480)) [@ajschmidt8](https://github.com/ajschmidt8)
- Build CUDA `11.8` and Python `3.10` Packages ([#476](https://github.com/rapidsai/cucim/pull/476)) [@ajschmidt8](https://github.com/ajschmidt8)
- Add GitHub Actions Workflows. ([#471](https://github.com/rapidsai/cucim/pull/471)) [@bdice](https://github.com/bdice)
- Fix conflicts in &quot;Forward-merge branch-22.12 to branch-23.02&quot; ([#468](https://github.com/rapidsai/cucim/pull/468)) [@jakirkham](https://github.com/jakirkham)
- Enable copy_prs. ([#465](https://github.com/rapidsai/cucim/pull/465)) [@bdice](https://github.com/bdice)

# cuCIM 22.12.00 (8 Dec 2022)

## 🚨 Breaking Changes

- Implement additional deprecations carried out for scikit-image 0.20 ([#451](https://github.com/rapidsai/cucim/pull/451)) [@grlee77](https://github.com/grlee77)
- improved implementation of ridge filters (bug fixes and reduced memory footprint) ([#423](https://github.com/rapidsai/cucim/pull/423)) [@grlee77](https://github.com/grlee77)

## 🐛 Bug Fixes

- pin to cmake !3.25.0 on CI to avoid bug with CUDA+conda during build ([#444](https://github.com/rapidsai/cucim/pull/444)) [@grlee77](https://github.com/grlee77)
- update incorrect argument and deprecated function for tifffile.TiffWriter ([#433](https://github.com/rapidsai/cucim/pull/433)) [@JoohyungLee0106](https://github.com/JoohyungLee0106)
- Fix rotate behavior for ndim &gt; 2 ([#432](https://github.com/rapidsai/cucim/pull/432)) [@grlee77](https://github.com/grlee77)

## 📖 Documentation

- add whole-slide tiled read/write demos for measuring GPUDirect Storage (GDS) I/O performance ([#452](https://github.com/rapidsai/cucim/pull/452)) [@grlee77](https://github.com/grlee77)
- Add demo for distance_transform_edt ([#394](https://github.com/rapidsai/cucim/pull/394)) [@grlee77](https://github.com/grlee77)

## 🚀 New Features

- Support no-compression method in converter ([#443](https://github.com/rapidsai/cucim/pull/443)) [@gigony](https://github.com/gigony)
- add three segmentation metrics ([#425](https://github.com/rapidsai/cucim/pull/425)) [@grlee77](https://github.com/grlee77)
- add isotropic binary morphology functions ([#421](https://github.com/rapidsai/cucim/pull/421)) [@grlee77](https://github.com/grlee77)
- Add blob feature detectors (blob_dog, blob_log, blob_doh) ([#413](https://github.com/rapidsai/cucim/pull/413)) [@monzelr](https://github.com/monzelr)

## 🛠️ Improvements

- additional minor updates (skimage 0.20) ([#455](https://github.com/rapidsai/cucim/pull/455)) [@grlee77](https://github.com/grlee77)
- Implement additional deprecations carried out for scikit-image 0.20 ([#451](https://github.com/rapidsai/cucim/pull/451)) [@grlee77](https://github.com/grlee77)
- Faster `hessian_matrix_*` and `structure_tensor_eigvals` via analytical eigenvalues for the 3D case ([#434](https://github.com/rapidsai/cucim/pull/434)) [@grlee77](https://github.com/grlee77)
- use fused kernels to reduce overhead in corner detector implementations ([#426](https://github.com/rapidsai/cucim/pull/426)) [@grlee77](https://github.com/grlee77)
- Misc updates for consistency with scikit-image 0.20 ([#424](https://github.com/rapidsai/cucim/pull/424)) [@grlee77](https://github.com/grlee77)
- improved implementation of ridge filters (bug fixes and reduced memory footprint) ([#423](https://github.com/rapidsai/cucim/pull/423)) [@grlee77](https://github.com/grlee77)
- analytical moments computations, support pixel spacings in moments and regionprops ([#422](https://github.com/rapidsai/cucim/pull/422)) [@grlee77](https://github.com/grlee77)
- Forward merge branch-22.10 to branch-22.12 ([#420](https://github.com/rapidsai/cucim/pull/420)) [@grlee77](https://github.com/grlee77)
- Support `sampling` kwarg for `distance_transform_edt` (take pixel/voxel sizes into account) ([#407](https://github.com/rapidsai/cucim/pull/407)) [@grlee77](https://github.com/grlee77)
- Improve performance of Euclidean distance transform ([#406](https://github.com/rapidsai/cucim/pull/406)) [@grlee77](https://github.com/grlee77)

# cuCIM 22.10.00 (12 Oct 2022)

## 🐛 Bug Fixes

- Correctly use dtype when computing shared memory requirements of separable convolution ([#409](https://github.com/rapidsai/cucim/pull/409)) [@grlee77](https://github.com/grlee77)
- Forward-merge branch-22.08 to branch-22.10 ([#403](https://github.com/rapidsai/cucim/pull/403)) [@jakirkham](https://github.com/jakirkham)
- Add missing imports of euler_number and perimeter_crofton ([#386](https://github.com/rapidsai/cucim/pull/386)) [@grlee77](https://github.com/grlee77)

## 📖 Documentation

- update pypi CHANGELOG.md for releases 22.08.00 and 22.08.01 ([#404](https://github.com/rapidsai/cucim/pull/404)) [@grlee77](https://github.com/grlee77)
- Update README.md ([#396](https://github.com/rapidsai/cucim/pull/396)) [@HesAnEasyCoder](https://github.com/HesAnEasyCoder)

## 🚀 New Features

- Allow cupy 11 ([#399](https://github.com/rapidsai/cucim/pull/399)) [@galipremsagar](https://github.com/galipremsagar)
- Add cucim.skimage.feature.match_descriptors ([#338](https://github.com/rapidsai/cucim/pull/338)) [@grlee77](https://github.com/grlee77)

## 🛠️ Improvements

- Merge docs and add links ([#415](https://github.com/rapidsai/cucim/pull/415)) [@jakirkham](https://github.com/jakirkham)
- Add benchmarks for scikit-image functions introduced in 22.08 ([#378](https://github.com/rapidsai/cucim/pull/378)) [@grlee77](https://github.com/grlee77)

# cuCIM 22.08.00 (17 Aug 2022)

## 🚨 Breaking Changes

- Stain extraction: use a less strict condition across channels when thresholding ([#316](https://github.com/rapidsai/cucim/pull/316)) [@grlee77](https://github.com/grlee77)

## 🐛 Bug Fixes

- create SimilarityTransform using CuPy 9.x-compatible indexing ([#365](https://github.com/rapidsai/cucim/pull/365)) [@grlee77](https://github.com/grlee77)
- Add `__init__.py` in `cucim.core` ([#359](https://github.com/rapidsai/cucim/pull/359)) [@jakirkham](https://github.com/jakirkham)
- Stain extraction: use a less strict condition across channels when thresholding ([#316](https://github.com/rapidsai/cucim/pull/316)) [@grlee77](https://github.com/grlee77)
- Incorporate bug fixes from skimage 0.19.3 ([#312](https://github.com/rapidsai/cucim/pull/312)) [@grlee77](https://github.com/grlee77)
- fix RawKernel bug for canny filter when quantiles are used ([#310](https://github.com/rapidsai/cucim/pull/310)) [@grlee77](https://github.com/grlee77)

## 📖 Documentation

- Defer loading of `custom.js` ([#383](https://github.com/rapidsai/cucim/pull/383)) [@galipremsagar](https://github.com/galipremsagar)
- add cucim.core.morphology to API docs + other docstring fixes ([#367](https://github.com/rapidsai/cucim/pull/367)) [@grlee77](https://github.com/grlee77)
- Update README.md ([#361](https://github.com/rapidsai/cucim/pull/361)) [@HesAnEasyCoder](https://github.com/HesAnEasyCoder)
- remove unimplemented functions from See Also and fix version numbers in deprecation warnings ([#356](https://github.com/rapidsai/cucim/pull/356)) [@grlee77](https://github.com/grlee77)
- Forward-merge branch-22.06 to branch-22.08 ([#344](https://github.com/rapidsai/cucim/pull/344)) [@grlee77](https://github.com/grlee77)
- Update README.md ([#315](https://github.com/rapidsai/cucim/pull/315)) [@HesAnEasyCoder](https://github.com/HesAnEasyCoder)
- Update index.rst ([#314](https://github.com/rapidsai/cucim/pull/314)) [@HesAnEasyCoder](https://github.com/HesAnEasyCoder)
- Update PyPI package documentation for v22.06.00 ([#311](https://github.com/rapidsai/cucim/pull/311)) [@gigony](https://github.com/gigony)

## 🚀 New Features

- Add segmentation with the Chan-Vese active contours method ([#343](https://github.com/rapidsai/cucim/pull/343)) [@grlee77](https://github.com/grlee77)
- Add cucim.skimage.morphology.medial_axis ([#342](https://github.com/rapidsai/cucim/pull/342)) [@grlee77](https://github.com/grlee77)
- Add cucim.skimage.segmentation.expand_labels ([#341](https://github.com/rapidsai/cucim/pull/341)) [@grlee77](https://github.com/grlee77)
- Add Euclidean distance transform for images/volumes ([#318](https://github.com/rapidsai/cucim/pull/318)) [@grlee77](https://github.com/grlee77)

## 🛠️ Improvements

- Revert &quot;Allow CuPy 11&quot; ([#362](https://github.com/rapidsai/cucim/pull/362)) [@galipremsagar](https://github.com/galipremsagar)
- Fix issues with day &amp; night modes in python docs ([#360](https://github.com/rapidsai/cucim/pull/360)) [@galipremsagar](https://github.com/galipremsagar)
- Allow CuPy 11 ([#357](https://github.com/rapidsai/cucim/pull/357)) [@jakirkham](https://github.com/jakirkham)
- more efficient separable convolution ([#355](https://github.com/rapidsai/cucim/pull/355)) [@grlee77](https://github.com/grlee77)
- Support resolution and spacing metadata ([#349](https://github.com/rapidsai/cucim/pull/349)) [@gigony](https://github.com/gigony)
- Performance optimizations to morphological segmentation functions ([#340](https://github.com/rapidsai/cucim/pull/340)) [@grlee77](https://github.com/grlee77)
- benchmarks: avoid use of deprecated pandas API ([#339](https://github.com/rapidsai/cucim/pull/339)) [@grlee77](https://github.com/grlee77)
- Reduce memory overhead and improve performance of normalize_colors_pca ([#328](https://github.com/rapidsai/cucim/pull/328)) [@grlee77](https://github.com/grlee77)
- Protect against obscure divide by zero error in edge case of `normalize_colors_pca` ([#327](https://github.com/rapidsai/cucim/pull/327)) [@grlee77](https://github.com/grlee77)
- complete parametrization of cucim.skimage benchmarks ([#324](https://github.com/rapidsai/cucim/pull/324)) [@grlee77](https://github.com/grlee77)
- parameterization of `filters` and `features` benchmarks (v2) ([#322](https://github.com/rapidsai/cucim/pull/322)) [@grlee77](https://github.com/grlee77)
- Add a fast histogram-based median filter ([#317](https://github.com/rapidsai/cucim/pull/317)) [@grlee77](https://github.com/grlee77)
- Remove custom compiler environment variables ([#307](https://github.com/rapidsai/cucim/pull/307)) [@ajschmidt8](https://github.com/ajschmidt8)

# cuCIM 22.06.00 (7 Jun 2022)

## 🚨 Breaking Changes

- Promote small integer types to single rather than double precision ([#278](https://github.com/rapidsai/cucim/pull/278)) [@grlee77](https://github.com/grlee77)

## 🐛 Bug Fixes

- Populate correct channel names for RGBA image ([#294](https://github.com/rapidsai/cucim/pull/294)) [@gigony](https://github.com/gigony)
- Merge branch-22.04 into branch-22.06 ([#258](https://github.com/rapidsai/cucim/pull/258)) [@jakirkham](https://github.com/jakirkham)

## 📖 Documentation

- update outdated links to example data ([#289](https://github.com/rapidsai/cucim/pull/289)) [@grlee77](https://github.com/grlee77)
- Add missing API docs ([#275](https://github.com/rapidsai/cucim/pull/275)) [@grlee77](https://github.com/grlee77)

## 🚀 New Features

- add missing `cucim.skimage.segmentation.clear_border` function ([#267](https://github.com/rapidsai/cucim/pull/267)) [@grlee77](https://github.com/grlee77)
- add `cucim.core.operations.color.stain_extraction_pca` and `cucim.core.operations.color.normalize_colors_pca` for digital pathology H&E stain extraction and normalization ([#273](https://github.com/rapidsai/cucim/pull/273)) [@grlee77](https://github.com/grlee77), [@drbeh](https://github.com/drbeh)

## 🛠️ Improvements

- Update to use DLPack v0.6 ([#295](https://github.com/rapidsai/cucim/pull/295)) [@gigony](https://github.com/gigony)
- Remove plugin-related messages temporarily ([#291](https://github.com/rapidsai/cucim/pull/291)) [@gigony](https://github.com/gigony)
- Simplify recipes ([#286](https://github.com/rapidsai/cucim/pull/286)) [@Ethyling](https://github.com/Ethyling)
- Use cupy.fuse to improve efficiency hessian_matrix_eigvals ([#280](https://github.com/rapidsai/cucim/pull/280)) [@grlee77](https://github.com/grlee77)
- Promote small integer types to single rather than double precision ([#278](https://github.com/rapidsai/cucim/pull/278)) [@grlee77](https://github.com/grlee77)
- improve efficiency of histogram-based thresholding functions ([#276](https://github.com/rapidsai/cucim/pull/276)) [@grlee77](https://github.com/grlee77)
- Remove unused dependencies in GPU tests job ([#268](https://github.com/rapidsai/cucim/pull/268)) [@Ethyling](https://github.com/Ethyling)
- Enable footprint decomposition for morphology ([#274](https://github.com/rapidsai/cucim/pull/274)) [@grlee77](https://github.com/grlee77)
- Use conda compilers ([#232](https://github.com/rapidsai/cucim/pull/232)) [@Ethyling](https://github.com/Ethyling)
- Build packages using mambabuild ([#216](https://github.com/rapidsai/cucim/pull/216)) [@Ethyling](https://github.com/Ethyling)

# cuCIM 22.04.00 (6 Apr 2022)

## 🚨 Breaking Changes

- Apply fixes to skimage.transform scheduled for scikit-image 0.19.2 ([#208](https://github.com/rapidsai/cucim/pull/208)) [@grlee77](https://github.com/grlee77)

## 🐛 Bug Fixes

- Fix ImportError from vendored code ([#252](https://github.com/rapidsai/cucim/pull/252)) [@grlee77](https://github.com/grlee77)
- Fix wrong dimension in metadata ([#248](https://github.com/rapidsai/cucim/pull/248)) [@gigony](https://github.com/gigony)
- Handle file descriptor ownership and update documents for GDS ([#234](https://github.com/rapidsai/cucim/pull/234)) [@gigony](https://github.com/gigony)
- Check nullptr of handler in CuFileDriver::close() ([#229](https://github.com/rapidsai/cucim/pull/229)) [@gigony](https://github.com/gigony)
- Fix docs builds ([#218](https://github.com/rapidsai/cucim/pull/218)) [@ajschmidt8](https://github.com/ajschmidt8)
- Apply fixes to skimage.transform scheduled for scikit-image 0.19.2 ([#208](https://github.com/rapidsai/cucim/pull/208)) [@grlee77](https://github.com/grlee77)

## 📖 Documentation

- Update PyPI cuCIM v22.02.01 CHANGELOG.md ([#249](https://github.com/rapidsai/cucim/pull/249)) [@gigony](https://github.com/gigony)
- Update GTC 2021 Spring video links ([#227](https://github.com/rapidsai/cucim/pull/227)) [@gigony](https://github.com/gigony)
- Update documents for v22.02.00 ([#226](https://github.com/rapidsai/cucim/pull/226)) [@gigony](https://github.com/gigony)
- Merge branch-22.02 into branch-22.04 ([#220](https://github.com/rapidsai/cucim/pull/220)) [@jakirkham](https://github.com/jakirkham)

## 🛠️ Improvements

- Expose data type of CuImage object for interoperability with NumPy ([#246](https://github.com/rapidsai/cucim/pull/246)) [@gigony](https://github.com/gigony)
- Temporarily disable new `ops-bot` functionality ([#239](https://github.com/rapidsai/cucim/pull/239)) [@ajschmidt8](https://github.com/ajschmidt8)
- Add `.github/ops-bot.yaml` config file ([#236](https://github.com/rapidsai/cucim/pull/236)) [@ajschmidt8](https://github.com/ajschmidt8)
- randomization per image per batch ([#231](https://github.com/rapidsai/cucim/pull/231)) [@shekhardw](https://github.com/shekhardw)

# cuCIM 22.02.00 (2 Feb 2022)

## 🚨 Breaking Changes

- Update cucim.skimage API to match scikit-image 0.19 ([#190](https://github.com/rapidsai/cucim/pull/190)) [@glee77](https://github.com/glee77)

## 🐛 Bug Fixes

- Fix a bug in [v21.12.01](https://github.com/rapidsai/cucim/wiki/release_notes_v21.12.01) ([#191](https://github.com/rapidsai/cucim/pull/191)) [@gigony](https://github.com/gigony)
  - Fix GPU memory leak when using nvJPEG API (when `device='cuda'` parameter is used in `read_region` method).
- Fix segfault for preferred_memory_capacity in Python 3.9+ ([#214](https://github.com/rapidsai/cucim/pull/214)) [@gigony](https://github.com/gigony)

## 📖 Documentation

- PyPI v21.12.00 release ([#182](https://github.com/rapidsai/cucim/pull/182)) [@gigony](https://github.com/gigony)

## 🚀 New Features

- Update cucim.skimage API to match scikit-image 0.19 ([#190](https://github.com/rapidsai/cucim/pull/190)) [@glee77](https://github.com/glee77)
- Support multi-threads and batch, and support nvJPEG for JPEG-compressed images ([#191](https://github.com/rapidsai/cucim/pull/191)) [@gigony](https://github.com/gigony)
- Allow CuPy 10 ([#195](https://github.com/rapidsai/cucim/pull/195)) [@jakikham](https://github.com/jakikham)

## 🛠️ Improvements

- Add missing imports tests ([#183](https://github.com/rapidsai/cucim/pull/183)) [@Ethyling](https://github.com/Ethyling)
- Allow installation with CuPy 10 ([#197](https://github.com/rapidsai/cucim/pull/197)) [@glee77](https://github.com/glee77)
- Upgrade Numpy to 1.18 for Python 3.9 support ([#196](https://github.com/rapidsai/cucim/pull/196)) [@Ethyling](https://github.com/Ethyling)
- Upgrade Numpy to 1.19 for Python 3.9 support ([#203](https://github.com/rapidsai/cucim/pull/203)) [@Ethyling](https://github.com/Ethyling)

# cuCIM 21.12.00 (9 Dec 2021)

## 🚀 New Features

- Support Aperio SVS with CPU LZW and jpeg2k decoder ([#141](https://github.com/rapidsai/cucim/pull/141)) [[@gigony](https://github.com/gigony)](https://github.com/gigony](https://github.com/gigony))
- Add NVTX support for performance analysis ([#144](https://github.com/rapidsai/cucim/pull/144)) [[@gigony](https://github.com/gigony)](https://github.com/gigony](https://github.com/gigony))
- Normalize operation ([#150](https://github.com/rapidsai/cucim/pull/150)) [[@shekhardw](https://github.com/shekhardw)](https://github.com/shekhardw](https://github.com/shekhardw))

## 🐛 Bug Fixes

- Load libcufile.so with RTLD_NODELETE flag ([#177](https://github.com/rapidsai/cucim/pull/177)) [[@gigony](https://github.com/gigony)](https://github.com/gigony](https://github.com/gigony))
- Remove rmm/nvcc dependencies to fix cudaErrorUnsupportedPtxVersion error ([#175](https://github.com/rapidsai/cucim/pull/175)) [[@gigony](https://github.com/gigony)](https://github.com/gigony](https://github.com/gigony))
- Do not compile code with nvcc if no CUDA kernel exists ([#171](https://github.com/rapidsai/cucim/pull/171)) [[@gigony](https://github.com/gigony)](https://github.com/gigony](https://github.com/gigony))
- Fix a segmentation fault due to unloaded libcufile ([#158](https://github.com/rapidsai/cucim/pull/158)) [[@gigony](https://github.com/gigony)](https://github.com/gigony](https://github.com/gigony))
- Fix thread contention in Cache ([#145](https://github.com/rapidsai/cucim/pull/145)) [[@gigony](https://github.com/gigony)](https://github.com/gigony](https://github.com/gigony))
- Build with NumPy 1.17 ([#148](https://github.com/rapidsai/cucim/pull/148)) [[@jakirkham](https://github.com/jakirkham)](https://github.com/jakirkham](https://github.com/jakirkham))

## 📖 Documentation

- Add Jupyter notebook for SVS Support ([#147](https://github.com/rapidsai/cucim/pull/147)) [[@gigony](https://github.com/gigony)](https://github.com/gigony](https://github.com/gigony))
- Update change log for v21.10.01 ([#142](https://github.com/rapidsai/cucim/pull/142)) [[@gigony](https://github.com/gigony)](https://github.com/gigony](https://github.com/gigony))
- update docs theme to pydata-sphinx-theme ([#138](https://github.com/rapidsai/cucim/pull/138)) [[@quasiben](https://github.com/quasiben)](https://github.com/quasiben](https://github.com/quasiben))
- Update Github links in README.md through script ([#132](https://github.com/rapidsai/cucim/pull/132)) [[@gigony](https://github.com/gigony)](https://github.com/gigony](https://github.com/gigony))
- Fix GDS link in Jupyter notebook ([#131](https://github.com/rapidsai/cucim/pull/131)) [[@gigony](https://github.com/gigony)](https://github.com/gigony](https://github.com/gigony))
- Update notebook for the interoperability with DALI ([#127](https://github.com/rapidsai/cucim/pull/127)) [[@gigony](https://github.com/gigony)](https://github.com/gigony](https://github.com/gigony))



## 🛠️ Improvements

- Update `conda` recipes for Enhanced Compatibility effort by ([#164](https://github.com/rapidsai/cucim/pull/164)) [[@ajschmidt8](https://github.com/ajschmidt8)](https://github.com/ajschmidt8](https://github.com/ajschmidt8))
- Fix Changelog Merge Conflicts for `branch-21.12` ([#156](https://github.com/rapidsai/cucim/pull/156)) [[@ajschmidt8](https://github.com/ajschmidt8)](https://github.com/ajschmidt8](https://github.com/ajschmidt8))
- Add cucim.kit.cumed plugin with skeleton ([#129](https://github.com/rapidsai/cucim/pull/129)) [[@gigony](https://github.com/gigony)](https://github.com/gigony](https://github.com/gigony))
- Update initial cpp unittests ([#128](https://github.com/rapidsai/cucim/pull/128)) [[@gigony](https://github.com/gigony)](https://github.com/gigony](https://github.com/gigony))
- Optimize zoom out implementation with separate padding kernel ([#125](https://github.com/rapidsai/cucim/pull/125)) [[@chirayuG-nvidia](https://github.com/chirayuG-nvidia)](https://github.com/chirayuG-nvidia](https://github.com/chirayuG-nvidia))
- Do not force install linux-64 version of openslide-python ([#124](https://github.com/rapidsai/cucim/pull/124)) [[@Ethyling](https://github.com/Ethyling)](https://github.com/Ethyling](https://github.com/Ethyling))

# cuCIM 21.10.00 (7 Oct 2021)

## 🐛 Bug Fixes

- fix failing regionprops test cases ([#110](https://github.com/rapidsai/cucim/pull/110)) [@grlee77](https://github.com/grlee77)

## 📖 Documentation

- Forward-merge branch-21.08 to branch-21.10 ([#88](https://github.com/rapidsai/cucim/pull/88)) [@jakirkham](https://github.com/jakirkham)
- Update PyPI cuCIM v21.08.01 README.md and CHANGELOG.md ([#87](https://github.com/rapidsai/cucim/pull/87)) [@gigony](https://github.com/gigony)

## 🚀 New Features

- Support raw RGB tiled TIFF ([#108](https://github.com/rapidsai/cucim/pull/108)) [@gigony](https://github.com/gigony)
- Add a mechanism for user to know the availability of cucim.CuImage ([#107](https://github.com/rapidsai/cucim/pull/107)) [@gigony](https://github.com/gigony)
- Enable GDS and Support Runtime Context (__enter__, __exit__) for CuFileDriver and CuImage ([#106](https://github.com/rapidsai/cucim/pull/106)) [@gigony](https://github.com/gigony)
- Add transforms for Digital Pathology ([#100](https://github.com/rapidsai/cucim/pull/100)) [@shekhardw](https://github.com/shekhardw)

## 🛠️ Improvements

- ENH Replace gpuci_conda_retry with gpuci_mamba_retry ([#69](https://github.com/rapidsai/cucim/pull/69)) [@dillon-cullinan](https://github.com/dillon-cullinan)

# cuCIM 21.08.00 (4 Aug 2021)

## 🐛 Bug Fixes

- Remove int-type bug on Windows in skimage.measure.label ([#72](https://github.com/rapidsai/cucim/pull/72)) [@grlee77](https://github.com/grlee77)
- Fix missing array interface for associated_image() ([#65](https://github.com/rapidsai/cucim/pull/65)) [@gigony](https://github.com/gigony)
- Handle zero-padding version string ([#59](https://github.com/rapidsai/cucim/pull/59)) [@gigony](https://github.com/gigony)
- Remove invalid conda environment activation ([#58](https://github.com/rapidsai/cucim/pull/58)) [@ajschmidt8](https://github.com/ajschmidt8)

## 📖 Documentation

- Fix a typo in cache document ([#66](https://github.com/rapidsai/cucim/pull/66)) [@gigony](https://github.com/gigony)

## 🚀 New Features

- Pin `isort` hook to 5.6.4 ([#73](https://github.com/rapidsai/cucim/pull/73)) [@charlesbluca](https://github.com/charlesbluca)
- Add skimage.morphology.thin ([#27](https://github.com/rapidsai/cucim/pull/27)) [@grlee77](https://github.com/grlee77)

## 🛠️ Improvements

- Add SciPy 2021 to README ([#79](https://github.com/rapidsai/cucim/pull/79)) [@jakirkham](https://github.com/jakirkham)
- Use more descriptive ElementwiseKernel names in cucim.skimage ([#75](https://github.com/rapidsai/cucim/pull/75)) [@grlee77](https://github.com/grlee77)
- Add initial Python unit/performance tests for TIFF loader module ([#62](https://github.com/rapidsai/cucim/pull/62)) [@gigony](https://github.com/gigony)
- Fix `21.08` forward-merge conflicts ([#57](https://github.com/rapidsai/cucim/pull/57)) [@ajschmidt8](https://github.com/ajschmidt8)

# cuCIM 21.06.00 (9 Jun 2021)

## 🐛 Bug Fixes

- Update `update-version.sh` ([#42](https://github.com/rapidsai/cucim/pull/42)) [@ajschmidt8](https://github.com/ajschmidt8)

## 🛠️ Improvements

- Update environment variable used to determine `cuda_version` ([#43](https://github.com/rapidsai/cucim/pull/43)) [@ajschmidt8](https://github.com/ajschmidt8)
- Update version script to remove bump2version dependency ([#41](https://github.com/rapidsai/cucim/pull/41)) [@gigony](https://github.com/gigony)
- Update changelog ([#40](https://github.com/rapidsai/cucim/pull/40)) [@ajschmidt8](https://github.com/ajschmidt8)
- Update docs build script ([#39](https://github.com/rapidsai/cucim/pull/39)) [@ajschmidt8](https://github.com/ajschmidt8)

# cuCIM 0.19.0 (15 Apr 2021)

- Initial release of cuCIM including cuClaraImage and [cupyimg](https://github.com/mritools/cupyimg).
