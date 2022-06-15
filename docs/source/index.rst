
Welcome to cuCIM's documentation!
====================================
cuCIM (Compute Unified Device Architecture Clara IMage) is an open-source, accelerated computer vision and image processing software library for multidimensional images used in biomedical, geospatial, material and life science, and remote sensing use cases.

cuCIM provides GPU-accelearted I/O,
computer vision and image processing primitives for N-Dimensional images including:
*color conversion, 
*exposure, 
*feature extraction, 
*filters, 
*measure, 
*metrics, 
*morphology, 
*registration, 
*restoration, 
*segmentation, and 
*transforms.   

cuCIM supports the following formats:
*Aperio ScanScope Virtual Slide (SVS), 
*Deflated Compressed, 
*Red Green Blue (RGB), 
*Generic TIFF, 
*JPEG/JPEG 2000 Compressed RGB,
*Lempel-Ziv-Welch (LZW) RGB,
*Philips TIFF, and
*Tiled, Multi–Resolution TIFF. 

Our API mirrors `scikit-image
<https://scikit-image.org/>`_ for image manipulation and `OpenSlide
<https://openslide.org/>`_ for image loading.

cuCIM is interoperable with the following workflows: 
*Albumentations, 
*cuPY, 
*Data Loading Library (DALI), 
*JFX, 
*MONAI, 
*Numba, 
*NumPy, 
*PyTorch, 
*Tensorflow, and
*Triton.

cuCIM is fully open sourced under the Apache-2.0 license, and the Clara and RAPIDS teams welcomes new and seasoned
contributors, users and hobbyists! You may download cuCIM via Anaconda [Conda](https://anaconda.org/rapidsai-nightly/cucim) or [PyPi](https://pypi.org/project/cucim/) Thank you for your wonderful support! Below, we provide some resources to help get you started. 

**Blogs**
[Accelerating Scikit-Image API with cuCIM: n-Dimensional Image Processing and IO on GPUs](https://developer.nvidia.com/blog/cucim-rapid-n-dimensional-image-processing-and-i-o-on-gpus/)
[Accelerating Digital Pathology Pipelines with NVIDIA Clara™ Deploy](https://developer.nvidia.com/blog/accelerating-digital-pathology-pipelines-with-nvidia-clara-deploy-2/)

**Webinars**
[cuCIM: a GPU Image IO and Processing Library](https://www.youtube.com/watch?v=G46kOOM9xbQ)

.. toctree::
   :maxdepth: 4
   :caption: Contents:

   api.rst

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
