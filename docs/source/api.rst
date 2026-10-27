~~~~~~~~~~~~~~~~~~~~~~
cuCIM API Reference
~~~~~~~~~~~~~~~~~~~~~~

Clara Submodules
================

.. automodule:: cucim.clara
    :members:
    :undoc-members:

Path and text arguments
-----------------------

File path arguments accept ``str``, ``bytes``, or an ``os.PathLike`` object
such as ``pathlib.Path``. These are the path of ``CuImage(path)``,
``CuImage.save()``, ``filesystem.open()``, ``filesystem.CuFileDriver()``
(``file_path``), and ``filesystem.discard_page_cache()``. A file name that is
not valid UTF-8 can be passed as ``bytes`` or as an ``os.fsdecode()`` string.

Other string arguments, such as ``dim_order``, device names, and the
``filesystem.open()`` flags, accept only ``str``.

.. versionchanged:: 26.12
   These arguments now raise ``TypeError``. Earlier releases accepted them.

   - A ``bytearray`` path.
   - A path that contains an embedded NUL character. Earlier releases cut the
     path at the NUL and opened the shorter path.
   - ``bytes`` or ``bytearray`` for a string argument that is not a path.

``CuImage.path`` and ``repr()`` of a ``CuImage`` decode the path as UTF-8. For
an image opened from a file name that is not valid UTF-8, both raise
``UnicodeDecodeError``.

cache
-----

.. automodule:: cucim.clara.cache
    :members:
    :undoc-members:

filesystem
----------

.. automodule:: cucim.clara.filesystem
    :members:
    :undoc-members:

io
---

.. automodule:: cucim.clara.io
    :members:
    :undoc-members:


core Submodules
===============

color
-----

.. automodule:: cucim.core.operations.color
    :members:
    :undoc-members:

expose
------

.. automodule:: cucim.core.operations.expose
    :members:
    :undoc-members:

intensity
---------

.. automodule:: cucim.core.operations.intensity
    :members:
    :undoc-members:

morphology
----------

.. automodule:: cucim.core.operations.morphology
    :members:
    :undoc-members:

spatial
-------

.. automodule:: cucim.core.operations.spatial
    :members:
    :undoc-members:


skimage Submodules
==================

color
-----

.. automodule:: cucim.skimage.color
    :members:
    :undoc-members:


data
----

.. automodule:: cucim.skimage.data
    :members:
    :undoc-members:

exposure
--------

.. automodule:: cucim.skimage.exposure
    :members:
    :undoc-members:


feature
-------

.. automodule:: cucim.skimage.feature
    :members:
    :undoc-members:


filters
-------

.. automodule:: cucim.skimage.filters
    :members:
    :undoc-members:


measure
-------

.. automodule:: cucim.skimage.measure
    :members:
    :undoc-members:


metrics
-------

.. automodule:: cucim.skimage.metrics
    :members:
    :undoc-members:


morphology
----------

.. automodule:: cucim.skimage.morphology
    :members:
    :undoc-members:


registration
------------

.. automodule:: cucim.skimage.registration
    :members:
    :undoc-members:


restoration
-----------

.. automodule:: cucim.skimage.restoration
    :members:
    :undoc-members:


segmentation
------------

.. automodule:: cucim.skimage.segmentation
    :members:
    :undoc-members:


transform
---------

.. automodule:: cucim.skimage.transform
    :members:
    :undoc-members:


util
----

.. automodule:: cucim.skimage.util
    :members:
    :undoc-members:


Submodule Contents
==================

skimage
-------

.. automodule:: cucim.skimage
    :members:
    :undoc-members:
