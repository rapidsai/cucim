
# Test Dataset

TUPAC-TR-488.svs and TUPAC-TR-467.svs are from the dataset
of [Tumor Proliferation Assessment Challenge 2016](http://tupac.tue-image.nl/node/3) (TUPAC16 | MICCAI Grand Challenge) which are publicly
available through [The Cancer Genome Atlas (TCGA)](https://www.cancer.gov/about-nci/organization/ccg/research/structural-genomics/tcga) under [CC BY 3.0 License](https://creativecommons.org/licenses/by/3.0/).

- Website: http://tupac.tue-image.nl/node/3
- Data link: https://drive.google.com/drive/u/0/folders/0B--ztKW0d17XYlBqOXppQmw0M2M

## Converted files

- image.tif : 256x256 multi-resolution/tiled TIF conversion of TUPAC-TR-467.svs
- image2.tif : 256x256 multi-resolution/tiled TIF conversion of TUPAC-TR-488.svs

## How to download test image files

Execute the following command from the cuCIM's repository root folder:

```bash
./run download_testdata
```