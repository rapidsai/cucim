# Getting Started

```{toctree}
:glob:
:hidden:

../notebooks/Basic_Usage.ipynb
../notebooks/Accessing_File_with_GDS.ipynb
../notebooks/File-access_Experiments_on_TIFF.ipynb
../notebooks/Multi-thread_and_Multi-process_Tests.ipynb
../notebooks/Working_with_DALI.ipynb
../notebooks/Working_with_Albumentation.ipynb
```

## Installation

Please download the latest SDK package (`cuCIM-v21.12.00-linux.tar.gz`).

Untar the downloaded file.

```bash
mkdir -p cuCIM-v21.12.00
tar -xzvf cuCIM-v21.12.00-linux.tar.gz -C cuCIM-v21.12.00

cd cuCIM-v21.12.00
```

## Run command

Executing `./run` command would show you available commands:

```bash
./run
```
```
USAGE: ./run [command] [arguments]...

Global Arguments

Command List
    help  ----------------------------  Print detailed description for a given argument (command name)
  Example
    download_testdata  ---------------  Download test data from Docker Hub
    launch_notebooks  ----------------  Launch jupyter notebooks
  Build
    build_train  ---------------------  Build Clara Train Docker image with cuCIM (& OpenSlide)
    build_examples  ------------------  Build cuCIM C++ examples
```

`./run help <command>` would show you detailed information about the command.

```bash
./run help build_train
```
```
Build Clara Train Docker image with cuCIM (& OpenSlide)

Build image from docker/Dockerfile-claratrain

Arguments:
  $1 - docker image name (default:cucim-train)
```

### download_testdata

It downloads test data from DockerHub (`gigony/svs-testdata:little-big`) and make it available at `notebooks/input` folder.

The folder has the following four files.

- `TUPAC-TR-488.svs`
- `TUPAC-TR-467.svs`
- `image.tif`
- `image2.tif`

#### Test Dataset

`TUPAC-TR-488.svs` and `TUPAC-TR-467.svs` are from the dataset
of Tumor Proliferation Assessment Challenge 2016 (TUPAC16 | MICCAI Grand Challenge).

- Website: <http://tupac.tue-image.nl/node/3>
- Data link: <https://drive.google.com/drive/u/0/folders/0B--ztKW0d17XYlBqOXppQmw0M2M>

#### Converted files

- `image.tif` : 256x256 multi-resolution/tiled TIF conversion of TUPAC-TR-467.svs
- `image2.tif` : 256x256 multi-resolution/tiled TIF conversion of TUPAC-TR-488.svs


### launch_notebooks

It launches a **Jupyter Lab** instance so that the user can experiment with cuCIM.

After executing the command, type a password for the instance and open a web browser to access the instance.

```bash
./run launch_notebooks
```

```bash
...
Port 10001 would be used...(http://172.26.120.129:10001)
2021-02-13 01:12:44 $ nvidia-docker run --gpus all -it --rm -v /home/repo/cucim/notebooks:/notebooks -p 10001:10001 cucim-jupyter -c echo -n 'Enter New Password: '; jupyter lab --ServerApp.password="$(python3 -u -c "from jupyter_server.auth import passwd;pw=input();print(passwd(pw));" | egrep 'sha|argon')" --ServerApp.root_dir=/notebooks --allow-root --port=10001 --ip=0.0.0.0 --no-browser
Enter New Password: <password>
[I 2021-02-13 01:12:47.981 ServerApp] dask_labextension | extension was successfully linked.
[I 2021-02-13 01:12:47.981 ServerApp] jupyter_server_proxy | extension was successfully linked.
...
```

### build_train

It builds an image from the Clara Deploy SDK image. The image would install other useful python package as well as cu
CIM wheel file.

`nvcr.io/nvidian/dlmed/clara-train-sdk:v3.1-ga-qa-5` is used and `docker/Dockerfile-claratrain` has the recipe of the image.

You will need to have a permission to access `nvidian/dlmed` group in NGC.

```bash
./run build_train

docker run -it --rm cucim-train /bin/bash
```

### build_examples

It builds C++ examples at `examples/cpp` folder by using `cmake` in `cucim-cmake` image that is built in runtime.

After the execution, it would copy built file into `bin` folder and show how to execute it.

```bash
./run build_examples
```

```bash
...

Execute the binary with the following commands:
  # Set library path
  export LD_LIBRARY_PATH=/ssd/repo/cucim/dist/install/lib:$LD_LIBRARY_PATH
  # Execute
  ./bin/tiff_image notebooks/input/image.tif .
```

Its execution would show some metadata information and create two files -- `output.ppm` and `output2.ppm`.

`.ppm` file can be viewed by `eog` in Ubuntu.
```
$ ./bin/tiff_image notebooks/input/image.tif .
[Plugin: cucim.kit.cuslide] Loading...
[Plugin: cucim.kit.cuslide] Loading the dynamic library from: cucim.kit.cuslide@21.12.00.so
[Plugin: cucim.kit.cuslide] loaded successfully. Version: 0
Initializing plugin: cucim.kit.cuslide (interfaces: [cucim::io::IImageFormat v0.1]) (impl: cucim.kit.cuslide)
is_loaded: true
device: cpu
metadata: {"key": "value"}
dims: YXC
shape: (26420, 19920, 3)
size('XY'): (19920, 26420)
channel_names: (R, G, B)

is_loaded: true
device: cpu
metadata: {"key": "value"}
dims: YXC
shape: (1024, 1024, 3)
size('XY'): (1024, 1024)
channel_names: (R, G, B)
[Plugin: cucim.kit.cuslide] Unloaded.

$ eog output.ppm
$ eog output2.ppm
```
