# gradunwarp

gradunwarp is a Python/Numpy package used to unwarp the distorted
volumes (due to the gradient field inhomogenities). Currently, it can
unwarp Siemens data.

This is the [Human Connectome Project version of the gradunwarp package][gradunwarp-hcp].

It is forked from a "no longer actively maintained" [gradunwarp package][gradunwarp-ksubramz].

This fork contains changes made for and by the WU-Minn Human Connectome Project consortium ([HCP][HCP])
for use within the [HCP Minimal Preprocessing Pipelines][HCP Pipelines].

## Installation

gradunwarp requires Python 3.8 or newer and can be installed with [uv][uv].
On Windows, uv exposes this console script as `gradient_unwarp`; on POSIX systems, use `gradient_unwarp.py`.

### Install as a CLI tool

Install the latest published package:

```bash
uv tool install gradunwarp
```

Or install directly from a fork:

```bash
uv tool install git+https://github.com/<owner>/gradunwarp.git
```

### Run without installing

```bash
uvx --from gradunwarp gradient_unwarp.py --help
```

On Windows:

```bash
uvx --from gradunwarp gradient_unwarp --help
```

### Developer setup

```bash
uv sync
uv run gradient_unwarp.py --help
uv run pytest
```

On Windows, use `uv run gradient_unwarp --help` for the CLI command.

### Build and publish

```bash
uv build --no-sources
uv publish
```

### Dependencies

* Python (>=3.8)
* [Numpy][Numpy]
* [Scipy][Scipy]
* [nibabel][nibabel] (3.2.1 or later)

## Usage

skeleton

```bash
    gradient_unwarp.py infile outfile manufacturer -g <coefficient file> [optional arguments]
```

typical usage

```bash
    gradient_unwarp.py sonata.mgh testoutson.mgh siemens -g coeff_Sonata.grad  --fovmin -.15 --fovmax .15 --numpoints 40

    gradient_unwarp.py avanto.mgh testoutava.mgh siemens -g coeff_AS05.grad -n
```

### Positional Arguments

The input file (in Nifti or MGH formats) followed by the output file
name (which has the Nifti or MGH extensions -- .nii/.nii.gz/.mgh/.mgz)
followed by the vendor name.

### Required Options

```bash
    -c <coef_file>
    -g <grad_file>
```

The coefficient file (which is acquired from the vendor) is specified
using a `-g` option, to be used with files of type `.grad`.

Or it can be specified using a `-c` in the case you have the `.coef`
file.

These two options are mutually exclusive.

### Other Options

```bash
    -n : If you want to suppress the jacobian intensity correction
    -w : if the volume is to be warped rather than unwarped

    --fovmin <fovmin> : a float argument which specifies the minimum extent of the grid where spherical harmonics are evaluated. (in meters). Default is -.3
    --fovmax <fovmax> : a float argument which specifies the maximum extent of the grid where spherical harmonics are evaluated. (in meters). Default is .3
    --numpoints <numpoints> : an int argument which specifies the number of points in the grid. (in each direction). Default is 60

    --interp_order <order of interpolation> : takes values from 1 to 4. 1 means the interpolation is going to be linear which is a faster method but not as good as higher order interpolations.

    --help : display help
```

## Memory Considerations

gradunwarp tends to use quite a bit of memory because of the intense
spherical harmonics calculation and interpolations performed multiple
times. For instance, it uses almost 85% memory of a 2GB memory 2.2GHz
DualCore system to perform unwarping of a 256^3 volume with 40^3
spherical harmonics grid. (It typically takes 4 to 5 minutes for the
entire unwarping)

Some thoughts:

* Use lower resolution volumes if possible
* Run gradunwarp in a computer with more memory
* Use -numpoints to reduce the grid size. -fovmin and -fovmax can
  be used to move the grid close to your data extents.
* Use non-compressed source volumes. i.e. .mgh and .nii instead of .mgz/.nii.gz
* Recent versions of Python, numpy and scipy

## [HCP][HCP] additions

* slice by slice processing
* x-y flip bug fix
* force 32-bit output in 64-bit systems
* modified for Python3 compatibility

## License

Please see the [Copying.md][Copying.md] file in the distribution.

## Credit

* Jon Polimeni - gradunwarp follows his original MATLAB code
* Karl Helmer - Project Incharge
* Nibabel team

## Note about change history

Some of the changes to this codebase that were made for the HCP, were made when this code
was not yet forked into its own repository. At that time, this modified version of the
gradient unwarping code was embedded in the `src/gradient_unwarping`
subdirectory of the [HCP Pipelines Repository][HCP Pipelines].  

The history (commit comments, changelog, etc. of those changes was not ported to this
repository.  The [HCP Pipelines Repository][HCP Pipelines] will keep that history.  

To get the last version of the [HCP Pipelines Repository][HCP Pipelines] before the
gradient unwarping code was separated, retrieve commit `2e06194921638394c7c0ffd90805fdf06051449a`
To do this, after cloning the [HCP Pipelines Repository][HCP Pipelines] use:

```bash
    git checkout 2e06194921638394c7c0ffd90805fdf06051449a
```

<!-- References -->

[gradunwarp-hcp]: https://github.com/Washington-University/gradunwarp
[gradunwarp-ksubramz]: https://github.com/ksubramz/gradunwarp
[HCP]: http://www.humanconnectome.org
[uv]: https://docs.astral.sh/uv/getting-started/installation/
[Numpy]: http://www.numpy.org
[Scipy]: http://www.scipy.org
[nibabel]: http://nipy.org/nibabel
[Copying.md]: Copying.md
[HCP Pipelines]: https://github.com/Washington-University/Pipelines
