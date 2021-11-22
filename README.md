# Instructions for Developers

### Building and Deploying
Please follow the instructions as described here.
```bash
# optional, only once: create a new conda build environment
conda create --name build_clean conda-build anaconda bump2version -c conda-forge
conda activate build_clean

# bump version so that all versions get updated automatically, creates a git version tag automatically
bump2version [major/minor/patch/release/build]  # --verbose --dry-run to see the effect

# upload git tag
git push --tags

# build wheels
python setup.py bdist_wheel
# edit git release and upload the wheels

# conda release
cd conda
conda-build [-c channels] .

anaconda upload -u [your username] [path as provided at the end of the conda-build output]
```
