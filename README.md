# Chemical Checker Repository 

The **Chemical Checker (CC)** is a resource of small molecule signatures. In the CC, compounds are described from multiple viewpoints, spanning every aspect of the drug discovery pipeline, from chemical properties to clinical outcomes.

* For a quick exploration of what this resource enables, please visit the [CC web app](http://chemicalchecker.org).
* For full documentation of the python package, please see the [Documentation](http://packages.sbnb-pages.irbbarcelona.org/chemical_checker).
* Concepts and methods are best described in the original CC publication, [Duran-Frigola et al. 2019](https://biorxiv.org/content/10.1101/745703v1).
* For more information about this repository, discussion, notes, etc... please refer to our [Wiki page](http://gitlabsbnb-old.irbbarcelona.org/packages/chemical_checker/wikis/home).

The **Chemical Checker Repository** holds the current implementation of the CC in our `SB&NB` laboratory. As such, the repository contains a significant number of functionalities and data not presented in the primary CC manuscript. The repository follows this directory structure:

* `container`: Deal with containerization of the CC, contains the definition files for Singularity image.
* `notebook`: Contains exemplary Jupyter Notebooks that showcase some CC features.
* `package`: The backbone of the CC in form of a Python package.
* `pipelines`: The pipeline scripts (e.g. for updating the CC, generating data for the web app, etc...).
* `setup`: The setup script to install the CC.


Due to the strong computational requirements of our pipeline, the code has been written and optimized to work in our local HPC facilities. Installation guides found below are mainly addressed to `SB&NB` users. As stated in the manuscript, the main deliverable of our resource are the CC _signatures_, which can be easily accessed:

* through a [REST API](https://chemicalchecker.com/help),
* downloaded as [data files](https://chemicalchecker.com/downloads) or 
* predicted from SMILES with the [Signaturizer](https://github.com/sbnb-irb/signaturizer).

## Chemical Checker `lite`

The CC package can be installed in a couple of minutes directly via `pip`:

```bash
pip install chemicalchecker
```

This installs the `lite` version of the CC that can be used for basic task (e.g. to open signatures) but most of the fancy CC package capabilities will be missing.

_**N.B.** Only bare minimum dependencies are installed along with the package_



# Installation

All the dependencies for the CC will be bundled within a singularity image generated during the installation process.
Generating such an image requires roughly 20 minutes:


1. [Install Singularity (version > 3.6)](https://sylabs.io/guides/3.8/admin-guide/admin_quickstart.html#installation-from-source)

                $ sudo apt-get update && sudo apt-get install -y \
                    build-essential \
                    uuid-dev \
                    libgpgme-dev \
                    squashfs-tools \
                    libseccomp-dev \
                    wget \
                    pkg-config \
                    git \
                    cryptsetup-bin\
                    golang-go

                $ export VERSION=3.8.0 && # adjust this as necessary \
                    wget https://github.com/sylabs/singularity/releases/download/v${VERSION}/singularity-ce-${VERSION}.tar.gz && \
                    tar -xzf singularity-ce-${VERSION}.tar.gz && \
                    cd singularity-ce-${VERSION}

                $ ./mconfig && \
                    make -C ./builddir && \
                    sudo make -C ./builddir install


2. Clone this repository to your code folder, you must use the git lfs (https://docs.github.com/en/repositories/working-with-files/managing-large-files/installing-git-large-file-storage):
        
        cd ~ && mkdir -p code && cd code
        git lfs clone https://github.com/sbnb-irb/chemical-checker.git

3. Run the setup script (this script will require to type your password) with:

        cd chemical_checker && sh setup/setup_chemicalchecker.sh

## Running `Vanilla` Chemical Checker

This is the easiest scenario where you simply use the CC code 'as is'.

The setup_chemicalchecker script has created an alias in your ~/.bashrc so you can start the CC with:
```bash
source ~/.bashrc
chemcheck
```

_**N.B.** If you are using another shell (e.g. zsh) just copy the chemcheck alias from your .bashrc to your .zshrc_


## Running custom Chemical Checker

If you are contributing with code to the CC you can run the singularity image specifying your local develop branch:

```bash
chemcheck -d /path/to/your/code/chemical_checker/package/
```
    
## Running with alternative config file

The CC rely on one config file containing the information for the current enviroment (e.g. the HPC, location of the default CC, database, etc...). The default configuration refere to our `SB&NB` enviroment and must be overridden specifying a custom config file when working in a different enviroment:

```bash
chemcheck -c /path/to/your/cc_config.json
```

## Running with alternative image

You might want to use a previously compiled or downloaded image:

```bash
chemcheck -i /path/to/your/cc_image.simg
```

## Usage

We make it trivial to either start a Jupyter Notebook within the image or to run a shell:

1. Run a Jupyter Notebook with:

        chemcheck

    1.1. Open your browser, paste the URL that the script has produced.

    1.2. Start a new notebook (on the top right Jupyter page click New -> Python )

    1.3. Type `import chemicalchecker`

2. Run a shell within the image:

        chemcheck -s [-d <PATH_TO_SOURCE_CODE_ROOT>] [-c <PATH_TO_CONFIG_FILE>]
        
    2.1 Type `ipython`
    
    2.2 Type `import chemicalchecker`


## Running an update pipeline

When properly configured the CC can be updated or generated from scratch. This operation critically depend on available infrastructure. In our HPC infrastructure comprising 12 nodes and 364 cores it takes roughly 2 weeks to complete and update. PLease check the `pipelines` directory for more details.


# `SB&NB` configuration

## Mounting `/aloy` in Singularity

1. Add bind paths to singularity config file:

        sudo echo "bind path = /aloy/web_checker" >> /usr/local/etc/singularity/singularity.conf


2. Make sure that `/aloy/web_checker` is available on your workstation (e.g. `ls /aloy/web_checker` should give a list of directories) if **not**:

        mkdir /aloy/web_checker
        sudo echo "fs-paloy.irbbarcelona.pcb.ub.es:/pa_webchecker /aloy/web_checker       nfs     defaults,_netdev 0 0" >> /etc/fstab
        sudo mount -a

## Working from a laptop

First, check that you are connected to the `SB&NB` local network:
```bash
ping pac-one-head.irb.pcb.ub.es
```
Then, mount the remote filesystem
```bash
sudo mkdir /aloy
chown <laptop_username>:<laptop_username> /aloy
sshfs <sbnb_username>@pac-one-head.irb.pcb.ub.es:/aloy /aloy
```
You can unmount the filesystem with:
```bash
# Linux
fusermount -u /aloy
# MacOSX
umount /aloy
```

# Contributing

## Introducing new dependencies

### Adding a package or software to the image

1. You will have to enter the singularity sandbox

        cd ~/chemical_checker
        sudo singularity shell --writable sandbox

2. Install the package/software and exit the image

        pip install <package_of_your_dreams>
        exit

3. Re-generate the image:

        rm cc.simg
        sudo singularity build cc.simg sandbox

4. In case you make use of the HPC utility, remember to copy your newly generated image to a directory accessible by the queuing system and edit the config file (section PATH.SINGULARITY_IMAGE) accordingly e.g.:

        cp cc.simg /aloy/scratch/<yout_user>/cc.simg


### Adding a permanent dependency to the package

Not re-inventing the wheel is a great philosophy, but each dependency we introduce comes at the cost of maintainability. Double check that the module you want to add is the best option for doing what you want to do. Check that it is actively developed and that it supports Python 3. Test it thoroughly using the sandbox approach presented above. When your approach is mature you can happily add the new dependency to the package.

To do so you can add a `pip install <package_of_your_dreams>` line to the following files in container/singularity:

* cc_py37.def (unit-testing Python 3 environment)

Don't forget to also add a short comment on why and where this new dependency is used, also in the commit message. E.g. "Added dependency used in preprocessing for space B5.003". The idea is that whenever B5.003 is obsoleted we can also safely remove the dependency.


## Release a new package version

Publication of the package on PyPI is automated in the CI pipeline, however bumping the version and creating a release tag (that triggers publication) is manual and should be performed as follows:

Be sure that all unit tests are passing.

Select the new version number. Consider that it is not possible to re-publish with the same version nor it is possible to reduce it.

Bump (e.g. 1.0.1 -> 1.0.2 or 1.1.0) the version number in the following files:

* [package/chemicalchecker/__init__.py](https://github.com/sbnb-irb/chemical-checker/blob/master/package/chemicalchecker/__init__.py)
* [package/setup.py](https://github.com/sbnb-irb/chemical-checker/blob/master/package/setup.py)

Push these changes.

Create a release tag and push it:

```bash
git tag v1.0.2
git push origin v1.0.2
```

This will trigger CI pipeline to publish the package officially (and definetively) on [PyPI](https://pypi.org/project/chemicalchecker/#history)
