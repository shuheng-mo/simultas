<div id="top"></div>

<!-- PROJECT SHIELDS -->
<!--
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->
<!-- using the static badge because it is private, covert to dynamic ones if public  -->
<!-- https://shields.io/#your-badge -->

<div>
<img src="https://img.shields.io/github/issues/acse-sm321/Mogo">
<img src="https://img.shields.io/github/forks/acse-sm321/Mogo">
<img src="https://img.shields.io/github/stars/acse-sm321/Mogo">
<img src="https://img.shields.io/github/license/acse-sm321/Mogo">
</div>

<!-- PROJECT LOGO -->
<div align="center">
  <!-- <a href="https://github.com/othneildrew/Best-README-Template">
    <img src="images/logo.png" alt="Logo" width="80" height="80">
  </a> -->

<h1 align="center">simultas</h1>
  <p align="center">
    <a href="https://docs.taichi-lang.org/docs"><strong>Read the code documentation »</strong></a>
    <br />
    <br />
    <a href="https://github.com/acse-sm321/Mogo">View Demo</a>
    ·
    <a href="https://github.com/acse-sm321/Mogo/issues">Report Bug</a>
    ·
    <a href="https://github.com/acse-sm321/Mogo/issues">Request Feature</a>
  </p>
</div>

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#code-metadata">Code Metadata</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#references">References</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
    <li><a href="#contact">Contact</a></li>
  </ol>
</details>

<!-- ABOUT THE PROJECT -->
## About The Project

<!-- [![Product Name Screen Shot][product-screenshot]](https://example.com) -->

[![Test](https://github.com/acse-sm321/Mogo/workflows/Test/badge.svg)](https://github.com/acse-sm321/Mogo/actions)


All about parallel computing patterns in Python. Providing parallelisation solutions for computing, engineering problems and more. This repository also contains the source code for publications [publication1]() and [publication2](). 

<p align="right">(<a href="#top">BACK TO TOP</a>)</p>

### Code Metadata

This section listed the major frameworks/libraries used to bootstrap this project. Other add-ons/plugins please refer to the acknowledgements section.

* macOS



<p align="right">(<a href="#top">BACK TO TOP</a>)</p>

<!-- GETTING STARTED -->
## Getting Started

Here is an example of how you may give instructions on setting up this project locally. To get a local copy up and running follow these simple example steps.

### Prerequisites

There are some prerequisites before you compile and run the project on local machine or your AI computers. Note that this project built by Python language and relevant packages, add-ons, dependencies.

```
numpy
tensorflow (2.0)
mpi4py
```

### Installation

_Below is an example of how you can instruct your audience on installing and setting up your app. This template doesn't rely on any external dependencies or services._

1. Clone the repo
   ```sh
   $ 
   ```
2. Install required packages / compile
    ```
    $ python3 -m pip install simultas
    ```

3. Trouble shooting & Issues (updating ...)
- To ensure install mpi4py using conda correctly, use `conda create -n ENV_NAME -c conda-forge 'python=3.10.*' openmpi mpi4py`, you may specify your open-sourced MPI version as `mpich` or `openmpi` and Python version according to your needs. This should work for both `Windows` and `macOS`, for `Linux` please build from the source. Test the installation with: 

    ```bash
    conda activate YOUR_ENV_NAME
    mpiexec -n 2048 python -m mpi4py.bench helloworld
    # this should return hello from 2048 processes
    ```

- Sometimes on macOS we might see an initialization error like this:
  ```
  A system call failed during shared memory initialization that should
  not have.  It is likely that your MPI job will now either abort or
  experience performance degradation.

  Local host:  XXXXXX
  System call: XXXXXX
  Error:       No such file or directory (errno 2)
  ```
  to tackle this issue, a quick solution is to use:
  ```bash
  export TMPDIR=/tmp
  ```
  run and activate `.zshrc` might solve this problem.
<p align="right">(<a href="#top">BACK TO TOP</a>)</p>



<!-- USAGE EXAMPLES -->
## Usage

Use this space to show useful examples of how a project can be used. Additional screenshots, code examples and demos work well in this space. You may also link to more resources.

_For more examples, please refer to the [Documentation](https://example.com)_

Code example:
```python
# import the libraries
from simultas.halo_exchange import HaloExchange


## 1. Halo Exchange: see more in examples/ folder
# create halo exchange instance
he = HaloExchange()
_, _, _ =  he.initialization()

problem_mesh = np.array() # or tensor of tensorflow
problem_mesh = he.strctured_halo_update_1D(problem_mesh)


```

<p align="right">(<a href="#top">BACK TO TOP</a>)</p>



<!-- ROADMAP -->
## Roadmap

- [ ] .

See the [open issues](https://github.com/acse-sm321/Mogo/issues) for a full list of proposed features (and known issues).

<p align="right">(<a href="#top">BACK TO TOP</a>)</p>

<!-- CONTRIBUTING -->
## Contributing

Contributions are what make the project such an amazing thing to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/your_feature`)
3. Commit your Changes (`git commit -m 'Add some new feature'`)
4. Push to the Branch (`git push origin feature/your_feature`)
5. Open a Pull Request

<p align="right">(<a href="#top">BACK TO TOP</a>)</p>



<!-- LICENSE -->
## License

Distributed under the GPL-3.0 License. See [`LICENSE.md`](https://github.com/shuheng-mo/simultas/blob/main/LICENSE) for more information.

<p align="right">(<a href="#top">BACK TO TOP</a>)</p>

## References
- 

<!-- ACKNOWLEDGMENTS -->
## Acknowledgments
This project won't born without the help of these wonderful people/coporations:

* Boyang Chen
* Christopher Pain
* Xiaohu Guo


<p align="right">(<a href="#top">BACK TO TOP</a>)</p>

<!-- CONTACT -->
## Contact

Shuheng Mo - [Contact me](https://linktr.ee/shuheng_mo)


<p align="right">(<a href="#top">BACK TO TOP</a>)</p>


