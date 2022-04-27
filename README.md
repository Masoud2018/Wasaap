


## Wasaap

<p align="center">
  <img src="/shot(1).png" />
</p>

Wasaap is a Python 3.6+ library for wavefront sensing with 
Hartmann Wavefront Sensors (HWS). It supports the centroids technique
[**B. Keitel, et. al., _J. Synch. Rad (2016)_**] (https://doi.org/10.1107/S1600577515020354) and 
the Fourier demodulation [**Y. Carmon, _Appl. Phys. Lett. (2004)_**] (https://doi.org/10.1063/1.1759770)
to cope with low and high numerical aperture wavefronts.
_Wasaap_ is featured by reconstructing discontinuous wavefronts 
within a part of measured intensities is blocked due to instrument or detector design.
_Wasaap_ supporst wavefront propagation in the near, intermediare and far field


Due to its composable structure, it plays well with 
other packages and can be integrated easily. _Wasaap_ outputs the complex wave field that
can be used as the input of any propagation based codes. 
A input must be as '.npy' formatted. The package calls all the data 
in the data folder continuously. For a massive run,
the display option can be switched off to speed up the code performance.





## Installation



We strongly recommend to run the package on a virtual environment


```bash
    python3 -m venv /path/to/new/virtual/environment [wfs_env]
    source wfs_env/bin/activate
```

The required libraries can be installed as follwoing:


```bash
    pip3 install --upgrade pip
    pip3 install -r requirement_gen.txt
    cat requirement_util.txt | xargs -n 1 pip install
```  
## Usage/Examples


A sample of data presented in our paper is given in the _data_ folder. 
The main interface _wasaap.py_ automatically loads the input file and
produces the figures of the paper. 

```python
    python3 wasaap.py
```

The sample is a hartmangramm measured at a soft X-ray wavelength 
with a Schwarzschild Objective (SO). The sample intensity is magnified,
blocked and has an annular shape. 
## Authors

- [@masoud_mehrjoo](https://www.linkedin.com/in/masoud-mehrjoo-232384141/)

