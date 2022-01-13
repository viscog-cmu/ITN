---   
<div align="center">    
 
# Interactive Topographic Network modeling framework 

[![Paper](https://img.shields.io/badge/PNAS-2022-brightgreen)](https://www.pnas.org/content/119/3/e2112566119)
</div>
 
## Description   
- Code for training and evaluating Interactive Topographic Network models of visual cortical topography

### Citation   
```
@article{Blauch2022Connectivity,
  title={A connectivity-constrained computational account of topographic organization in primate high-level visual cortex},
  author={Nicholas M. Blauch, Marlene Behrmann, David C. Plaut},
  journal={Proceedings of the National Academy of Sciences},
  year={2022},
  issue={199 (3)},
}
```   

## Setup 

- Setup an appropriate virtual environment on a CUDA-ready machine using `requirements.txt`. with `conda`:
```
conda create --name topographic python=3.6.10
conda activate topographic
conda install cudatoolkit=10.1 -c pytorch # if on linux or otherwise needed
pip install -r requirements.txt
# optionally pip install the topographic module 
python setup.py install develop
```
- modify your `.bashrc` file to set the environment variable `SERVER_NAME`. and then configure directories as appropriate using `toographic/config.py`
```
export SERVERNAME='nick-local'
```

- Configure use of SLURM for job submission if planning to train models (local use is also possible). Other HPC systems are possible but will require separate configuration.
 - If using SLURM, edit the file ```run_slurm.sbatch``` to adjust for your particular setup. 

- Acquire ImageNet, VGGFace2, and Places365 from common web sources



## Pre-train encoders
```
. scripts/train_encoders.sh
```  
## Train ITN models
```
. scripts/train_all_itns.sh
```

## Data on KiltHub
- Alternatively to running the models yourself, you can download all of the pre-trained and fine-tuned models along with relevant results necessary for all plots in the paper on Kilthub: https://doi.org/10.1184/R1/17131319

## Plot the results
View the jupyter notebooks; we recommend using Jupyter Lab (which can be installed and launched with:)
``` 
conda install -c conda-forge jupyterlab
jupyter-lab
```
You must have your paths configured and results produced, either through running the experiments yourself, or downloading the KiltHub data and configuring `topographic.config.DATA_DIR` to point to the directory holding its data. 

`notebooks/main_model.ipynb` shows how to visualize in depth results of the main model and other variants
`notebooks/variants.ipynb` shows how to compare ITN variants, with different architectures and wiring penalties ($\lambda_w$)

## questions and bugs: 
create an issue. if a bug, please describe in detail what you have tried and any steps necessary to reproduce the issue. 
