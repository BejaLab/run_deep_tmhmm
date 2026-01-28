run\_deep\_tmhmm
================

DeepTMHMM wrapper. Use as follows:

1. Download DeepTMHMM from https://dtu.biolib.com/DeepTMHMM manually
2. Create an environment with python 3.8 and torch and install this package in it. E.g. with the following conda environment:

```
$ cat env.yaml
dependencies:
 - python=3.8
 - pip
 - pip:
   - --extra-index-url https://download.pytorch.org/whl/cu113
   - torch==1.12.1+cu113
   - git+https://github.com/BejaLab/run_deep_tmhmm
$ mamba env create -f env.yaml -n deep_tmhmm
$ conda activate deep_tmhmm
```
3. Initialize the data with `deep_tmhmm_init -D PATH_TO_DEEP_TMHMM_DATA`
4. Run the inference with `deep_tmhmm_run -D PATH_TO_DEEP_TMHMM_DATA -E PATH_TO_DEEP_TMHMM -i INPUT_FASTA -o OUTPUT_RESULTS`, see `deep_tmhmm_run -h` for details.
