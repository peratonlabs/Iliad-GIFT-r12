# r12


This repository contains code developed by the Perspecta Labs/Project Iliad "GIFT" team for the [IARPA/TrojAI program](https://pages.nist.gov/trojai/docs/about.html). 
Code was developed by Todd Huster and Emmanuel Ekwedike. 

Contact: thuster@peraton.com

## Env Setup

```
conda env create -f environment_trojai12.yml
conda activate trojai12
```


## Run Inference
Inference on round 12 models:
```
python entrypoint.py infer --model_filepath ./data/cyber-pdf-dec2022-train/models/id-00000001/model.pt --examples_dirpath ./data/cyber-pdf-dec2022-train/models/id-00000001/clean-example-data --gift_basepath ./
```

## Run Calibration
Calibration on round 12 models:
```
python entrypoint.py configure 
```
