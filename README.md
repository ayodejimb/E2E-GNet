# **A Novel End-to-End Skeleton-based Geometric Deep Neural Network for Human Motion Recognition** 

## Abstract
<div style="text-align: justify"> 
Geometric deep learning has recently gained significant attention in the computer vision and AI community for its ability to capture meaningful representations of data lying in a non-Euclidean space. To this end, we propose E2E-GNet, a novel end-to-end geometric deep neural network for skeleton-based human motion recognition. To enhance the discriminative power between different motions in the non-Euclidean space, E2E-GNet introduces a geometric transformation layer that jointly optimizes skeleton motion sequences on this space and applies a differentiable logarithm map activation to project them onto a linear space. Building on this, we further design a distortion-aware optimization layer that limits skeleton shape distortions caused by this projection, enabling the network to retain discriminative geometric cues and achieve a higher motion recognition rate. We demonstrate the impact of each layer through ablation studies and extensive experiments across five datasets spanning three domains---action recognition, disease analysis, and rehabilitation---show that E2E-GNet outperforms all other state-of-the-art methods on all benchmarks on both performance and cost.
</div>

<div align="center">
    <img src="E2E_GNet.png">
</div>

## Packages and Dependencies
- For packages and dependencies, first create an enviroment using Python, activate the enviroment and run `pip install -r requirements.txt` . We run all our experiments on Python version 3.12.6

## Datasets and Preprocessing
### Action Recongition Datasets (NTU-RDB+D and NTU-RGB+D120)
- Download the dataset (the skeleton files only are sufficient) from the dataset page [here](https://rose1.ntu.edu.sg/dataset/actionRecognition/)
- After downloading the skeleton files, (a) cd into `/NTU_data_preprocessing_folder/data/NTU-RGB+D/`and paste the skeleton files here for NTU-60. For NTU-120, paste the skeleton files into its corresponding folder. (b) Now cd into `E2E-GNet/NTU_data_preprocessing_folder/data_gen/` and run the file `ntu_gen_preprocess60.py` and then the file `generate_2bodies.py` to generate and preprocess the skeleton files. Follow the same step for NTU-120. 
### Disease and Rehabilitation Datasets (EHE, KIMORE and UI-PRMD)  
- These can be obtained from [here](https://github.com/bruceyo/EGCN/tree/master)
- The preprocessing codes are also available on the data webpage. 

## Scripts Organization
- Each dataset folder contains all the scripts for the respective dataset and category.

## Training and Testing 
- To run E2E-GNet for action recognition datasets, cd into `run_files` in the `action_recognition` folder and run the file `E2E_NTU120_DML_glob_in.py` or the file `E2E_NTU60_DML_glob_in.py` for NTU-60 or NTU-120 as the case may be.
- To run E2E-GNet for disease datasets, cd into `run_file_EHE` or `run_file_KIMORE` and run the file `EHE_E2E_DML_glob_h.py` or `KIMORE_E2E_DML_glob_h.py` for EHE or KIMORE as the case may be.
- To run E2E-GNet for rehabilitation datasets, cd into `run_file` and run the file `E2E_DML_glob_h.py` as the case may be.

## Ablation Study Scripts
- Ablation study scripts for running parallel transport (PT) are available in `ablation_files/parallel_transport/run_file_PT/` under the `parallel transport` folder.
- Ablation study scripts for contribution of GTL and DML are available in `/ablation_files/components_contribution/run_files/` under the `components_contribution` folder. 
