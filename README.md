# Functional Relation Field: A Model-Agnostic Framework for Multi-variate Time Series Forecasting
## Inrtoduction
Functional Relation Field (FRF) is a model-agnostic framework to enhance the multivariate time series forecasting with a new inductive bias. Specifically, we first train a neural network with a selected target node as its output and all the other nodes as dependent variables (i.e. the input of this neural network), and identify the most relevant dependent nodes based on this trained network. We then re-train it to learn the relationship among the target and the discovered relevant nodes. Next, we incorporate these functional constraints into the network backbones by imposing them to the predicted output during both training and test process.

This work has been published in Artificial Intelligence 2024, please cite: 

@article{li2024functional,
  abbr={AI Journal},
  title={Functional Relation Field: A Model-Agnostic Framework for Multivariate Time Series Forecasting},
  author={Li, Ting and Yu, Bing and Li, Jianguo and Zhu, Zhanxing},
  journal={Artificial Intelligence},
  year={2024}
}

The complete code including three kinds of backbones (SCINet, Autoformer and FEDformer) will be released later.
  
## Running
- Run commands
  - run the STGCN backbone `python Run.py --model STGCN --dataset bintree --reg_lambda 0 --proj_times 0`
  - run the FRF-STGCN `python Run.py --model STGCN --dataset bintree --reg_lambda 0.1 --proj_times 10`
  - run the AGCRN backbone `python Run.py --model AGCRN --dataset bintree --reg_lambda 0 --proj_times 0`
  - run the FRF-AGCRN `python Run.py --model AGCRN --dataset bintree --reg_lambda 0.1 --proj_times 10`
  - run the FRF-GTS 
    - git clone the GTS network code from https://github.com/chaoshangcs/GTS.git
    - replace the `supervisor.py` with `model/GTSTrainer.py`, then run the `train.py`

- Explaination of args：  
  - model name `--model AGCRN` or `--model STGCN`；  
  - iteration or not `--iter non`  or `--iter iter` （only used for STGCN）；  
  - dataset name `--dataset bintree` or  `--dataset miniapp1` or  `--dataset miniapp2`；  
  - real constraints (only for bintree) or learned constraints `--constraints real` or `--constraints learn`  

- The hyper-parameters for FRF：   
  - `--reg_lambda` regularization weight for loss minimization transformation；  
  - `--proj_times` output transformation times, as well as the K in the paper.
  
## Learned constraints
- The directory of  **./code/constraints_model/** DATASET_nodes.py is the constraint nodes and relevant nodes set，DATASET_cnet.pth contain the network parameter of constraint network.

- retrain the constraint network：  
  - In **./code/model** , and run `python ./res_predict.py`；  
  - explaination of args
    - `--nodes_file` existing graph structure，if '', then target nodes have relationship with all others in initilize stage.；  
    - `--dataset` dataset name  
    - `--output_nodes_file` file to save constraint graph structures. 
    - `--output_model_file` file to save weights of relation network.  
    - `--K` the number of neighbor nodes；  
    - `--nodes` number of nodes for target dataset；  
    - `--thresh` epsion_error to filter the strong constraint relation；  

  
