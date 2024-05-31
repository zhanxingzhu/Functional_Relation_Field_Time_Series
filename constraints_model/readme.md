1. bintree_nodes.py
    constraint_models/bintree_nodes.py is the constraint nodes, the first column is the target node set and the other columns are the relevant node set.
    run `model/res_predict.py --data bintree --nodes 255 --thresh 0.01` to generate the relation function. And, the model will be saved in `constraint_models/bintree_cnet.pth`

2. miniapp1_nodes.py
    miniapp1_nodes.py is the constraint nodes for miniapp1 dataset.
    run `model/res_predict.py --data miniapp1 --K 6 --nodes 30 --thresh 0.01` to generate the relation function. And, the model will be saved in `constraint_models/miniapp1_cnet.pth`

2. miniapp2_nodes.py
    miniapp2_nodes.py is the constraint nodes for miniapp2 dataset.
    run `model/res_predict.py --data miniapp2 --K 6 --nodes 21 --thresh 0.01` to generate the relation function. And, the model will be saved in `constraint_models/miniapp2_cnet.pth`