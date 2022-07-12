# 1. Dataset
https://www.kaggle.com/code/kerneler/starter-ip-network-traffic-flows-49778383-1/data

# 2. Workflow
- Extract suspected samples from data using k-NN,
- test AutoEncoder against anomalous data, assuming that k-NN's decision has been correct,
- compare results of three models: k-NN, AutoEncoder and ensemble IsolationForest