###### EL-nivMF as Regularizer over ProxyAnchor.
# NOTE: To evaluate EL-nivMF as standalone objective, simply turn off the ProxyAnchor part by setting --loss_proxyvmf_guidance_w 0.

parent_folder="path_to_parent_folder"
project="W&B project name"
gpu="gpu_id"

#### With simple LR scheduling - for relative comparisons, simply turn off scheduling.
### CUB-200
# R128
python main.py --loss proxyvmf_panc --seed 0 --gpu $gpu --source_path $parent_folder --project $project --log_online --group Cub200_Prob_RN50_dim128 --dataset cub200 \
--n_epochs 250 --tau 25 50 --gamma 0.3 --no_train_metrics --embed_dim 128 \
--loss_proxyvmf_proxylrmulti 5000 --loss_proxyvmf_conclrmulti 20000 --loss_proxyvmf_concentration 6 --loss_proxyvmf_init_norm_multiplier 30 --loss_proxyvmf_temp 0.003 --loss_proxyvmf_templrmulti 200 --loss_proxyvmf_warmstart --loss_proxyvmf_n_samples 5 --loss_proxyvmf_guidance_w 3

# R512
python main.py --loss proxyvmf_panc --seed 0 --gpu $gpu --source_path $parent_folder --project $project --log_online --group Cub200_Prob_RN50_dim512 --dataset cub200\ 
--n_epochs 250 --tau 31 --gamma 0.1 --no_train_metrics --embed_dim 512 \
--loss_proxyvmf_proxylrmulti 5000 --loss_proxyvmf_conclrmulti 20000 --loss_proxyvmf_concentration 18 --loss_proxyvmf_init_norm_multiplier 50 --loss_proxyvmf_temp 0.003 --loss_proxyvmf_templrmulti 200 --loss_proxyvmf_warmstart --loss_proxyvmf_n_samples 5 --loss_proxyvmf_guidance_w 5

### CARS-196
# R128
python main.py --loss proxyvmf_panc --seed 0 --gpu $gpu --source_path $parent_folder --project $project --log_online --group Cars196_Prob_RN50_dim128 --dataset cars196 \
--n_epochs 300 --tau 60 80 --gamma 0.1 --no_train_metrics --embed_dim 128 \
--loss_proxyvmf_proxylrmulti 7500 --loss_proxyvmf_conclrmulti 7500 --loss_proxyvmf_concentration 9 --loss_proxyvmf_init_norm_multiplier 12 --loss_proxyvmf_temp 0.003 --loss_proxyvmf_templrmulti 200 --loss_proxyvmf_warmstart --loss_proxyvmf_n_samples 5 --loss_proxyvmf_guidance_w 0.3

# R512
python main.py --loss proxyvmf_panc --seed 0 --gpu $gpu --source_path $parent_folder --project $project --log_online --group Cars196_Prob_RN50_dim512 --dataset cars196 \
--n_epochs 300 --tau 105 170 --gamma 0.2 --no_train_metrics --embed_dim 512 \
--loss_proxyvmf_proxylrmulti 5000 --loss_proxyvmf_conclrmulti 20000 --loss_proxyvmf_concentration 22 --loss_proxyvmf_init_norm_multiplier 10 --loss_proxyvmf_temp 0.003 --loss_proxyvmf_templrmulti 200 --loss_proxyvmf_warmstart --loss_proxyvmf_n_samples 5 --loss_proxyvmf_guidance_w 0.1

#### Comparing the different distribution-to-distribution and distribution-to-points metrics (reproduce figure 5 of section 4.3)

# EL-nivMF (without PANC, so guidance_w = 0)
python main.py --loss proxyvmf_panc --seed 0 --gpu $gpu --source_path $parent_folder --project $project --log_online --group Cub200_Prob_RN50_dim512 --dataset cub200\
--n_epochs 50 --bs 106 --gamma 0 --no_train_metrics --embed_dim 512 \
--loss_proxyvmf_proxylrmulti 5000 --loss_proxyvmf_conclrmulti 50000 --loss_proxyvmf_concentration 50 --loss_proxyvmf_init_norm_multiplier 150 --loss_proxyvmf_temp 1 --loss_proxyvmf_templrmulti 500 --loss_proxyvmf_warmstart --loss_proxyvmf_n_samples 10 --loss_proxyvmf_guidance_w 0

# EL-vMF (Expected Likelihood, rho = 1)
python main.py --loss proxyvmf --loss_proxyvmf_rho 1 --seed 0 --gpu $gpu --source_path $parent_folder --project $project --log_online --group Cub200_Prob_RN50_dim512 --dataset cub200\
--n_epochs 50 --bs 106 --gamma 0 --no_train_metrics --embed_dim 512 \
--loss_proxyvmf_proxylrmulti 5000 --loss_proxyvmf_conclrmulti 50000 --loss_proxyvmf_concentration 10 --loss_proxyvmf_init_norm_multiplier 150 --loss_proxyvmf_temp 1 --loss_proxyvmf_templrmulti 500 --loss_proxyvmf_warmstart

# B-vMF (Bhattacharyya, rho = 0.5)
python main.py --loss proxyvmf --loss_proxyvmf_rho 0.5 --seed 0 --gpu $gpu --source_path $parent_folder --project $project --log_online --group Cub200_Prob_RN50_dim512 --dataset cub200\
--n_epochs 50 --bs 106 --gamma 0 --no_train_metrics --embed_dim 512 \
--loss_proxyvmf_proxylrmulti 5000 --loss_proxyvmf_conclrmulti 50000 --loss_proxyvmf_concentration 10 --loss_proxyvmf_init_norm_multiplier 150 --loss_proxyvmf_temp 1 --loss_proxyvmf_templrmulti 500 --loss_proxyvmf_warmstart

# KL-vMF (Kullback Leibler)
python main.py --loss proxyvmf_kl --seed 0 --gpu $gpu --source_path $parent_folder --project $project --log_online --group Cub200_Prob_RN50_dim512 --dataset cub200\
--n_epochs 50 --bs 106 --gamma 0 --no_train_metrics --embed_dim 512 \
--loss_proxyvmf_proxylrmulti 5000 --loss_proxyvmf_conclrmulti 50000 --loss_proxyvmf_concentration 10 --loss_proxyvmf_init_norm_multiplier 150 --loss_proxyvmf_temp 1 --loss_proxyvmf_templrmulti 500 --loss_proxyvmf_warmstart

# nivMF (images are point embeddings, nivMFs as proxies)
python main.py --loss proxynca_nivmf --seed 0 --gpu $gpu --source_path $parent_folder --project $project --log_online --group Cub200_Prob_RN50_dim512 --dataset cub200\
--n_epochs 50 --bs 106 --gamma 0 --no_train_metrics --embed_dim 512 \
--loss_proxyvmf_proxylrmulti 5000 --loss_proxyvmf_conclrmulti 50000 --loss_proxyvmf_concentration 10 --loss_proxyvmf_init_norm_multiplier 150 --loss_proxyvmf_temp 0.03125 --loss_proxyvmf_templrmulti 500 --loss_proxyvmf_warmstart

# L2
python main.py --loss proxynca_l2 --seed 0 --gpu $gpu --source_path $parent_folder --project $project --log_online --group Cub200_Prob_RN50_dim512 --dataset cub200\
--n_epochs 50 --bs 106 --gamma 0 --no_train_metrics --embed_dim 512 \
--loss_proxyvmf_proxylrmulti 5000 --loss_proxyvmf_conclrmulti 50000 --loss_proxyvmf_concentration 200 --loss_proxyvmf_init_norm_multiplier 150 --loss_proxyvmf_temp 1 --loss_proxyvmf_templrmulti 500 --loss_proxyvmf_warmstart

# Cos
python main.py --loss proxynca_cos --seed 0 --gpu $gpu --source_path $parent_folder --project $project --log_online --group Cub200_Prob_RN50_dim512 --dataset cub200\
--n_epochs 50 --bs 106 --gamma 0 --no_train_metrics --embed_dim 512 \
--loss_proxyvmf_proxylrmulti 5000 --loss_proxyvmf_conclrmulti 50000 --loss_proxyvmf_concentration 10 --loss_proxyvmf_init_norm_multiplier 150 --loss_proxyvmf_temp 0.00390625 --loss_proxyvmf_templrmulti 500 --loss_proxyvmf_warmstart
