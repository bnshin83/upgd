EXP_NAME="si"
EXP_TYPE="supervised"
JOB_NAME="si_baselines"

DATASETS=("label_permuted_emnist_stats" "label_permuted_cifar10_stats" "label_permuted_mini_imagenet_stats" "input_permuted_mnist_stats")
DATASET_NAMES=("emnist" "cifar10" "mini_imagenet" "imnist")
NUM_SEEDS=5

WANDB_PROJECT="upgd"
WANDB_ENTITY="shin283-purdue-university"

run_experiment() {
    python3 core/run/run_stats_with_curvature.py \
        --task ${TASK} \
        --learner si \
        --seed ${SEED} \
        --lr 0.001 \
        --beta_weight 0.999 \
        --beta_importance 0.9 \
        --lamda 1.0 \
        --network fully_connected_relu_with_hooks \
        --n_samples 1000000 \
        --compute_curvature_every 1000000 \
        --save_path logs
}
