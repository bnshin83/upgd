EXP_NAME="adam"
EXP_TYPE="supervised"
JOB_NAME="adam_baselines"

DATASETS=("label_permuted_emnist_stats" "label_permuted_cifar10_stats" "label_permuted_mini_imagenet_stats" "input_permuted_mnist_stats")
DATASET_NAMES=("emnist" "cifar10" "mini_imagenet" "imnist")
NUM_SEEDS=5

WANDB_PROJECT="upgd"
WANDB_ENTITY="shin283-purdue-university"

run_experiment() {
    python3 core/run/run_stats_with_curvature.py \
        --task ${TASK} \
        --learner adam \
        --seed ${SEED} \
        --lr 0.0001 \
        --weight_decay 0.1 \
        --beta1 0.0 \
        --beta2 0.9999 \
        --eps 1e-08 \
        --network fully_connected_relu_with_hooks \
        --n_samples 1000000 \
        --compute_curvature_every 1000000 \
        --save_path logs
}
