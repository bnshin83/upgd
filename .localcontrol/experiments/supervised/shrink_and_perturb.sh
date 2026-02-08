EXP_NAME="snp"
EXP_TYPE="supervised"
JOB_NAME="snp_baselines"

DATASETS=("label_permuted_emnist_stats" "label_permuted_cifar10_stats" "label_permuted_mini_imagenet_stats" "input_permuted_mnist_stats")
DATASET_NAMES=("emnist" "cifar10" "mini_imagenet" "imnist")
NUM_SEEDS=5

WANDB_PROJECT="upgd"
WANDB_ENTITY="shin283-purdue-university"

run_experiment() {
    # Per-dataset hyperparameters
    case $DATASET_IDX in
        0) LR=0.005; WEIGHT_DECAY=0.01; SIGMA=0.001 ;;  # EMNIST
        1) LR=0.005; WEIGHT_DECAY=0.01; SIGMA=0.001 ;;  # CIFAR-10
        2) LR=0.005; WEIGHT_DECAY=0.01; SIGMA=0.001 ;;  # Mini-ImageNet
        3) LR=0.005; WEIGHT_DECAY=0.01; SIGMA=0.1 ;;    # Input-MNIST
    esac

    python3 core/run/run_stats_with_curvature.py \
        --task ${TASK} \
        --learner shrink_and_perturb \
        --seed ${SEED} \
        --lr ${LR} \
        --weight_decay ${WEIGHT_DECAY} \
        --sigma ${SIGMA} \
        --network fully_connected_relu_with_hooks \
        --n_samples 1000000 \
        --compute_curvature_every 1000000 \
        --save_path logs
}
