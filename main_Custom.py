import torch

from common.etc import print_gpu_info
from common.plot import confusion_matrix, accuracy_graph, loss_graph
from common.trainer import Trainer
from data.load_data import load_data
from models.Custom import Custom

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print_gpu_info(device)

    # Hyper Parameters - Fix
    N = 64000
    BATCH_SIZE = 64
    CONV_FILTER_SIZE = 3
    CONV_STRIDE_SIZE = 1

    # Hyper Parameters - Flex
    LR = 0.0001
    EPOCHS = 30
    HIDDEN_SIZE = 100
    DROPOUT = True
    BATCH_NORM = True

    CONV_NUM = 2
    # conv layer 1
    CONV1_FILTER_NUM = 15
    PADDING1 = 0

    # conv layer 2
    CONV2_FILTER_NUM = 15
    PADDING2 = 0

    # Load data
    datas = load_data(N_=N, print_=False)

    # RGB, 48*48

    network = Custom(
        conv1_param={
            "filter_num": CONV1_FILTER_NUM,
            "filter_size": CONV_FILTER_SIZE,
            "pad": PADDING1,
            "stride": CONV_STRIDE_SIZE,
        },
        conv2_param={
            "filter_num": CONV2_FILTER_NUM,
            "filter_size": CONV_FILTER_SIZE,
            "pad": PADDING2,
            "stride": CONV_STRIDE_SIZE,
        },
        hidden_size=HIDDEN_SIZE,
        weight_init_std=0.01,
        device=device,
        dropout=DROPOUT,
        batch_norm=BATCH_NORM,
        visualize=True,
    )

    trainer = Trainer(
        network=network,
        datas=datas,
        epochs=EPOCHS,
        mini_batch_size=BATCH_SIZE,
        optimizer="Adam",
        optimizer_param={"lr": LR},
        evaluate_sample_num_per_epoch=1000,
    )
    trainer.train()

    # Accuracy Graph
    accuracy_graph(trainer, EPOCHS)

    confusion_matrix(trainer)

    loss_graph(trainer, EPOCHS)
