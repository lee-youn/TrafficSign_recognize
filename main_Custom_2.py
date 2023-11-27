import torch

from common.etc import print_gpu_info
from common.plot import confusion_matrix, accuracy_graph, loss_graph
from common.trainer import Trainer
from data.load_data import load_data
from models.Custom2 import Custom2

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print_gpu_info(device)

    # Hyper Parameters - Fix
    N = 64000
    BATCH_SIZE = 64
    CONV_FILTER_SIZE = 3
    CONV_STRIDE_SIZE = 1
    PADDING1 = 1
    PADDING2 = 1
    PADDING3 = 0
    CONV_NUM = 3

    # Hyper Parameters - Flex
    LR = 0.0001
    EPOCHS = 10
    HIDDEN_SIZE = 100
    DROPOUT = True
    BATCH_NORM = True
    CONV1_FILTER_NUM = 15
    CONV2_FILTER_NUM = 15
    CONV3_FILTER_NUM = 15
    DROPOUT_RATIO = [0.3, 0.3, 0.3]

    # Load data
    datas = load_data(N_=N, print_=False)

    network = Custom2(
        conv_param=[
            {
                "filter_num": CONV1_FILTER_NUM,
                "filter_size": CONV_FILTER_SIZE,
                "pad": PADDING1,
                "stride": 1,
            },
            {
                "filter_num": CONV2_FILTER_NUM,
                "filter_size": CONV_FILTER_SIZE,
                "pad": PADDING2,
                "stride": 1,
            },
            {
                "filter_num": CONV3_FILTER_NUM,
                "filter_size": CONV_FILTER_SIZE,
                "pad": PADDING3,
                "stride": 1,
            },
        ],
        dropout_ratio=DROPOUT_RATIO,
        hidden_size=HIDDEN_SIZE,
        weight_init_std=0.01,
        device=device,
        dropout=DROPOUT,
        batch_norm=BATCH_NORM,
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
