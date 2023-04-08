"""To train a video fight classifier."""

import torch
import logging
from torch import Generator
from utils.load_args import get_args
from model.resnet50_lstm import create_model
from utils.data import get_dataframe, VideoDataset
from train.fit import fit_model

logging.basicConfig(
    filename="runing.log",
    filemode="a",
    level=logging.INFO,
    format="%(asctime)s-%(name)s-%(levelname)s-%(message)s",
)

if __name__ == "__main__":
    args = get_args()
    NUM_FEAT = 12800
    RNN_HID_SIZE = 30
    RNN_NUM_LAYERS = 2
    DR_RATE= 0.2
    NUM_CLASSES = 1
    BATCH_SIZE = args.batchsize
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        df = get_dataframe(args.data_path)
    except FileNotFoundError:
        logging.error(f"No such file or directory: {args.data_path}")
    dataset = VideoDataset(df)
    train_set, val_set, test_set = torch.utils.data.random_split(dataset, [140, 30, 30], 
                                                                 generator=Generator().manual_seed(42))
    
    trainloader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    validloader = torch.utils.data.DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=True)
    testloader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False)

    model = create_model(NUM_FEAT, RNN_HID_SIZE, RNN_NUM_LAYERS, DR_RATE, NUM_CLASSES)
    
    logging.info("Start training...")
    fit_model(model, 
              trainloader, 
              validloader, 
              epochs=args.epochs, 
              lr=args.lr, 
              weights_path=args.wgts_path, 
              device=DEVICE)
    logging.info("Finished!")