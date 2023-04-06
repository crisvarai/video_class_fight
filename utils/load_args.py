import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default="./Peliculas_New/")
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batchsize', type=int, default=3)
    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument('--wgts_path', type=str, default="./weights/best_model_py.pth")
    args = parser.parse_args()
    return args