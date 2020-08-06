import torch
import syft as sy
hook = sy.TorchHook(torch)

class DefaultConfig():
    def __init__(self):
        self.batch_size = 64
        self.test_batch_size = 1000
        self.epochs = 1
        self.lr = 0.01
        self.momentum = 0.5
        self.no_cuda = False
        self.seed = 1
        self.log_interval = 30
        self.save_model = True

        self.train_data_path = r'datasets/train'
        self.test_data_path = r'datasets/test'
        self.load_model_path = r'savedmodels/'
        self.result_path = r'results'

        self.workers = [sy.VirtualWorker(hook, id="1"), sy.VirtualWorker(hook, id="2")]
