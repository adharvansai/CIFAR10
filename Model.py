### YOUR CODE HERE
# import tensorflow as tf
import torch
import torch.nn as nn
import os, time
import numpy as np
from Network import MyNetwork
from ImageUtils import parse_record
from tqdm import tqdm

"""This script defines the training, validation and testing process.
"""

class MyModel(object):

    def __init__(self, configs):
        super(MyModel, self).__init__()
        self.configs = configs
        self.network = MyNetwork(configs).cuda()

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.network.parameters(), lr=self.configs['learning_rate'], momentum=self.configs['momentum'], weight_decay=5e-4)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.1)

    def model_setup(self):
        pass

    def train(self, x_train, y_train, configs):
        self.network.train()
        # Determine how many batches in an epoch
        num_samples = x_train.shape[0]
        num_batches = num_samples // configs['batch_size']

        print('### Training... ###')
        for epoch in range(1, configs['max_epochs'] + 1):
            start_time = time.time()
            # Shuffle
            shuffle_index = np.random.permutation(num_samples)
            curr_x_train = x_train[shuffle_index]
            curr_y_train = y_train[shuffle_index]

            ### YOUR CODE HERE
            # Set the learning rate for this epoch
            # Usage example: divide the initial learning rate by 10 after several epochs
            lr = configs['learning_rate']
            bs = configs['batch_size']
            ### YOUR CODE HERE

            for i in range(num_batches):
                ### YOUR CODE HERE
                # Construct the current batch.
                # Don't forget to use "parse_record" to perform data preprocessing.
                # Don't forget L2 weight decay

                start = i * bs
                end = (i + 1) * bs
                batch_curr_x_train = curr_x_train[start:end, :]
                batch_curr_y_train = curr_y_train[start:end]
                current_batch_pp = []
                for j in range(bs):
                    current_batch_pp.append(parse_record(batch_curr_x_train[j], True))
                current_batch_pp = np.array(current_batch_pp)
                x_train_tensor = torch.cuda.FloatTensor(current_batch_pp)
                y_train_tensor = torch.cuda.LongTensor(batch_curr_y_train)

                outputs = self.network(x_train_tensor)
                loss = self.criterion(outputs, y_train_tensor)


                ### YOUR CODE HERE
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                print('Batch {:d}/{:d} Loss {:.6f}'.format(i, num_batches, loss), end='\r', flush=True)

            duration = time.time() - start_time
            print('Epoch {:d} Loss {:.6f} Duration {:.3f} seconds.'.format(epoch, loss, duration))

            if epoch % configs['save_interval'] == 0:
                self.save(epoch)

    def validate(self, x, y, checkpoint_num_list):
        self.network.eval()
        print('### Test or Validation ###')
        for checkpoint_num in checkpoint_num_list:
            checkpointfile = os.path.join(self.configs['save_dir'], 'model-%d.ckpt' % (checkpoint_num))
            self.load(checkpointfile)

            preds = []
            for i in tqdm(range(x.shape[0])):
            ### YOUR CODE HERE
                img_pp = parse_record(x[i], False)
                img_input = np.expand_dims(img_pp, axis=0)
                x_test_tensor = torch.FloatTensor(img_input)
                pred = self.network(x_test_tensor)
                _, pred = torch.max(pred, 1)
                preds.append(pred)
            ### END CODE HERE

            y = torch.tensor(y)
            preds = torch.tensor(preds)
            print('Test accuracy: {:.4f}'.format(torch.sum(preds == y) / y.shape[0]))

    def evaluate(self, x, y):
        pass

    def predict_prob(self, x):
        pass

    def save(self, epoch):
        checkpoint_path = os.path.join(self.configs['save_dir'], 'model-%d.ckpt' % (epoch))
        os.makedirs(self.configs['save_dir'], exist_ok=True)
        torch.save(self.network.state_dict(), checkpoint_path)
        print("Checkpoint has been created.")

    def load(self, checkpoint_name):
        ckpt = torch.load(checkpoint_name, map_location="cpu")
        self.network.load_state_dict(ckpt, strict=True)
        print("Restored model parameters from {}".format(checkpoint_name))


### END CODE HERE