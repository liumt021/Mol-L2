import os
import shutil
import sys
import yaml
import numpy as np
import pandas as pd
import csv
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from load_data import finetuneDataset, DDIDataset
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, accuracy_score, mean_absolute_error, mean_squared_error
torch.multiprocessing.set_sharing_strategy('file_system')
apex_support = False
LOAD_MODEL_NAME = './...'
try:
    sys.path.append('./apex')
    from apex import amp

    apex_support = True
except Exception:
    print("Please install apex for mixed precision training from: https://github.com/NVIDIA/apex")
    apex_support = False
def _save_config_file(model_checkpoints_folder):
    if not os.path.exists(model_checkpoints_folder):
        os.makedirs(model_checkpoints_folder)
        shutil.copy('config_finetune2.yaml', os.path.join(model_checkpoints_folder, 'config_finetune2.yaml'))


def write_csv(path, data, write_type='a'):
    with open(path, write_type, newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(data)


class Normalizer(object):

    def __init__(self, tensor):
        self.mean = torch.mean(tensor)
        self.std = torch.std(tensor)

    def norm(self, tensor):
        return (tensor - self.mean) / self.std

    def denorm(self, normed_tensor):
        return normed_tensor * self.std + self.mean

    def state_dict(self):
        return {'mean': self.mean,
                'std': self.std}

    def load_state_dict(self, state_dict):
        self.mean = state_dict['mean']
        self.std = state_dict['std']

class FineTune:
    def __init__(self, dataset, config, task_name):
        self.config = config
        self.task_name = task_name
        self.device = self._get_device()

        current_time = datetime.now().strftime('%b%d_%H-%M-%S')
        dir_name = current_time + '_' + config['dataset']['target']
        log_dir = os.path.join('finetune', self.task_name, dir_name)
        self.writer = SummaryWriter(log_dir=log_dir)

        self.dataset = dataset
        self.normalizer = None

        if config['dataset']['task'] in ['classification', "ddi"]:
            self.criterion = nn.CrossEntropyLoss()
            self.pre_val = 0.0
        elif config['dataset']['task'] == 'regression':
            self.criterion = nn.MSELoss()
            self.pre_val = 9999.99

        from models.ginet import GINet
        from models.T5MN import T5EncoderProjection
        from models.model import CombinedModel

        model = CombinedModel(
            gnn_model=GINet(),
            t5_model=T5EncoderProjection(config['t5_ckpt_path']),
        ).to(self.device)

        self.model = self._load_pre_trained_weights(model)

    def _get_device(self):
        if torch.cuda.is_available() and self.config['gpu'] != 'cpu':
            device = self.config['gpu']
            torch.cuda.set_device(device)
        else:
            device = 'cpu'
        print("Running on:", device)
        return device

    def _step(self, model, data):
        if self.config['dataset']['task'] == 'ddi':
            data1, data2, label = data
            data1, data2, label = data1.to(self.device), data2.to(self.device), label.to(self.device)
            _, pred = model(data1, data2)
            loss = self.criterion(pred, label)
        else:
            data = data.to(self.device)
            _, pred = model(data)
            if self.config['dataset']['task'] == 'classification':
                loss = self.criterion(pred, data.y.float())
            elif self.config['dataset']['task'] == 'regression':
                if self.normalizer:
                    loss = self.criterion(pred, self.normalizer.norm(data.y))
                else:
                    loss = self.criterion(pred, data.y)
        return loss


    def train(self):
        train_loader, valid_loader, test_loader = self.dataset.get_data_loaders()

        model = self.model
        optimizer = torch.optim.Adam(model.parameters(), lr=self.config['init_lr'], weight_decay=eval(self.config['weight_decay']))

        if apex_support and self.config['fp16_precision']:
            model, optimizer = amp.initialize(model, optimizer, opt_level='O2', keep_batchnorm_fp32=True)

        model_checkpoints_folder = os.path.join(self.writer.log_dir, 'checkpoints')
        _save_config_file(model_checkpoints_folder)

        n_iter = 0
        valid_n_iter = 0

        for epoch_counter in range(self.config['epochs']):
            for bn, data in enumerate(train_loader):
                optimizer.zero_grad()
                loss = self._step(model, data, n_iter)

                if n_iter % self.config['log_every_n_steps'] == 0:
                    self.writer.add_scalar('train_loss', loss, global_step=n_iter)
                    print(epoch_counter, bn, loss.item())

                if apex_support and self.config['fp16_precision']:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()

                optimizer.step()
                n_iter += 1

            if epoch_counter % self.config['eval_every_n_epochs'] == 0:
                valid_loss, valid_metric = self._validate(model, valid_loader, epoch_counter)
                if valid_metric > self.pre_val:
                    self.pre_val = valid_metric
                    torch.save(model.state_dict(), os.path.join(model_checkpoints_folder, 'model.pth'))
                self.writer.add_scalar('validation_loss', valid_loss, global_step=valid_n_iter)
                valid_n_iter += 1

            self._test(model, test_loader, False)
        self._test(model, test_loader, True)


    def _validate(self, model, valid_loader, epoch_counter):
        predictions = []
        labels = []
        valid_loss = 0.0
        num_data = 0
        model.eval()
        with torch.no_grad():
            for data in valid_loader:
                if self.config['dataset']['task'] == 'ddi':
                    data1, data2, label = data
                    data1, data2, label = data1.to(self.device), data2.to(self.device), label.to(self.device)
                    _, pred = model(data1, data2)
                    loss = self.criterion(pred, label)
                    pred = torch.sigmoid(pred)
                    predictions.extend(pred.cpu().numpy())
                    labels.extend(label.cpu().numpy())
                    valid_loss += loss.item() * label.size(0)
                    num_data += label.size(0)
                else:
                    data = data.to(self.device)
                    __, pred = model(data)
                    loss = self._step(model, data, 0)
                    valid_loss += loss.item() * data.y.size(0)
                    num_data += data.y.size(0)
                    if self.normalizer:
                        pred = self.normalizer.denorm(pred)
                    predictions.extend(pred.cpu().detach().numpy())
                    labels.extend(data.y.cpu().flatten().numpy())

        valid_loss /= num_data
        model.train()

        if self.config['dataset']['task'] == 'regression':
            mae = mean_absolute_error(labels, predictions)
            print(f"Epoch {epoch_counter} Validation loss: {valid_loss:.4f} MAE: {mae:.4f}")
            return valid_loss, mae
        else:
            preds = np.array(predictions).flatten()
            targets = np.array(labels).flatten()
            preds_binary = (preds >= 0.5).astype(int)

            roc = roc_auc_score(targets, preds)
            ap = average_precision_score(targets, preds)
            f1 = f1_score(targets, preds_binary)
            acc = accuracy_score(targets, preds_binary)

            print(f"Epoch {epoch_counter} Validation loss: {valid_loss:.4f} | ROC-AUC: {roc:.4f} | AP: {ap:.4f} | F1: {f1:.4f} | ACC: {acc:.4f}")
            return valid_loss, roc

    def _test(self, model, test_loader, final_epoch, epoch_counter):

        if final_epoch:
            model_path = os.path.join(self.writer.log_dir, 'checkpoints', 'model.pth')
            state_dict = torch.load(model_path, map_location=self.device)
            model.load_state_dict(state_dict)
            print("Loaded best model for final test.")

        predictions = []
        labels = []
        test_loss = 0.0
        num_data = 0
        model.eval()
        with torch.no_grad():
            for data in test_loader:
                if self.config['dataset']['task'] == 'ddi':
                    data1, data2, label = data
                    data1, data2, label = data1.to(self.device), data2.to(self.device), label.to(self.device)
                    _, pred = model(data1, data2)
                    loss = self.criterion(pred, label)
                    pred = torch.sigmoid(pred)
                    predictions.extend(pred.cpu().numpy())
                    labels.extend(label.cpu().numpy())
                    test_loss += loss.item() * label.size(0)
                    num_data += label.size(0)
                else:
                    data = data.to(self.device)
                    __, pred = model(data)
                    loss = self._step(model, data, 0)
                    test_loss += loss.item() * data.y.size(0)
                    num_data += data.y.size(0)
                    if self.normalizer:
                        pred = self.normalizer.denorm(pred)
                    predictions.extend(pred.cpu().detach().numpy())
                    labels.extend(data.y.cpu().flatten().numpy())
        test_loss /= num_data

        if self.config['dataset']['task'] == 'regression':
            mae = mean_absolute_error(labels, predictions)
            print(f"Test MAE: {mae:.4f}")
            write_csv(f'./{self.writer.log_dir}/TEST_RESULTS.csv', [epoch_counter, test_loss, mae])

        else:
            preds = np.array(predictions).flatten()
            targets = np.array(labels).flatten()
            preds_binary = (preds >= 0.5).astype(int)

            roc = roc_auc_score(targets, preds)
            ap = average_precision_score(targets, preds)
            f1 = f1_score(targets, preds_binary)
            acc = accuracy_score(targets, preds_binary)

            print(f"Test ROC-AUC: {roc:.4f} | AP: {ap:.4f} | F1: {f1:.4f} | ACC: {acc:.4f}")
            write_csv(f'./{self.writer.log_dir}/TEST_RESULTS.csv', [epoch_counter, test_loss, roc, ap, f1, acc])



def main(config_input, task_name_input):


    if config_input['dataset']['task'] != 'ddi':
        dataset = finetuneDataset(config_input['batch_size'], **config_input['dataset'])
    else:
        dataset = DDIDataset(task_name_input, f'./data/downstream/{task_name_input}/')
    fine_tune = FineTune(dataset, config_input, task_name_input)
    fine_tune.train()

    return fine_tune.pre_val


if __name__ == "__main__":
    config = yaml.load(open("config_finetune2.yaml", "r"), Loader=yaml.FullLoader)
    task_names = config['task_name']
    config['t5_ckpt_path'] = LOAD_MODEL_NAME
    for task_name in task_names:

        if task_name == 'BBBP':
            config['dataset']['task'] = 'classification'
            config['dataset']['data_path'] = 'data/downstream/bbbp/raw/BBBP.csv'
            target_list = ["p_np"]

        elif task_name == 'ClinTox':
            config['dataset']['task'] = 'classification'
            config['dataset']['data_path'] = 'data/downstream/clintox/raw/clintox.csv'
            target_list = ['CT_TOX', 'FDA_APPROVED']

        elif task_name == 'BACE':
            config['dataset']['task'] = 'classification'
            config['dataset']['data_path'] = 'data/downstream/bace/raw/bace.csv'
            target_list = ["Class"]

        elif task_name == 'FreeSolv':
            config['dataset']['task'] = 'regression'
            config['dataset']['data_path'] = 'data/downstream/freesolv/raw/freesolv.csv'
            target_list = ["expt"]

        elif task_name == 'ESOL':
            config['dataset']['task'] = 'regression'
            config['dataset']['data_path'] = 'data/downstream/esol/raw/esol.csv'
            target_list = ["measured log solubility in mols per litre"]

        elif task_name == 'Lipo':
            config['dataset']['task'] = 'regression'
            config['dataset']['data_path'] = 'data/downstream/lipophilicity/raw/Lipophilicity.csv'
            target_list = ["exp"]

        elif task_name == 'ZhangDDI':
            config['dataset']['task'] = 'ddi'
            target_list = ["label"]

        elif task_name == 'ChChMiner':
            config['dataset']['task'] = 'ddi'
            target_list = ["label"]

        elif task_name == 'DeepDDI':
            config['dataset']['task'] = 'ddi'
            target_list = ["label"]

        else:
            raise ValueError('Undefined downstream task!')

        print(config)

        results_list = []
        for target in target_list:
            try:
                config['dataset']['target'] = target
                result = main(config, task_name)
                results_list.append([target, result])
            except Exception:
                continue

        os.makedirs('experiments', exist_ok=True)
        df = pd.DataFrame(results_list)
        df.to_csv(
            'experiments/{}_{}_finetune.csv'.format(config['t5_ckpt_path'], task_name),
            mode='a', index=False, header=False)
