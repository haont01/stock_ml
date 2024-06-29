from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual, visual_plus
from utils.metrics import metric
from utils.losses import smape_loss, mape_loss, mase_loss
import torch # type: ignore
import torch.nn as nn # type: ignore
from torch import optim # type: ignore
import os
import time
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')


class Exp_MultiStock_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_MultiStock_Forecast, self).__init__(args)

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        loss_name = self.args.loss
        if loss_name == 'MSE':
            return nn.MSELoss()
        elif loss_name == 'MAE':
            return nn.L1Loss()
        elif loss_name == 'MAPE':
            return mape_loss()
        elif loss_name == 'MASE':
            return mase_loss()
        elif loss_name == 'SMAPE':
            return smape_loss()
        else:
            return nn.MSELoss()
        # return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)

                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        if self.args.predict_multi_stock:
            print("Training multi stocks")
            stockList = pd.read_csv(f"{self.args.root_path}/tickers.csv")['Symbol'].unique()
        
        train_data = {}
        train_loader = {}
        vali_data = {}
        vali_loader = {}
        test_data = {}
        test_loader = {}

        for stock in stockList:
            self.args.data_path = f'{stock}.csv'
            train_data[stock], train_loader[stock] = self._get_data(flag='train')
            vali_data[stock], vali_loader[stock] = self._get_data(flag='val')
            test_data[stock], test_loader[stock] = self._get_data(flag='test')
            

        # train_data, train_loader = self._get_data(flag='train')
        # vali_data, vali_loader = self._get_data(flag='val')
        # test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        # train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for stock in stockList:
            train_steps = len(train_loader[stock])
            print("training:", stock)
            for epoch in range(self.args.train_epochs):
                iter_count = 0
                train_loss = []

                self.model.train()
                epoch_time = time.time()
                for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader[stock]):
                    iter_count += 1
                    model_optim.zero_grad()
                    batch_x = batch_x.float().to(self.device)

                    batch_y = batch_y.float().to(self.device)
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)

                    # decoder input
                    dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                    dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                    # encoder - decoder
                    if self.args.use_amp:
                        with torch.cuda.amp.autocast():
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                            f_dim = -1 if self.args.features == 'MS' else 0
                            outputs = outputs[:, -self.args.pred_len:, f_dim:]
                            batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                            loss = criterion(outputs, batch_y)
                            train_loss.append(loss.item())
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())

                    if (i + 1) % 100 == 0:
                        print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                        speed = (time.time() - time_now) / iter_count
                        left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                        print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                        iter_count = 0
                        time_now = time.time()

                    if self.args.use_amp:
                        scaler.scale(loss).backward()
                        scaler.step(model_optim)
                        scaler.update()
                    else:
                        loss.backward()
                        model_optim.step()

                print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
                train_loss = np.average(train_loss)
                vali_loss = self.vali(vali_data[stock], vali_loader[stock], criterion)
                test_loss = self.vali(test_data[stock], test_loader[stock], criterion)

                print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                    epoch + 1, train_steps, train_loss, vali_loss, test_loss))
                early_stopping(vali_loss, self.model, path)
                if early_stopping.early_stop:
                    print("Early stopping")
                    continue

                adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0):
        if self.args.predict_multi_stock:
            print("Testing multi stocks")
            stockList = pd.read_csv(f"{self.args.root_path}/tickers.csv")['Symbol']
        
        test_data = {}
        test_loader = {}
        for stock in stockList:
            self.args.data_path = f"{stock}.csv"
            test_data[stock], test_loader[stock] = self._get_data(flag='test')
        # test_data, test_loader = self._get_data(flag='test')

        if test:
            # print('loading model:', os.path.join('./checkpoints/', 'checkpoint.pth'))
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))
            

        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # preds = []
        # trues = []
        # gt_values = []
        # pd_values = []

        self.model.eval()

        for stock in stockList:
            # folder_path = folder_path
            preds = []
            trues = []
            gt_values = []
            pd_values = []
            pd_low = []
            pd_high = []

            with torch.no_grad():
                for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader[stock]):
                    batch_x = batch_x.float().to(self.device)
                    batch_y = batch_y.float().to(self.device)

                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)
                    # decoder input
                    dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                    dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                    # if i == 1:
                        # print("x", batch_x.shape)
                        # print("y", batch_y.shape)
                        # print("x_mark", batch_x_mark.shape)
                        # print("y_mark", batch_y_mark.shape)
                        # print("dec_inp", dec_inp.shape)
                    # encoder - decoder
                    if self.args.use_amp:
                        with torch.cuda.amp.autocast():
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]

                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, :]
                    batch_y = batch_y[:, -self.args.pred_len:, :].to(self.device)
                    outputs = outputs.detach().cpu().numpy()
                    batch_y = batch_y.detach().cpu().numpy()
                    if test_data[stock].scale and self.args.inverse:
                        shape = outputs.shape
                        outputs = test_data[stock].inverse_transform(outputs.squeeze(0)).reshape(shape)
                        batch_y = test_data[stock].inverse_transform(batch_y.squeeze(0)).reshape(shape)
            
                    outputs = outputs[:, :, f_dim:]
                    batch_y = batch_y[:, :, f_dim:]
                    # print("Number: ", i)
                    # print(batch_y)

                    pred = outputs
                    true = batch_y

                    preds.append(pred)
                    trues.append(true)
                    if i % self.args.pred_len == 0:
                        # print("predict: ", pred)
                        # print
                        gt_values.append(true[0, :, -1])
                        pd_values.append(pred[0, :, -1])
                        pd_low.append(pred[0, :, -2])
                        pd_high.append(pred[0, :, -3])

                        input = batch_x.detach().cpu().numpy()
                        if test_data[stock].scale and self.args.inverse:
                            shape = input.shape
                            input = test_data.inverse_transform(input.squeeze(0)).reshape(shape)
                        # _gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                        # _pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                        # visual(_gt, _pd, os.path.join(folder_path, str(i) + '.pdf'))
                            # visual(true[0, :, -1], pred[0, :, -1], os.path.join(folder_path, str(i) + '.pdf'))

            preds = np.array(preds)
            trues = np.array(trues)
            groundtruth = np.concatenate(gt_values, axis=0)
            predicted = np.concatenate(pd_values, axis=0)
            predict_low = np.concatenate(pd_low, axis=0)
            predict_high = np.concatenate(pd_high, axis=0)
            # gts = np.array_split(groundtruth, 10)
            # pds = np.array_split( predicted, 10)
            # a = 0
            # for _gt, _pd in zip(gts, pds):
            #     visual(_gt, _pd, os.path.join(folder_path, str(a) + '.pdf'))
            #     a = a + 1

            # visual(groundtruth, predicted, os.path.join(folder_path, stock + '.pdf'))


            # print('test shape:', preds.shape, trues.shape)
            preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
            trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
            # print('test shape:', preds.shape, trues.shape)

            # result save
            result_folder_path = './results/' + setting + '/'
            if not os.path.exists(result_folder_path):
                os.makedirs(result_folder_path)

            mae, mse, rmse, mape, mspe = metric(preds, trues)

            print(type(groundtruth))

            visual_plus(true=groundtruth[: 300], preds=predicted, low=predict_low[:300], high=predict_high[:300], name=os.path.join(folder_path, stock + '.pdf'), symbol=stock, mse=mse, mae=mae)


            print('mse:{}, mae:{}'.format(mse, mae))
            f = open("result_long_term_forecast.txt", 'a')
            f.write(setting + "  \n")
            f.write('mse:{}, mae:{}'.format(mse, mae))
            f.write('\n')
            f.write('\n')
            f.close()

            np.save(result_folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
            np.save(result_folder_path + 'pred.npy', preds)
            np.save(result_folder_path + 'true.npy', trues)

        return
