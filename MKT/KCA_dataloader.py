import torch
import torch.utils.data as Data
from KCA_readdata_for_data_analysis import data_reader
# print("%%%%%"+__name__)
from config import *


def get_data_loader(batch_size, num_of_questions, max_step, fold):
    # (batch_size=128, num_of_questions=10, max_step=50, fold=0决定是哪个文件)
    handle = data_reader(os.path.join(cfg.splited_data_dir, "train_firstatt_" + str(fold) + ".csv"),
                                      os.path.join(cfg.splited_data_dir, "val_firstatt_" + str(fold) + ".csv"),
                                                  os.path.join(cfg.splited_data_dir, "test_data.csv"), max_step,
                                                   num_of_questions)

    # get_train_data  ->   self.get_data(self.train_path)
    # drain对应一个train_attr_id文件中的所有信息，也就是几个学生的序列
    dtrain = torch.tensor(handle.get_train_data().astype(float).tolist(),
                          dtype=torch.float32)
    dtest = torch.tensor(handle.get_test_data().astype(float).tolist(),
                         dtype=torch.float32)

    train_loader = Data.DataLoader(dtrain, batch_size=batch_size, shuffle=True)
    test_loader = Data.DataLoader(dtest, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader
