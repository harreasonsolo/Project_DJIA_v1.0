import torch.nn as nn
import torch
import math
import torch.nn.functional as F
from utils.pytorchtools import EarlyStopping
import h5py
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from torch.autograd import Variable
from matplotlib import pyplot as plt
from tqdm import tqdm
import gc
import torch.utils.data as Data
from torch.utils.data import SubsetRandomSampler
from base_utils_and_data_generator import *

class PositionalEmbedding(nn.Module):

    def __init__(self, d_model, max_len=2048):
        super().__init__()

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()
        # print("PositionalEmbedding position", position)
        # print("PositionalEmbedding div_term", div_term)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        print("pe", pe.shape)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # print("PositionalEmbedding type(x)", type(x))
        # print("PositionalEmbedding x.shape", x.shape)
        # print("PositionalEmbedding x.size", x.size)
        return self.pe[:, :x.size(1)]


class Embedding(nn.Module):
    def __init__(self, embedding_dim, dropout=0.1):

        super().__init__()
        # self.token = TokenEmbedding(vocab_size=vocab_size, embed_size=embed_size)
        self.position = PositionalEmbedding(d_model=embedding_dim)
        # self.segment = SegmentEmbedding(embed_size=self.token.embedding_dim)
        self.dropout = nn.Dropout(p=dropout)
        # self.embed_size = embed_size

    def forward(self, representation):
        #print("Embedding representation.shape", representation.shape)
        #print("Embedding self.position(representation).shape", self.position(representation).shape)
        x = representation + self.position(representation)
        #print("Embedding x.shape", x.shape)
        return self.dropout(x)


class FeedForward(nn.Module):

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(FeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        #self.activation = GELU()

    def forward(self, x):
        # return self.w_2(self.dropout(self.activation(self.w_1(x))))
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class SublayerConnection(nn.Module):

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = nn.LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


class Attention(nn.Module):

    def forward(self, query, key, value, mask=None, dropout=None):
        scores = torch.matmul(query, key.transpose(-2, -1)) \
                 / math.sqrt(query.size(-1))

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        p_attn = F.softmax(scores, dim=-1)

        if dropout is not None:
            p_attn = dropout(p_attn)

        return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):

    def __init__(self, h, d_model, dropout=0.1):
        super().__init__()
        assert d_model % h == 0

        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h

        self.linear_layers = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(3)])
        self.output_linear = nn.Linear(d_model, d_model)
        self.attention = Attention()

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linear_layers, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, attn = self.attention(query, key, value, mask=mask, dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)

        return self.output_linear(x)


class TransformerBlock(nn.Module):

    def __init__(self, hidden, attn_heads, feed_forward_hidden, dropout):

        super().__init__()
        self.attention = MultiHeadedAttention(h=attn_heads, d_model=hidden)
        # self.multi_head_attention = nn.MultiheadAttention(embed_dim=hidden, num_heads=attn_heads)
        self.feed_forward = FeedForward(d_model=hidden, d_ff=feed_forward_hidden, dropout=dropout)
        self.input_sublayer = SublayerConnection(size=hidden, dropout=dropout)
        self.output_sublayer = SublayerConnection(size=hidden, dropout=dropout)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.input_sublayer(x, lambda _x: self.attention.forward(_x, _x, _x))
        x = self.output_sublayer(x, self.feed_forward)
        return self.dropout(x)


class linearRegressor(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        #print("linear_regressor hidden", hidden)
        self.regressor = nn.Sequential(
        nn.Linear(hidden, 20),
        nn.ReLU(True),
        nn.Linear(20, 1)
        )

    def forward(self, x):
        x = self.regressor(x)
        return x


class linearClassifier(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        #print("linear_classifier hidden", hidden)
        self.classifier = nn.Sequential(
        nn.Linear(hidden, 20),
        nn.ReLU(True),
        nn.Linear(20, 2),
        nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.classifier(x)
        return x


class selfAttentionDjia(nn.Module):
    def __init__(self, hidden=36, n_layers=6, attn_heads=3, dropout=0.1, train_method="regression", target_method="last_one"):

        super().__init__()
        self.hidden = hidden
        self.n_layers = n_layers
        self.attn_heads = attn_heads
        self.target_method = target_method
        self.train_method = train_method
        # paper noted they used 4*hidden_size for ff_network_hidden_size
        self.feed_forward_hidden = hidden * 4

        # embedding for BERT, sum of positional, segment, token embeddings
        self.embedding = Embedding(dropout=0.1, embedding_dim=hidden)

        # self.segment = SegmentEmbedding(embed_size=self.token.embedding_dim)
        self.dropout = nn.Dropout(p=dropout)

        # multi-layers transformer blocks, deep network
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(hidden, attn_heads, hidden * 4, dropout) for _ in range(n_layers)])
        # TransformerBlock have a problem about "mask"

        if self.train_method == "classification":
            self.linear_classifier = linearClassifier(hidden=hidden)
        elif self.train_method == "regression":
            self.linear_regressor = linearRegressor(hidden=hidden)

    def forward(self, x):
        # attention masking for padded token
        # torch.ByteTensor([batch_size, 1, seq_len, seq_len)
        # mask = (x > 0).unsqueeze(1).repeat(1, x.size(1), 1).unsqueeze(1)

        # embedding with position information with dropout
        x = self.embedding(x)
        # running over multiple transformer blocks
        for transformer in self.transformer_blocks:
            x = transformer.forward(x)

        # for different feature representation
        if self.target_method == "last_one":
            x = x[:, -1, :]
        elif self.target_method == "average":
            x = torch.mean(x, 1, keepdim=False)
        elif self.target_method == "padding_one":
            x = x[:, -1, :]

        # for different train method
        if self.train_method == "classification":
            x = self.linear_classifier(x)
        elif self.train_method == "regression":
            x = self.linear_regressor(x)

        # print("x.shape", x.shape)
        return x
#
#
def train(model, trainX, trainY, patience, valid_ratio, learning_rate, epoch_num,
          model_save_name = "new_model",
          train_method="regression", target_method="last_one", pre_trained_model_file=""):

    if pre_trained_model_file:
        save_model = torch.load(pre_trained_model_file)
        #print("save_model", save_model.keys())
        new_model_dict = model.state_dict()
        #print("new_model_dict", model.state_dict().keys())
        state_dict = {k: v for k, v in save_model.items() if k in new_model_dict.keys()}
        #print("state_dict1", state_dict.keys())
        # this should be classifier with no classifier layer and no regression layer
        new_model_dict.update(state_dict)
        model.load_state_dict(new_model_dict)
        #print("model", model.state_dict().keys())
        del save_model, new_model_dict, state_dict
        gc.collect()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # initial the model
    # model = fc_classifier()
    model.to(device)
    # setting the early stopping
    early_stopping = EarlyStopping(patience=patience, verbose=True)

    # initial the training dataset and validation dataset
    trainX = torch.Tensor(trainX)
    trainY = torch.Tensor(trainY)
    training_dataset = Data.TensorDataset(trainX, trainY)
    # print("trainX", trainX.shape)
    # print("trainY", trainY.shape)
    del trainX, trainY
    gc.collect()
    #data_loader = Data.DataLoader(training_dataset, batch_size=50, shuffle=True)

    # obtain training indices that will be used for validation
    num_train = len(training_dataset)
    indices = list(range(num_train))
    np.random.shuffle(indices)
    split = int(valid_ratio * num_train)
    train_idx, valid_idx = indices[split:], indices[:split]

    # define samplers for obtaining training and validation batches
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    # load training data in batches
    train_loader = torch.utils.data.DataLoader(training_dataset,
                                               batch_size=50,
                                               sampler=train_sampler,
                                               num_workers=1)

    # load validation data in batches
    valid_loader = torch.utils.data.DataLoader(training_dataset,
                                               batch_size=50,
                                               sampler=valid_sampler,
                                               num_workers=1)

    # initial the loss function
    if train_method=="regression":
        criterion = nn.MSELoss()
    elif train_method=="classification":
        criterion = nn.CrossEntropyLoss()

    # initial the optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=1e-5
    )

    for epoch in tqdm(range(epoch_num)):
        # loss array
        train_loss_array = []
        valid_loss_array = []

        # train
        model.train()
        for train_X, train_Y in train_loader:
            train_X = train_X.to(device)
            if train_method == "regression":
                train_Y = train_Y.to(device)
            elif train_method == "classification":
                train_Y = train_Y.to(device, dtype=torch.int64)

            # fwd
            output = model(train_X)

            # print("output", output.shape)
            # print("train_Y", train_Y.shape)
            if train_method == "regression":
                #print("output before", output.shape)
                output = output.squeeze(1)
            loss = criterion(output, train_Y)
            # bp
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss_array.append(loss.cpu().data)
        # print("train_loss_array", len(train_loss_array))
        # validation
        model.eval()
        for valid_X, valid_Y in valid_loader:
            valid_X = valid_X.to(device)
            if train_method == "regression":
                valid_Y = valid_Y.to(device)
            elif train_method == "classification":
                valid_Y = valid_Y.to(device, dtype=torch.int64)

            output = model(valid_X)
            if train_method == "regression":
                #print("output before", output.shape)
                output = output.squeeze(1)
            loss = criterion(output, valid_Y)
            valid_loss_array.append(loss.cpu().data)
        # print("valid_loss_array", len(valid_loss_array))
        print("epoch:[{}/{}], training loss:{:.4f}, validation loss:{:.4f}".format(epoch + 1, epoch_num, np.array(train_loss_array).mean(), np.array(valid_loss_array).mean()))

        early_stopping(np.array(valid_loss_array).mean(), model, save_model=False)

        if early_stopping.early_stop:
            print("early stop.")
            break

    # print("train_loss_array", train_loss_array)
    # print("train_loss_array", len(train_loss_array))
    # print("valid_loss_array", valid_loss_array)
    # print("valid_loss_array", len(valid_loss_array))

    pth_name = "./" + model_save_name + "_" + train_method + "_" + target_method + ".pth"
    torch.save(model.state_dict(), pth_name)
    return model


def pred_with_model(model, input):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input = torch.Tensor(input)
    input = input.to(device)
    model = model.to(device)
    model.eval()
    batch_size = 50

    input_loader = torch.utils.data.DataLoader(input,
                                               batch_size=batch_size,
                                               shuffle=False,
                                               num_workers=0)

    output = torch.empty([input.shape[0], 2])
    # print("output.shape", output.shape)
    with torch.no_grad():
        index = 0
        for batch_input in input_loader:
            length = batch_input.shape[0]
            print("batch_input", batch_input.shape)
            print("from", index * batch_size, "to", index * batch_size + length)
            print("model(batch_input)", model(batch_input).shape)
            print("output[index*batch_size : (index+1)*length].shape",
                  output[index * batch_size: index * batch_size + length].shape)
            output[index * batch_size: index * batch_size + length] = model(batch_input)
            index += 1
    return output

if __name__ == '__main__':
    # print("start!")

        # print("model.state_dict", model.state_dict)
    # for k in model.state_dict().keys():
    #     print(k)

    model = selfAttentionDjia(hidden=36, n_layers=6, attn_heads=3, dropout=0.1, train_method="classification",
                              target_method="average")

    with h5py.File('ae_data - Copy.h5', 'r') as hf:
        a = hf["training_data_1095"][:5]
    # print("a", a.shape, type(a))
    a = torch.Tensor(a)
    # print("a", a.shape, type(a))
    output = model.forward(a)
    # print("output1", output.shape)
    # print("output2", output)
    # print("output3", output[:, -1, :].shape)
    # print("output4", output[:, -1, :])

    output = torch.mean(output, 1, keepdim=False)
    # print("output5", output.shape)
    # print("output6", output)

