from __future__ import print_function
import time
import numpy as np
import torch
import torch.nn as nn
import copy
import codecs
import random
from torch.nn.utils import clip_grad_norm_
from torchtext.legacy import data
from torchtext import datasets
import gensim
from gensim.models import Word2Vec
import torch.nn.functional as F
from torch.autograd import Function, Variable
import pandas as pd
import math
import sys

SEED = int(sys.argv[1])

# from model import *
# from optim import Optim
BATCH_SIZE = 32
root_dir = "model/"
cuda = torch.cuda.is_available()
device = torch.device("cpu") if not cuda else torch.device("cuda")
EMBED_DIM = 100
HIDDEN_DIM = 128 # Default from paper
N_LAYERS = 1 # Default from paper
DROPOUT = 0 # Default from paper
TOPIC_DROPOUT = 0.2 # Default from paper
NUM_TOPICS = 34 #Number of politicians in training set
GRAD_CLIP = 5 # Default from paper
LR = 0.0001 # from Kumar (2019)
PRETRAIN_EPOCHS = 5 #originally 3
EPOCHS = 3
C_STEPS = 3
T_STEPS = 10
PARAM_INIT = 0.1
TOPIC_LOSS = "kl" #[mse|ce|kl]
BOTTLENECK_DIM = 0
reset_classifier = False


def seed_everything(seed, cuda=False):
    # Set the random seed manually for reproducibility.
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed_all(seed)
        
class Optim(object):

    def set_parameters(self, params):
        self.params = list(params)  # careful: params may be a generator
        self.params = filter(lambda p: p.requires_grad, self.params)
        if self.method == 'sgd':
            self.optimizer = torch.optim.SGD(self.params, lr=self.lr)
        elif self.method == 'adagrad':
            self.optimizer = torch.optim.Adagrad(self.params, lr=self.lr)
        elif self.method == 'adadelta':
            self.optimizer = torch.optim.Adadelta(self.params, lr=self.lr)
        elif self.method == 'adam':
            self.optimizer = torch.optim.Adam(self.params, lr=self.lr)
        else:
            raise RuntimeError("Invalid optim method: " + self.method)

    def __init__(self, method, lr, max_grad_norm, lr_decay=1, start_decay_at=None):
        self.last_ppl = None
        self.best_ppl = None
        self.lr = lr
        self.max_grad_norm = max_grad_norm
        self.method = method
        self.lr_decay = lr_decay
        self.start_decay_at = start_decay_at
        self.start_decay = False

    def step(self):
        # Compute gradients norm.
        if self.max_grad_norm:
            clip_grad_norm_(self.params, self.max_grad_norm)
        self.optimizer.step()
        
def create_iterators(batch_size, trainfile, valid, test ,vectors=None, device=-1):
    TEXT = data.Field(include_lengths=True)
    LABEL = data.LabelField()
    INDEX = data.Field(sequential=False, use_vocab=False, batch_first=True)

    TOPICS = data.Field(sequential=True, use_vocab=False, preprocessing=data.Pipeline(lambda x:float(x)), batch_first=True)
    train = data.TabularDataset(path=trainfile, format="tsv", fields=[('index', INDEX), ('text',TEXT),  ('label', LABEL), ('topics', TOPICS)])

    TEXT.build_vocab(train, vectors=vectors, max_size=30000)
    LABEL.build_vocab(train)
    print(LABEL.vocab.stoi)
    LOC = data.LabelField()

    val = data.TabularDataset(path=valid, format="tsv", fields=[('index', INDEX), ('text',TEXT), ('label', LABEL)])
    test = data.TabularDataset(path=test, format="tsv", fields=[('index', INDEX), ('text',TEXT), ('label', LABEL), ('loc', LOC)])
    LOC.build_vocab(test)
    print(LOC.vocab.stoi)
    train_iter, val_iter, test_iter = data.BucketIterator.splits((train, val, test), batch_sizes=(batch_size, 256, 256), device=device, repeat=False, sort_key=lambda x: len(x.text))

    return (train_iter, val_iter, test_iter), TEXT, LABEL, TOPICS, INDEX, LOC

class Encoder(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, nlayers=1, dropout=0.,
               bidirectional=True, rnn_type='GRU'):
        super(Encoder, self).__init__()
        self.bidirectional = bidirectional
        assert rnn_type in ['LSTM', 'GRU'], 'Use one of the following: {}'.format(str(RNNS))
        rnn_cell = getattr(nn, rnn_type) # fetch constructor from torch.nn, cleaner than if
        self.rnn = rnn_cell(embedding_dim, hidden_dim, nlayers,
                            dropout=dropout, bidirectional=bidirectional)

    def forward(self, input, hidden=None):
        self.rnn.flatten_parameters()
        return self.rnn(input, hidden)

class BahdanauAttention(nn.Module):
    def __init__(self, hidden_dim, attn_dim):
        super(BahdanauAttention, self).__init__()
        self.linear = nn.Linear(hidden_dim, attn_dim)
        self.linear2 = nn.Linear(attn_dim, 1)

    def forward(self, hidden, mask=None):
        # hidden = [TxBxH]
        # mask = [TxB]
        # Outputs = a:[TxB], lin_comb:[BxV]

        # print (hidden.size())
        # Here we assume q_dim == k_dim (dot product attention)
        hidden = hidden.transpose(0,1) # [TxBxH] -> [BxTxH]
        energy = self.linear(hidden) # [BxTxH] -> [BxTxA]
        energy = F.tanh(energy)
        energy = self.linear2(energy) # [BxTxA] -> [BxTx1]
        energy = F.softmax(energy, dim=1) # scale, normalize

        # print (energy.size())
        if mask is not None:
            mask = mask.transpose(0, 1).unsqueeze(2)
            # print (mask.size())
            energy = energy * mask
            # print (energy.size())
            Z = energy.sum(dim=1, keepdim=True) #[BxTx1] -> [Bx1x1]
            # print (Z.size())
            # input()
            energy = energy/Z #renormalize

        energy = energy.transpose(1, 2) # [BxTx1] -> [Bx1xT]
        # hidden = hidden.transpose(0,1) # [TxBxH] -> [BxTxH]
        linear_combination = torch.bmm(energy, hidden).squeeze(1) #[Bx1xT]x[BxTxH] -> [BxH]
        return energy, linear_combination

class Classifier_GANLike(nn.Module):
    def __init__(self, embedding, encoder, attention, hidden_dim, num_classes=10):
        super(Classifier_GANLike, self).__init__()
        # num_classes=2
        self.embedding = embedding
        self.encoder = encoder
        self.attention = attention
        self.decoder = nn.Linear(hidden_dim, num_classes)

        size = 0
        for p in self.parameters():
            size += p.nelement()
        print('Total param size: {}'.format(size))

    def forward(self, input, padding_mask=None, rationale_mask = None):
        if rationale_mask is not None:
            x_embeds = self.embedding(input.squeeze(1))
            x_embeds = x_embeds * rationale_mask.unsqueeze(-1)
        else:
            x_embeds = self.embedding(input)
        outputs, hidden = self.encoder(x_embeds)
        if isinstance(hidden, tuple): # LSTM
            hidden = hidden[1] # take the cell state

        if self.encoder.bidirectional: # need to concat the last 2 hidden layers
            hidden = torch.cat([hidden[-1], hidden[-2]], dim=1)
        else:
            hidden = hidden[-1]

        # max across T?
        # Other options (work worse on a few tests):
        # linear_combination, _ = torch.max(outputs, 0)
        # linear_combination = torch.mean(outputs, 0)

        energy, linear_combination = self.attention(outputs, padding_mask)
        logits = self.decoder(linear_combination)

        # if gradreverse:
        #   reverse_linear_comb = ReverseLayerF.apply(linear_combination, alpha)
        #   topic_logprobs = self.topic_decoder(reverse_linear_comb)
        # else:
        #   topic_logprobs = self.topic_decoder(linear_combination)
        return logits, energy, linear_combination
    
def update_stats(accuracy, confusion_matrix, logits, y):
    _, max_ind = torch.max(logits, 1)
    equal = torch.eq(max_ind, y)
    correct = int(torch.sum(equal))

    for j, i in zip(max_ind, y):
        confusion_matrix[int(i),int(j)]+=1

    return accuracy + correct, confusion_matrix

def update_stats_topics(accuracy, confusion_matrix, logits, y):
    _, max_ind = torch.max(logits, 1)
    _, max_ind_y = torch.max(y, 1)
    equal = torch.eq(max_ind, max_ind_y)
    correct = int(torch.sum(equal))

    for j, i in zip(max_ind, max_ind_y):
        confusion_matrix[int(i),int(j)]+=1

    return accuracy + correct, confusion_matrix

def pretrain_classifier(model, data, optimizer, criterion, nlabels, epoch):

    model.train()
    accuracy, confusion_matrix = 0.0, np.zeros((nlabels, nlabels), dtype=int)

    t = time.time()
    total_loss = 0
    num_batches = len(data)

    for batch_num, batch in enumerate(data):

        model.zero_grad()
        x, lens = batch.text
        y = batch.label
        padding_mask = x.ne(1).float()

        logits, energy, topic_logprobs = model(x, padding_mask=padding_mask)
        if energy is not None:
            energy = torch.squeeze(energy)

        loss = criterion(logits.view(-1, nlabels), y)

        if torch.isnan(loss):
            print ()
            print ("something has become nan")
            print(logits)
            print (y)
            print (x)
            print (lens)
            input("Press Ctrl+C")
    total_loss += float(loss)

    accuracy, confusion_matrix = update_stats(accuracy, confusion_matrix, logits, y)

    loss.backward()
    # torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
    optimizer.step()

    print("[Batch]: {}/{} in {:.5f} seconds".format(
          batch_num, len(data), time.time() - t), end='\r')
    t = time.time()

    print()
    print("[PreTraining Epoch {}: Training Loss]: {:.5f}".format(epoch, total_loss / len(data)), end=" ")
    print("[Training Accuracy]: {}/{} : {:.3f}%".format(accuracy, len(data.dataset), accuracy / len(data.dataset) * 100))
    # print(confusion_matrix)
    return total_loss / len(data)

def train_topic_predictor(c_model, t_model, data, optimizer, criterion, num_topics, topic_loss, epoch, steps):

    c_model.train()
    t_model.train()

    accuracy_fromtopics, confusion_matrix_ = 0.0, np.zeros((num_topics, num_topics), dtype=int)
    t = time.time()
    total_topic_loss = 0
    num_batches = len(data)

    step = 0
    for batch_num, batch in enumerate(data):
        # if step > steps:
        #   break
        c_model.zero_grad()
        t_model.zero_grad()
        x, lens = batch.text
        y = batch.label
        padding_mask = x.ne(1).float()
        logits, energy, textrep = c_model(x, padding_mask=padding_mask)
        topic_logprobs = t_model(textrep)

        topics = batch.topics
        if topic_loss == "kl":
            topic_loss = criterion(topic_logprobs, topics.float())
        elif topic_loss == "ce":
            topic_loss = torch.sum(topic_logprobs*topics)
        else:
            g = (topics - torch.exp(topic_logprobs))
            topic_loss = (g*g).sum(dim=-1).mean()

        loss = topic_loss
#         loss.requires_grad = True
        total_topic_loss += float(topic_loss)

        accuracy_fromtopics, confusion_matrix_ = update_stats_topics(accuracy_fromtopics, confusion_matrix_, topic_logprobs, topics)
        loss.backward()
        optimizer.step()

        print("[Topic Decoder Epoch {} Batch]: {}/{} in {:.5f} seconds".format(epoch, batch_num, len(data), time.time() - t), end='\r')
        t = time.time()
        step += 1

    print()
    print("[Epoch {} Topic Loss]: {:.5f}".format(epoch, total_topic_loss / len(data)), end=" ")
    print("[Accuracy From Topics]: {}/{} : {}%".format(
        accuracy_fromtopics, np.sum(confusion_matrix_), accuracy_fromtopics / np.sum(confusion_matrix_) * 100))
    # print(confusion_matrix_)
    return total_topic_loss / len(data)

def train_classifier(c_model, t_models, data, optimizer, classify_criterion, topic_criterion, nlabels, num_topics,topic_loss, epoch, steps):

    c_model.train()
    # print (len(t_models))
    if t_models is not None:
        for t_model in t_models:
            t_model.train()
    accuracy, confusion_matrix = 0.0, np.zeros((nlabels, nlabels), dtype=int)
    accuracy_fromtopics, confusion_matrix_ = 0.0, np.zeros((num_topics, num_topics), dtype=int)

    t = time.time()
    total_loss = 0
    total_topic_loss = 0
    num_batches = len(data)

    step = 0
    d_id = 0 #which decoder to use. ++ and modulo len(t_models) after every step
    for batch_num, batch in enumerate(data):

        # if step >= steps:
        #   break
        c_model.zero_grad()
        if t_models is not None:
            t_models[d_id].zero_grad()

        x, lens = batch.text
        y = batch.label
        padding_mask = x.ne(1).float()

        logits, energy, sentrep = c_model(x, padding_mask=padding_mask)

        if t_models is not None:
            fake_topics = torch.ones(batch.topics.size()).cuda() #want the model to predict uniform topics
            fake_topics = fake_topics.div(fake_topics.sum(dim=-1, keepdim=True))

            real_topics = batch.topics

            topic_logprobs = t_models[d_id](sentrep).cuda()

        loss = classify_criterion(logits.view(-1, nlabels), y)

        if torch.isnan(loss):
            print ()
            print ("something has become nan")
            print(logits)
            print (y)
            print (x)
            print (lens)
            input("Press Ctrl+C")
        total_loss += float(loss)

        if t_models is not None:
            if topic_loss == "kl":
                topic_loss = topic_criterion(topic_logprobs, fake_topics)
            elif topic_loss == "ce":
                topic_loss = -torch.sum(topic_logprobs*fake_topics)
            else:
                g = (fake_topics - torch.exp(topic_logprobs))
                topic_loss = (g*g).sum(dim=-1).mean()

        loss += topic_loss
#         loss.requires_grad = True
        total_topic_loss += float(topic_loss)

        accuracy, confusion_matrix = update_stats(accuracy, confusion_matrix, logits, y)
        if t_models is not None:
            accuracy_fromtopics, confusion_matrix_ = update_stats_topics(accuracy_fromtopics, confusion_matrix_, topic_logprobs, real_topics)
        loss.backward()
        optimizer.step()

        print("[Batch]: {}/{} in {:.5f} seconds".format(
              batch_num, len(data), time.time() - t), end='\r')
        t = time.time()
        step += 1
        if t_models is not None:
            d_id = (d_id + 1) % len(t_models)

    print()
    print("[Epoch {}: Fake Topic Loss]: {:.5f}".format(epoch, total_topic_loss / len(data)), end=" ")
    print("Loss]: {:.5f}".format(total_loss / len(data)), end=" ")
    print("Training Accuracy]: {}/{} : {}%".format(accuracy, np.sum(confusion_matrix), accuracy / np.sum(confusion_matrix) * 100), end=" ")
    if t_models is not None:
        print("accuracy_from_real_topics]: {}/{} : {}%".format(
          accuracy_fromtopics, np.sum(confusion_matrix_), accuracy_fromtopics / np.sum(confusion_matrix_) * 100))
    # print(confusion_matrix)
    return total_loss / len(data)

def evaluate(model, t_models, data, criterion, topic_criterion, nlabels, topic_loss, datatype='Valid', itos=None, litos=None):

    model.eval()

    if itos is not None:
        attention_file = codecs.open(f"{root_dir}attention_{SEED}.txt", "w", encoding="utf8")

    accuracy, confusion_matrix = 0.0, np.zeros((nlabels, nlabels), dtype=int)

    t = time.time()
    total_loss = 0
    total_topic_loss = 0
    d_id = 0 #which decoder to use. ++ and modulo len(t_models) after every step
    with torch.no_grad():

        for batch_num, batch in enumerate(data):
            x, lens = batch.text
            y = batch.label
            indices = batch.index.cpu().data.numpy()
            # if args.data in ["RT_GENDER"]:
            #   indices = batch.index.cpu().data.numpy()
            #   # print (indices.size())
            # else:
            #   indices = np.array(([0]*len(y)))
            padding_mask = x.ne(1).float()

            logits, energy, sentrep = model(x, padding_mask=padding_mask)
            topic_logprobs = t_models[d_id](sentrep).cuda()
            fake_topics = torch.ones(topic_logprobs.size()).cuda() #want the model to predict uniform topics
            fake_topics = fake_topics.div(fake_topics.sum(dim=-1, keepdim=True))
            if topic_loss == "kl":
                topic_loss = topic_criterion(topic_logprobs, fake_topics)
            elif topic_loss == "ce":
                topic_loss = -torch.sum(topic_logprobs*fake_topics)
            else:
                g = (fake_topics - torch.exp(topic_logprobs))
                topic_loss = (g*g).sum(dim=-1).mean()

            total_topic_loss += float(topic_loss)

            if itos is not None:
                m = torch.nn.Softmax()
                soft_logits = m(logits)
                max_val, max_ind = torch.max(soft_logits, 1)

                energy = energy.squeeze(1).cpu().data.numpy()
                for sentence, length, attns, ll, mi, index, max_val in zip(x.permute(1,0).cpu().data.numpy(), lens.cpu().data.numpy(), energy, y.cpu().data.numpy(), max_ind.cpu().data.numpy(), indices, max_val.cpu().data.numpy()):
                    s = ""
                    for wordid, attn in zip(sentence[:length], attns[:length]):
                        s += str(itos[wordid])+":"+str(attn)+" "
                    gold = str(litos[ll])
                    pred = str(litos[mi])
                    # print (index)
                    index = str(index)
                    max_val = str(max_val)
                    z = s+"\t"+gold+"\t"+pred+"\t"+index+"\t"+max_val+"\n"
                    attention_file.write(z)
            bloss = criterion(logits.view(-1, nlabels), y)

            if torch.isnan(bloss):
                print ("NANANANANANA")
                print (logits)
                print (y)
                print (x)
                input("Press Ctrl+C")

            total_loss += float(bloss)
            accuracy, confusion_matrix = update_stats(accuracy, confusion_matrix, logits, y)

            print("[Batch]: {}/{} in {:.5f} seconds".format(
                batch_num, len(data), time.time() - t), end='\r')
            t = time.time()
            d_id = (d_id + 1) % len(t_models)

    if itos is not None:
        attention_file.close()

    print()
    print("[{} loss]: {:.5f}".format(datatype, total_loss / len(data)), end=" ")
    print("[{} Topic loss]: {:.5f}".format(datatype, total_topic_loss / len(data)), end=" ")
    print("[{} accuracy]: {}/{} : {:.3f}%".format(datatype,
        accuracy, len(data.dataset), accuracy / len(data.dataset) * 100))
    print(confusion_matrix)
    return (total_loss / len(data)) + (total_topic_loss / len(data))

def create_dataset(SEED, root_dir):
    df = pd.read_csv(f"{root_dir}subset_odds2.tsv",sep="\t", names=['index','text', 'sex','los'])
    HALF_SAMPLE = df[df.sex == 'female'].shape[0]
    new_df = pd.concat([df[df.sex=='male'].sample(n=HALF_SAMPLE), df[df.sex == 'female']],axis=0)
    print("Train size", new_df.shape[0])
    new_df.to_csv(f"{root_dir}subset_{SEED}.tsv", sep="\t", header=False, index=False)

seed_everything(seed=SEED, cuda=cuda)

# create dataset
create_dataset(SEED, root_dir)

iters, TEXT, LABEL, TOPICS, INDEX, SUBS = create_iterators(BATCH_SIZE, f"{root_dir}subset_{SEED}.tsv", f"{root_dir}valid_set_clean.tsv", f"{root_dir}test_set_clean.tsv", device=device)
train_iter, val_iter, test_iter = iters
ntokens, nlabels = len(TEXT.vocab), len(LABEL.vocab)

model = gensim.models.KeyedVectors.load_word2vec_format(f'{root_dir}100d_embeds_new.txt', unicode_errors='ignore', binary=False)
# weights = torch.FloatTensor(model.wv.vectors)
weights = torch.FloatTensor(model.vectors)
embedding = nn.Embedding.from_pretrained(weights,freeze=True, padding_idx=1)
encoder = Encoder(EMBED_DIM, HIDDEN_DIM, N_LAYERS,
                  dropout=DROPOUT, bidirectional=False, rnn_type='LSTM')

attention_dim = HIDDEN_DIM
attention = BahdanauAttention(attention_dim, attention_dim)

classifier_model = Classifier_GANLike(embedding, encoder, attention, attention_dim, nlabels)
topic_decoder = [nn.Sequential(nn.Dropout(TOPIC_DROPOUT), nn.Linear(attention_dim, NUM_TOPICS), nn.LogSoftmax())]

classifier_model.to(device)
topic_decoder[0].to(device)

classify_criterion = nn.CrossEntropyLoss()
topic_criterion = nn.KLDivLoss(size_average=False)

classify_optim = Optim('adam',  0.0001,  5)
topic_optim = Optim('adam',  0.0001,  5)

params = []
num_train = 0

for n, p in classifier_model.named_parameters():
    if n == "embedding.weight":
        continue
    if not p.requires_grad:
        print("OMG", p)
        p.requires_grad = True
    p.data.uniform_(-PARAM_INIT, PARAM_INIT)
    params.append(p)
    num_train += p.nelement()

for p in topic_decoder[0].parameters():
    if not p.requires_grad:
        print ("OMG", p)
        p.requires_grad = True
    p.data.uniform_(-PARAM_INIT, PARAM_INIT)
    num_train += p.nelement()

print("Trainable parameters", num_train)

#classify_optim.set_parameters(classifier_model.parameters())
classify_optim.set_parameters(params) # Does not include embedding parameters

topic_optim.set_parameters(topic_decoder[0].parameters())
#torch.autograd.set_detect_anomaly(True)
try:
    best_valid_loss = None
    best_model = None

    #pretraining the classifier
    for epoch in range(1, PRETRAIN_EPOCHS+1):
        pretrain_classifier(classifier_model, train_iter, classify_optim, classify_criterion,nlabels, epoch)
        loss = evaluate(classifier_model, topic_decoder, val_iter, classify_criterion, topic_criterion, nlabels, TOPIC_LOSS)
        #oodLoss = evaluate(classifier_model, outdomain_test_iter[0], classify_criterion, args, datatype="oodtest")

        if not best_valid_loss or loss < best_valid_loss:
            best_valid_loss = loss
            print("Updating best pretrained_model")
            best_model = copy.deepcopy(classifier_model)
            torch.save(best_model, f"{root_dir}pretrained_bestmodel")
        torch.save(classifier_model, f"{root_dir}pretrained_latestmodel")

    print("Done pretraining")
    print()

    best_valid_loss = None
    best_model = None
    #alternating training like GANs
    for epoch in range(1, EPOCHS + 1):
        for t_step in range(1, T_STEPS+1):
            print()
            print("Training topic predictor")
            train_topic_predictor(classifier_model, topic_decoder[-1], train_iter, topic_optim, topic_criterion, NUM_TOPICS, TOPIC_LOSS, epoch, T_STEPS)

        if reset_classifier:
            for n, p in classifier_model.named_parameters():
                if n == "embeddings.weight":
                    continue
                if not p.requires_grad:
                    print ("OMG", p)
                    p.requires_grad = True
                p.data.uniform_(-PARAM_INIT, PARAM_INIT)

        for c_step in range(1, C_STEPS+1):
            print()
            print("Training classifier")
            train_classifier(classifier_model, topic_decoder, train_iter, classify_optim, classify_criterion, topic_criterion, nlabels, NUM_TOPICS, TOPIC_LOSS, epoch, C_STEPS)
            loss = evaluate(classifier_model, topic_decoder, val_iter, classify_criterion, topic_criterion, nlabels, TOPIC_LOSS)
            #oodLoss = evaluate(classifier_model, outdomain_test_iter[0], classify_criterion, args, datatype="oodtest")

        #creating a new instance of a decoder
        attention_dim = HIDDEN_DIM # if not args.bi else 2*args.hidden
        if BOTTLENECK_DIM == 0:
            topic_decoder.append(nn.Sequential(nn.Dropout(TOPIC_DROPOUT), nn.Linear(attention_dim, NUM_TOPICS), nn.LogSoftmax()))
        else:
            topic_decoder.append(nn.Sequential(nn.Dropout(TOPIC_DROPOUT), nn.Linear(BOTTLENECK_DIM, NUM_TOPICS), nn.LogSoftmax()))

        #attaching a new optimizer to the new topic decode
        topic_decoder[-1].to(device)
        topic_optim = Optim('adam', LR, GRAD_CLIP)
        for p in topic_decoder[-1].parameters():
            if not p.requires_grad:
                print ("OMG", p)
                p.requires_grad = True
            p.data.uniform_(-PARAM_INIT, PARAM_INIT)
        topic_optim.set_parameters(topic_decoder[-1].parameters())

        if not best_valid_loss or loss < best_valid_loss:
            best_valid_loss = loss
            print ("Updating best model")
            best_model = copy.deepcopy(classifier_model)
            torch.save(best_model, f"{root_dir}bestmodel")
        torch.save(classifier_model, f"{root_dir}latestmodel")

except KeyboardInterrupt:
    print("[Ctrl+C] Training stopped!")


# if args.finetune:
#   best_valid_loss = None
#   for c_step in range(1, args.c_steps+1):
#     print()
#     print("Fine-tuning classifier")
#     train_classifier(classifier_model, None, train_iter, classify_optim, classify_criterion, None, args, c_step, args.c_steps)
#     loss = evaluate(classifier_model, topic_decoder, val_iter, classify_criterion, topic_criterion, args)

#     if not best_valid_loss or loss < best_valid_loss:
#       best_valid_loss = loss
#       print ("Updating best model")
#       best_model = copy.deepcopy(classifier_model)
#       torch.save(best_model, args.save_dir+"/"+args.model_name+"finetune_bestmodel")
#     torch.save(classifier_model, args.save_dir+"/"+args.model_name+"finetune_latestmodel")


trainloss = evaluate(best_model, topic_decoder, train_iter, classify_criterion, topic_criterion, nlabels, TOPIC_LOSS, datatype='train', itos=TEXT.vocab.itos, litos=LABEL.vocab.itos)
valloss = evaluate(best_model, topic_decoder, val_iter, classify_criterion, topic_criterion, nlabels, TOPIC_LOSS, datatype='valid', itos=TEXT.vocab.itos, litos=LABEL.vocab.itos)
loss = evaluate(best_model, topic_decoder, test_iter, classify_criterion, topic_criterion, nlabels, TOPIC_LOSS, datatype='test', itos=TEXT.vocab.itos, litos=LABEL.vocab.itos)

def see_details(model, t_models, data, criterion, topic_criterion, nlabels, topic_loss, datatype='Valid', itos=None, litos=None):

    model.eval()

    accuracy, confusion_matrix = 0.0, np.zeros((nlabels, nlabels), dtype=int)
    males = { "cor" : {}, "inc": {}}
    females = { "cor" : {}, "inc": {}}
    for i in range(24):
        for label in ['cor', 'inc']:
            males[label][i] = 0
            females[label][i] = 0
            
    with torch.no_grad():
        for batch_num, batch in enumerate(data):
            x, lens = batch.text
            loc = batch.loc
            y = batch.label
            # if args.data in ["RT_GENDER"]:
            #   indices = batch.index.cpu().data.numpy()
            #   # print (indices.size())
            # else:
            #   indices = np.array(([0]*len(y)))
            padding_mask = x.ne(1).float()

            logits, energy, sentrep = model(x, padding_mask=padding_mask)

            _, max_ind = torch.max(logits, 1)
            equal = torch.eq(max_ind, y)
            accuracy += int(torch.sum(equal))
            
            for i in range(equal.shape[0]):
                if y[i] == 0:
                    if equal[i] == True:
                        males['cor'][int(loc[i].item())] += 1
                    else:
                        males['inc'][int(loc[i].item())] += 1
                else:
                    if equal[i] == True:
                        females['cor'][int(loc[i].item())] += 1
                    else:
                        females['inc'][int(loc[i].item())] += 1
                        
            for j, i in zip(max_ind, y):
                confusion_matrix[int(i),int(j)]+=1

    print(confusion_matrix)
    return males, females, accuracy, confusion_matrix

print(see_details(best_model, topic_decoder, test_iter, classify_criterion, topic_criterion, nlabels, TOPIC_LOSS, datatype='test', itos=TEXT.vocab.itos, litos=LABEL.vocab.itos))
