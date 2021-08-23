from __future__ import print_function

import argparse
import os, sys
import time
import numpy as np
import torch
import torch.nn as nn
import copy
import codecs
import random
import pandas as pd
from torchtext.legacy import data
from torchtext import datasets
import gensim
from gensim.models import Word2Vec

from model import *
from optim import Optim
from IPython.core.debugger import set_trace

root_dir="model/"
BATCH_SIZE = 32
SEED = 1337
EMBED_DIM = 100
HIDDEN_DIM = 128 # Default from paper
N_LAYERS = 1 # Default from paper
DROPOUT = 0 # Default from paper
TOPIC_DROPOUT = 0.2 # Default from paper
NUM_TOPICS = 34 #Number of politicians in training set
GRAD_CLIP = 5 # Default from paper
LR = 0.0001 # from Kumar (2019)
PRETRAIN_EPOCHS = 3
EPOCHS = 3
C_STEPS = 3
T_STEPS = 10
PARAM_INIT = 0.1
TOPIC_LOSS = "kl" #[mse|ce|kl]
BOTTLENECK_DIM = 0
#reset_classifier = True


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

  train_iter, val_iter, test_iter = data.BucketIterator.splits((train, val, test), batch_sizes=(batch_size, 256, 256), device=device, repeat=False, sort_key=lambda x: len(x.text))

  return (train_iter, val_iter, test_iter), TEXT, LABEL, TOPICS, INDEX, LOC

def seed_everything(seed, cuda=False):
  # Set the random seed manually for reproducibility.
  np.random.seed(seed)
  random.seed(seed)
  torch.manual_seed(seed)
  if cuda:
    torch.cuda.manual_seed_all(seed)

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
  print("[Training Accuracy]: {}/{} : {:.3f}%".format(epoch, accuracy, len(data.dataset), accuracy / len(data.dataset) * 100))
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
      topic_loss = criterion(topic_logprobs, topics)
    elif topic_loss == "ce":
      topic_loss = torch.sum(topic_logprobs*topics)
    else:
      g = (topics - torch.exp(topic_logprobs))
      topic_loss = (g*g).sum(dim=-1).mean()

    loss = topic_loss
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
    attention_file = codecs.open(f"{root_dir}attention.txt", "w", encoding="utf8")

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
  # print(confusion_matrix)
  return (total_loss / len(data)) + (total_topic_loss / len(data))

cuda = torch.cuda.is_available()
device = torch.device("cpu") if not cuda else torch.device("cuda")
seed_everything(seed=SEED, cuda=cuda)
iters, TEXT, LABEL, TOPICS, INDEX, SUBS = create_iterators(BATCH_SIZE, f"{root_dir}subset_odds.tsv", f"{root_dir}valid_set_clean.tsv", f"{root_dir}test_set_clean.tsv")
train_iter, val_iter, test_iter = iters
ntokens, nlabels = len(TEXT.vocab), len(LABEL.vocab)
model = gensim.models.KeyedVectors.load_word2vec_format(f'{root_dir}100d_embeds.txt', unicode_errors='ignore', binary=True)
weights = torch.FloatTensor(model.wv.vectors)
embedding = nn.Embedding.from_pretrained(weights, padding_idx=1)
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

classify_optim = Optim('adam',  LR,  GRAD_CLIP)
topic_optim = Optim('adam',  LR,  GRAD_CLIP)

for p in classifier_model.parameters():
    if not p.requires_grad:
      print ("OMG", p)
      p.requires_grad = True
    p.data.uniform_(-PARAM_INIT, PARAM_INIT)

for p in topic_decoder[0].parameters():
  if not p.requires_grad:
    print ("OMG", p)
    p.requires_grad = True
  p.data.uniform_(-PARAM_INIT, PARAM_INIT)

classify_optim.set_parameters(classifier_model.parameters())
topic_optim.set_parameters(topic_decoder[0].parameters())

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
      for p in classifier_model.parameters():
        if not p.requires_grad:
          print ("OMG", p)
          p.requires_grad = True
        p.data.uniform_(-PARAM_INIT, PARAM_INIT)

    for c_step in range(1, C_STEPS+1):
      print()
      print("Training classifier")
      train_classifier(classifier_model, topic_decoder, train_iter, classify_optim, classify_criterion, topic_criterion, NLABELS, NUM_TOPICS, TOPIC_LOSS, epoch, C_STEPS)
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

# if args.ood_test_file:
#   loss = evaluate(best_model, topic_decoder, test_iter, classify_criterion, topic_criterion, args, datatype=os.path.basename(args.ood_test_file).replace(".txt", "").replace(".tsv", ""), itos=TEXT.vocab.itos, litos=LABEL.vocab.itos)

