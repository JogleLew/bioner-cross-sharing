from __future__ import print_function

__author__ = 'max'
"""
Implementation of Bi-directional LSTM-CNNs-CRF model for NER.
"""

import sys
import os

sys.path.append(".")
sys.path.append("..")

import six
import json
import time
import random
import argparse
import uuid
import math
import importlib 

from path import Path
import numpy as np
import torch
from torch.optim import Adam, SGD, RMSprop
from neuronlp2.io import get_logger, bionlp_data, BioNLPWriter
from neuronlp2.models import FullySharedBiRecurrentCRF
from neuronlp2 import utils
from torch.nn.utils import clip_grad_norm
from allennlp.commands.elmo import ElmoEmbedder
from tensorboardX import SummaryWriter

uid = uuid.uuid4().hex[:6]

def evaluate(output_file):
    score_file = "tmp/score_%s" % str(uid)
    os.system("examples/eval/bionlpeval < %s > %s" % (output_file, score_file))
    with open(score_file, 'r') as fin:
        fin.readline()
        line = fin.readline()
        fields = line.split(";")
        acc = float(fields[0].split(":")[1].strip()[:-1])
        precision = float(fields[1].split(":")[1].strip()[:-1])
        recall = float(fields[2].split(":")[1].strip()[:-1])
        f1 = float(fields[3].split(":")[1].strip())
    return acc, precision, recall, f1

def reverse_reflect(reflects, length):
    # task-to-all reflect -> all-to-task reflect
    result = []
    for reflect in reflects:
        new_reflect = [0 for _ in range(length)]
        for task_idx, all_idx in enumerate(reflect):
            new_reflect[all_idx] = task_idx
        result.append(new_reflect)
    return result

def main():
    parser = argparse.ArgumentParser(description='Tuning with Multitask bi-directional RNN-CNN-CRF')
    parser.add_argument('--config', help='Config file (Python file format)', default="config_multitask.py")
    parser.add_argument('--grid', help='Grid Search Options', default="{}")
    args = parser.parse_args()
    logger = get_logger("Multi-Task")
    use_gpu = torch.cuda.is_available()

    # Config Tensorboard Writer
    log_writer = SummaryWriter()

    # Load from config file
    spec = importlib.util.spec_from_file_location("config", args.config)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)
    config = config_module.entries

    # Load options from grid search
    options = eval(args.grid)
    for k, v in options.items():
        if isinstance(v, six.string_types):
            cmd = "%s = \"%s\"" % (k, v)
        else:
            cmd = "%s = %s" % (k, v)
            log_writer.add_scalar(k, v, 1) 
        exec(cmd)

    # Load embedding dict
    embedding = config.embedding.embedding_type
    embedding_path = config.embedding.embedding_dict
    embedd_dict, embedd_dim = utils.load_embedding_dict(embedding, embedding_path)

    # Collect data path
    data_dir = config.data.data_dir
    data_names = config.data.data_names
    train_paths = [os.path.join(data_dir, data_name, "train.tsv") for data_name in data_names]
    dev_paths = [os.path.join(data_dir, data_name, "devel.tsv") for data_name in data_names]
    test_paths = [os.path.join(data_dir, data_name, "test.tsv") for data_name in data_names]

    # Create alphabets
    logger.info("Creating Alphabets")
    if not os.path.exists('tmp'):
        os.mkdir('tmp')
    word_alphabet, char_alphabet, pos_alphabet, chunk_alphabet, ner_alphabet, ner_alphabet_task, label_reflect  = \
            bionlp_data.create_alphabets(os.path.join(Path(data_dir).abspath(), "alphabets", "_".join(data_names)), train_paths, 
                    data_paths=dev_paths + test_paths, use_cache=True,
                    embedd_dict=embedd_dict, max_vocabulary_size=50000)

    logger.info("Word Alphabet Size: %d" % word_alphabet.size())
    logger.info("Character Alphabet Size: %d" % char_alphabet.size())
    logger.info("POS Alphabet Size: %d" % pos_alphabet.size())
    logger.info("Chunk Alphabet Size: %d" % chunk_alphabet.size())
    logger.info("NER Alphabet Size: %d" % ner_alphabet.size())
    logger.info("NER Alphabet Size per Task: %s", str([task_alphabet.size() for task_alphabet in ner_alphabet_task]))

    #task_reflects = torch.LongTensor(reverse_reflect(label_reflect, ner_alphabet.size()))
    #if use_gpu:
    #    task_reflects = task_reflects.cuda()

    if embedding == 'elmo':
        logger.info("Loading ELMo Embedder")
        ee = ElmoEmbedder(
                options_file=config.embedding.elmo_option, 
                weight_file=config.embedding.elmo_weight, 
                cuda_device=config.embedding.elmo_cuda
        )
    else:
        ee = None

    logger.info("Reading Data")

    # Prepare dataset
    data_trains = [bionlp_data.read_data_to_variable(train_path, word_alphabet, char_alphabet, pos_alphabet,
                        chunk_alphabet, ner_alphabet_task[task_id], use_gpu=use_gpu, elmo_ee=ee) 
                        for task_id, train_path in enumerate(train_paths)]
    num_data = [sum(data_train[1]) for data_train in data_trains]
    num_labels = ner_alphabet.size()
    num_labels_task = [task_item.size() for task_item in ner_alphabet_task]

    data_devs = [bionlp_data.read_data_to_variable(dev_path, word_alphabet, char_alphabet, pos_alphabet,
                        chunk_alphabet, ner_alphabet_task[task_id], use_gpu=use_gpu, volatile=True, elmo_ee=ee)
                        for task_id, dev_path in enumerate(dev_paths)]

    data_tests = [bionlp_data.read_data_to_variable(test_path, word_alphabet, char_alphabet, pos_alphabet,
                        chunk_alphabet, ner_alphabet_task[task_id], use_gpu=use_gpu, volatile=True, elmo_ee=ee)
                        for task_id, test_path in enumerate(test_paths)]

    writer = BioNLPWriter(word_alphabet, char_alphabet, pos_alphabet, chunk_alphabet, ner_alphabet)

    def construct_word_embedding_table():
        scale = np.sqrt(3.0 / embedd_dim)
        table = np.empty([word_alphabet.size(), embedd_dim], dtype=np.float32)
        table[bionlp_data.UNK_ID, :] = np.random.uniform(-scale, scale, [1, embedd_dim]).astype(np.float32)
        oov = 0
        for word, index in word_alphabet.items():
            if not embedd_dict == None and word in embedd_dict:
                embedding = embedd_dict[word]
            elif not embedd_dict == None and word.lower() in embedd_dict:
                embedding = embedd_dict[word.lower()]
            else:
                embedding = np.random.uniform(-scale, scale, [1, embedd_dim]).astype(np.float32)
                oov += 1
            table[index, :] = embedding
        print('oov: %d' % oov)
        return torch.from_numpy(table)

    word_table = construct_word_embedding_table()
    logger.info("constructing network...")

    # Construct network
    window = 3
    num_layers = 1
    mode = config.rnn.mode
    hidden_size = config.rnn.hidden_size
    char_dim = config.rnn.char_dim
    num_filters = config.rnn.num_filters
    tag_space = config.rnn.tag_space
    bigram = config.rnn.bigram
    attention_mode = config.rnn.attention
    if config.rnn.dropout == 'std':
        network = FullySharedBiRecurrentCRF(len(data_trains), embedd_dim, word_alphabet.size(),
                                     char_dim, char_alphabet.size(),
                                     num_filters, window,
                                     mode, hidden_size, num_layers, num_labels,
                                     num_labels_task=num_labels_task,
                                     tag_space=tag_space, embedd_word=word_table,
                                     p_in=config.rnn.p, p_rnn=config.rnn.p, bigram=bigram,
                                     elmo=(embedding == 'elmo'), attention_mode=attention_mode,
                                     adv_loss_coef=config.multitask.adv_loss_coef, 
                                     diff_loss_coef=config.multitask.diff_loss_coef, 
                                     char_level_rnn=config.rnn.char_level_rnn)
    else:
        raise NotImplementedError

    if use_gpu:
        network.cuda()

    # Prepare training
    unk_replace = config.embedding.unk_replace
    num_epochs = config.training.num_epochs
    batch_size = config.training.batch_size
    lr = config.training.learning_rate
    momentum = config.training.momentum
    alpha = config.training.alpha
    lr_decay = config.training.lr_decay
    schedule = config.training.schedule
    gamma = config.training.gamma

    # optim = SGD(network.parameters(), lr=lr, momentum=momentum, weight_decay=gamma, nesterov=True)
    optim = RMSprop(network.parameters(), lr=lr, alpha=alpha, momentum=momentum, weight_decay=gamma)
    logger.info("Network: %s, num_layer=%d, hidden=%d, filter=%d, tag_space=%d, crf=%s" % (
        mode, num_layers, hidden_size, num_filters, tag_space, 'bigram' if bigram else 'unigram'))
    logger.info("training: l2: %f, (#training data: %s, batch: %d, dropout: %.2f, unk replace: %.2f)" % (
        gamma, num_data, batch_size, config.rnn.p, unk_replace))

    num_batches = [x // batch_size + 1 for x in num_data]
    dev_f1 = [0.0 for x in num_data]
    dev_acc = [0.0 for x in num_data]
    dev_precision = [0.0 for x in num_data]
    dev_recall = [0.0 for x in num_data]
    test_f1 = [0.0 for x in num_data]
    test_acc = [0.0 for x in num_data]
    test_precision = [0.0 for x in num_data]
    test_recall = [0.0 for x in num_data]
    best_epoch = [0 for x in num_data]

    # Training procedure
    for epoch in range(1, num_epochs + 1):
        print('Epoch %d (%s(%s), learning rate=%.4f, decay rate=%.4f (schedule=%d)): ' % (
            epoch, mode, config.rnn.dropout, lr, lr_decay, schedule))
        train_err = 0.
        train_total = 0.

        # Gradient decent on training data
        start_time = time.time()
        num_back = 0
        network.train()
        batch_count = 0
        for batch in range(1, 2 * num_batches[0] + 1):
            r = random.random()
            task_id = 0 if r <= 0.5 else random.randint(1, len(num_data) - 1)
            batch_count += 1
            word, char, _, _, labels, masks, lengths, elmo_embedding = bionlp_data.get_batch_variable(data_trains[task_id], batch_size, unk_replace=unk_replace)

            optim.zero_grad()
            loss, task_loss, adv_loss, diff_loss = network.loss(task_id, word, char, labels, mask=masks, elmo_word=elmo_embedding)
            #log_writer.add_scalars(
            #        'train_loss_task' + str(task_id), 
            #        {'all_loss': loss, 'task_loss': task_loss, 'adv_loss': adv_loss, 'diff_loss': diff_loss}, 
            #        (epoch - 1) * (num_batches[task_id] + 1) + batch
            #) 
            #log_writer.add_scalars(
            #        'train_loss_overview', 
            #        {'all_loss': loss, 'task_loss': task_loss, 'adv_loss': adv_loss, 'diff_loss': diff_loss}, 
            #        (epoch - 1) * (sum(num_batches) + 1) + batch_count 
            #) 
            loss.backward()
            clip_grad_norm(network.parameters(), 5.0)
            optim.step()

            num_inst = word.size(0)
            train_err += loss.data[0] * num_inst
            train_total += num_inst

            time_ave = (time.time() - start_time) / batch
            time_left = (2 * num_batches[0] - batch) * time_ave

            # update log
            if batch % 100 == 0:
                sys.stdout.write("\b" * num_back)
                sys.stdout.write(" " * num_back)
                sys.stdout.write("\b" * num_back)
                log_info = 'train: %d/%d loss: %.4f, time left (estimated): %.2fs' % (
                    batch, 2 * num_batches[0], train_err / train_total, time_left)
                sys.stdout.write(log_info)
                sys.stdout.flush()
                num_back = len(log_info)

        sys.stdout.write("\b" * num_back)
        sys.stdout.write(" " * num_back)
        sys.stdout.write("\b" * num_back)
        print('train: %d loss: %.4f, time: %.2fs' % (2 * num_batches[0], train_err / train_total, time.time() - start_time))

        # Evaluate performance on dev data
        network.eval()
        for task_id in range(len(num_batches)):
            tmp_filename = 'tmp/%s_dev%d%d' % (str(uid), epoch, task_id)
            writer.start(tmp_filename)

            for batch in bionlp_data.iterate_batch_variable(data_devs[task_id], batch_size):
                word, char, pos, chunk, labels, masks, lengths, elmo_embedding = batch
                preds, _ = network.decode(task_id, word, char, target=labels, mask=masks, 
                                         leading_symbolic=bionlp_data.NUM_SYMBOLIC_TAGS, elmo_word=elmo_embedding)
                writer.write(word.data.cpu().numpy(), pos.data.cpu().numpy(), chunk.data.cpu().numpy(),
                         preds.cpu().numpy(), labels.data.cpu().numpy(), lengths.cpu().numpy())
            writer.close()
            acc, precision, recall, f1 = evaluate(tmp_filename)
            log_writer.add_scalars(
                    'dev_task' + str(task_id), 
                    {'accuracy': acc, 'precision': precision, 'recall': recall, 'f1': f1}, 
                    epoch
            ) 
            print('dev acc: %.2f%%, precision: %.2f%%, recall: %.2f%%, F1: %.2f%%' % (acc, precision, recall, f1))

            if dev_f1[task_id] < f1:
                dev_f1[task_id] = f1
                dev_acc[task_id] = acc
                dev_precision[task_id] = precision
                dev_recall[task_id] = recall
                best_epoch[task_id] = epoch

                # Evaluate on test data when better performance detected
                tmp_filename = 'tmp/%s_test%d%d' % (str(uid), epoch, task_id)
                writer.start(tmp_filename)

                for batch in bionlp_data.iterate_batch_variable(data_tests[task_id], batch_size):
                    word, char, pos, chunk, labels, masks, lengths, elmo_embedding = batch
                    preds, _ = network.decode(task_id, word, char, target=labels, mask=masks, 
                                          leading_symbolic=bionlp_data.NUM_SYMBOLIC_TAGS, elmo_word=elmo_embedding)
                    writer.write(word.data.cpu().numpy(), pos.data.cpu().numpy(), chunk.data.cpu().numpy(),
                             preds.cpu().numpy(), labels.data.cpu().numpy(), lengths.cpu().numpy())
                writer.close()
                test_acc[task_id], test_precision[task_id], test_recall[task_id], test_f1[task_id] = evaluate(tmp_filename)
                log_writer.add_scalars(
                        'test_task' + str(task_id), 
                        {'accuracy': test_acc[task_id], 'precision': test_precision[task_id], 
                            'recall': test_recall[task_id], 'f1': test_f1[task_id]}, 
                        epoch
                ) 

            print("================================================================================")
            print("dataset: %s" % data_names[task_id])
            print("best dev  acc: %.2f%%, precision: %.2f%%, recall: %.2f%%, F1: %.2f%% (epoch: %d)" % (
                dev_acc[task_id], dev_precision[task_id], dev_recall[task_id], dev_f1[task_id], best_epoch[task_id]))
            print("best test acc: %.2f%%, precision: %.2f%%, recall: %.2f%%, F1: %.2f%% (epoch: %d)" % (
                test_acc[task_id], test_precision[task_id], test_recall[task_id], test_f1[task_id], best_epoch[task_id]))
            print("================================================================================\n")

            if epoch % schedule == 0:
                # lr = learning_rate / (1.0 + epoch * lr_decay)
                # optim = SGD(network.parameters(), lr=lr, momentum=momentum, weight_decay=gamma, nesterov=True)
                lr = lr * lr_decay
                optim.param_groups[0]['lr'] = lr

    # writer.export_scalars_to_json("./all_scalars.json")
    writer.close()

if __name__ == '__main__':
    main()
