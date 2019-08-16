__author__ = 'max'

import sys
import os.path
import random
import numpy as np
from .reader import BioNLPReader
from .alphabet import Alphabet
from .logger import get_logger
from . import utils
import torch
from torch.autograd import Variable

# Special vocabulary symbols - we always put them at the start.
PAD = "_PAD"
PAD_POS = "_PAD_POS"
PAD_CHUNK = "_PAD_CHUNK"
PAD_NER = "_PAD_NER"
PAD_CHAR = "_PAD_CHAR"
_START_VOCAB = [PAD,]

UNK_ID = 0
PAD_ID_WORD = 1
PAD_ID_CHAR = 1
PAD_ID_TAG = 0

NUM_SYMBOLIC_TAGS = 1

_buckets = [5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100, 140]


def create_alphabets(alphabet_directory, train_paths, data_paths=None, max_vocabulary_size=50000, embedd_dict=None, min_occurence=1, normalize_digits=True, use_cache=True):

    def expand_vocab():
        vocab_set = set(vocab_list)
        for data_path in data_paths:
            # logger.info("Processing data: %s" % data_path)
            with open(data_path, 'r') as file:
                word_count = 0
                for line in file:
                    # line = line.decode('utf-8')
                    line = line.strip()
                    if len(line) == 0:
                        word_count = 0
                        continue

                    word_count = word_count + 1
                    line = str(word_count) + '\t' + line
                    tokens = line.split('\t')
                    # word = utils.DIGIT_RE.sub(b"0", tokens[1]) if normalize_digits else tokens[1]
                    word = utils.DIGIT_RE.sub("0", tokens[1]) if normalize_digits else tokens[1]
                    ner = tokens[2]

                    pos_alphabet.add("O")
                    chunk_alphabet.add("O")
                    ner_alphabet.add(ner)

                    if word not in vocab_set and (word in embedd_dict or word.lower() in embedd_dict):
                        vocab_set.add(word)
                        vocab_list.append(word)

    logger = get_logger("Create Alphabets")
    if isinstance(train_paths, str):
        train_paths = [train_paths]
    word_alphabet = Alphabet('word', defualt_value=True, singleton=True)
    char_alphabet = Alphabet('character', defualt_value=True)
    pos_alphabet = Alphabet('pos')
    chunk_alphabet = Alphabet('chunk')
    ner_alphabet = Alphabet('ner')
    ner_alphabet_task = [Alphabet('ner' + str(i)) for i in range(len(train_paths))]
    task_tag_reflect = [[] for _ in ner_alphabet_task]

    if not os.path.isdir(alphabet_directory) or not use_cache:
        logger.info("Creating Alphabets: %s" % alphabet_directory)

        char_alphabet.add(PAD_CHAR)
        pos_alphabet.add(PAD_POS)
        chunk_alphabet.add(PAD_CHUNK)
        ner_alphabet.add(PAD_NER)
        [task_alphabet.add(PAD_NER) for task_alphabet in ner_alphabet_task]
        [reflect.append(0) for reflect in task_tag_reflect]

        vocab = dict()
        for task_id, train_path in enumerate(train_paths):
            with open(train_path, 'r') as file:
                word_count = 0
                for line in file:
                    # line = line.decode('utf-8')
                    line = line.strip()
                    if len(line) == 0:
                        word_count = 0
                        continue

                    word_count = word_count + 1
                    line = str(word_count) + '\t' + line
                    tokens = line.split('\t')
                    for char in tokens[1]:
                        char_alphabet.add(char)

                    # word = utils.DIGIT_RE.sub(b"0", tokens[1]) if normalize_digits else tokens[1]
                    word = utils.DIGIT_RE.sub("0", tokens[1]) if normalize_digits else tokens[1]
                    word = str(word)
                    ner = tokens[2]

                    pos_alphabet.add("O")
                    chunk_alphabet.add("O")
                    ner_alphabet.add(ner)
                    ner_label_id = ner_alphabet.get_index(ner)
                    ner_alphabet_task[task_id].add(ner)
                    ner_label_id_task = ner_alphabet_task[task_id].get_index(ner)
                    if ner_label_id_task == len(task_tag_reflect[task_id]):
                        task_tag_reflect[task_id].append(ner_label_id)

                    if word in vocab:
                        vocab[word] += 1
                    else:
                        vocab[word] = 1

        # collect singletons
        singletons = set([word for word, count in vocab.items() if count == 1])

        # if a singleton is in pretrained embedding dict, set the count to 2
        if embedd_dict is not None:
            for word in vocab.keys():
                if word in embedd_dict or word.lower() in embedd_dict:
                    vocab[word] += 1

        vocab_list = _START_VOCAB + sorted(vocab, key=vocab.get, reverse=True)
        logger.info("Total Vocabulary Size: %d" % len(vocab_list))
        logger.info("Total Singleton Size:  %d" % len(singletons))
        vocab_list = [word for word in vocab_list if word in _START_VOCAB or vocab[word] > min_occurence]
        logger.info("Total Vocabulary Size (w.o rare words): %d" % len(vocab_list))

        if len(vocab_list) > max_vocabulary_size:
            vocab_list = vocab_list[:max_vocabulary_size]

        if data_paths is not None and embedd_dict is not None:
            expand_vocab()

        for word in vocab_list:
            word_alphabet.add(word)
            if word in singletons:
                word_alphabet.add_singleton(word_alphabet.get_index(word))

        word_alphabet.save(alphabet_directory)
        char_alphabet.save(alphabet_directory)
        pos_alphabet.save(alphabet_directory)
        chunk_alphabet.save(alphabet_directory)
        ner_alphabet.save(alphabet_directory)
        [task_alphabet.save(alphabet_directory) for task_alphabet in ner_alphabet_task]
    else:
        word_alphabet.load(alphabet_directory)
        char_alphabet.load(alphabet_directory)
        pos_alphabet.load(alphabet_directory)
        chunk_alphabet.load(alphabet_directory)
        ner_alphabet.load(alphabet_directory)
        [task_alphabet.load(alphabet_directory) for task_alphabet in ner_alphabet_task]

    word_alphabet.close()
    char_alphabet.close()
    pos_alphabet.close()
    chunk_alphabet.close()
    ner_alphabet.close()
    [task_alphabet.close() for task_alphabet in ner_alphabet_task]
    logger.info("Word Alphabet Size (Singleton): %d (%d)" % (word_alphabet.size(), word_alphabet.singleton_size()))
    logger.info("Character Alphabet Size: %d" % char_alphabet.size())
    logger.info("POS Alphabet Size: %d" % pos_alphabet.size())
    logger.info("Chunk Alphabet Size: %d" % chunk_alphabet.size())
    logger.info("NER Alphabet Size: %d" % ner_alphabet.size())
    logger.info("NER Alphabet Size per Task: %s", str([task_alphabet.size() for task_alphabet in ner_alphabet_task]))
    if len(ner_alphabet_task) > 1:
        return word_alphabet, char_alphabet, pos_alphabet, chunk_alphabet, ner_alphabet, ner_alphabet_task, task_tag_reflect
    return word_alphabet, char_alphabet, pos_alphabet, chunk_alphabet, ner_alphabet


def read_data(source_path, word_alphabet, char_alphabet, pos_alphabet, chunk_alphabet, ner_alphabet, max_size=None,
              normalize_digits=True, data_reduce=1.0):
    data = [[] for _ in _buckets]
    max_char_length = [0 for _ in _buckets]
    print('Reading data from %s' % source_path)
    counter = 0
    reader = BioNLPReader(source_path, word_alphabet, char_alphabet, pos_alphabet, chunk_alphabet, ner_alphabet)
    inst = reader.getNext(normalize_digits)
    while inst is not None and (not max_size or counter < max_size):
        if random.random() > data_reduce:
            continue
        counter += 1
        if counter % 10000 == 0:
            print("reading data: %d" % counter)

        inst_size = inst.length()
        sent = inst.sentence
        for bucket_id, bucket_size in enumerate(_buckets):
            if inst_size < bucket_size:
                data[bucket_id].append([sent.word_ids, sent.char_id_seqs, inst.pos_ids, inst.chunk_ids, inst.ner_ids])
                max_len = max([len(char_seq) for char_seq in sent.char_seqs])
                if max_char_length[bucket_id] < max_len:
                    max_char_length[bucket_id] = max_len
                break

        inst = reader.getNext(normalize_digits)
    reader.close()
    print("Total number of data: %d" % counter)
    return data, max_char_length


def get_batch(data, batch_size, word_alphabet=None, unk_replace=0.):
    data, max_char_length = data
    bucket_sizes = [len(data[b]) for b in range(len(_buckets))]
    total_size = float(sum(bucket_sizes))
    # A bucket scale is a list of increasing numbers from 0 to 1 that we'll use
    # to select a bucket. Length of [scale[i], scale[i+1]] is proportional to
    # the size if i-th training bucket, as used later.
    buckets_scale = [sum(bucket_sizes[:i + 1]) / total_size for i in range(len(bucket_sizes))]

    # Choose a bucket according to data distribution. We pick a random number
    # in [0, 1] and use the corresponding interval in train_buckets_scale.
    random_number = np.random.random_sample()
    bucket_id = min([i for i in range(len(buckets_scale)) if buckets_scale[i] > random_number])

    bucket_length = _buckets[bucket_id]
    char_length = min(utils.MAX_CHAR_LENGTH, max_char_length[bucket_id] + utils.NUM_CHAR_PAD)
    bucket_size = bucket_sizes[bucket_id]
    batch_size = min(bucket_size, batch_size)

    wid_inputs = np.empty([batch_size, bucket_length], dtype=np.int64)
    cid_inputs = np.empty([batch_size, bucket_length, char_length], dtype=np.int64)
    pid_inputs = np.empty([batch_size, bucket_length], dtype=np.int64)
    chid_inputs = np.empty([batch_size, bucket_length], dtype=np.int64)
    nid_inputs = np.empty([batch_size, bucket_length], dtype=np.int64)

    masks = np.zeros([batch_size, bucket_length], dtype=np.float32)
    single = np.zeros([batch_size, bucket_length], dtype=np.int64)

    for b in range(batch_size):
        wids, cid_seqs, pids, chids, nids = random.choice(data[bucket_id])

        inst_size = len(wids)
        # word ids
        wid_inputs[b, :inst_size] = wids
        wid_inputs[b, inst_size:] = PAD_ID_WORD
        for c, cids in enumerate(cid_seqs):
            cid_inputs[b, c, :len(cids)] = cids
            cid_inputs[b, c, len(cids):] = PAD_ID_CHAR
        cid_inputs[b, inst_size:, :] = PAD_ID_CHAR
        # pos ids
        pid_inputs[b, :inst_size] = pids
        pid_inputs[b, inst_size:] = PAD_ID_TAG
        # chunk ids
        chid_inputs[b, :inst_size] = chids
        chid_inputs[b, inst_size:] = PAD_ID_TAG
        # ner ids
        nid_inputs[b, :inst_size] = nids
        nid_inputs[b, inst_size:] = PAD_ID_TAG
        # masks
        masks[b, :inst_size] = 1.0

        if unk_replace:
            for j, wid in enumerate(wids):
                if word_alphabet.is_singleton(wid):
                    single[b, j] = 1

    if unk_replace:
        noise = np.random.binomial(1, unk_replace, size=[batch_size, bucket_length])
        wid_inputs = wid_inputs * (1 - noise * single)

    return wid_inputs, cid_inputs, pid_inputs, chid_inputs, nid_inputs, masks


def iterate_batch(data, batch_size, word_alphabet=None, unk_replace=0., shuffle=False):
    data, max_char_length = data
    bucket_sizes = [len(data[b]) for b in range(len(_buckets))]
    total_size = float(sum(bucket_sizes))
    bucket_indices = np.arange(len(_buckets))
    if shuffle:
        np.random.shuffle((bucket_indices))

    for bucket_id in bucket_indices:
        bucket_size = bucket_sizes[bucket_id]
        if bucket_size == 0:
            continue

        bucket_length = _buckets[bucket_id]
        char_length = min(utils.MAX_CHAR_LENGTH, max_char_length[bucket_id] + utils.NUM_CHAR_PAD)
        wid_inputs = np.empty([bucket_size, bucket_length], dtype=np.int64)
        cid_inputs = np.empty([bucket_size, bucket_length, char_length], dtype=np.int64)
        pid_inputs = np.empty([bucket_size, bucket_length], dtype=np.int64)
        chid_inputs = np.empty([bucket_size, bucket_length], dtype=np.int64)
        nid_inputs = np.empty([bucket_size, bucket_length], dtype=np.int64)

        masks = np.zeros([bucket_size, bucket_length], dtype=np.float32)
        single = np.zeros([bucket_size, bucket_length], dtype=np.int64)

        for i, inst in enumerate(data[bucket_id]):
            wids, cid_seqs, pids, chids, nids = inst
            inst_size = len(wids)
            # word ids
            wid_inputs[i, :inst_size] = wids
            wid_inputs[i, inst_size:] = PAD_ID_WORD
            for c, cids in enumerate(cid_seqs):
                cid_inputs[i, c, :len(cids)] = cids
                cid_inputs[i, c, len(cids):] = PAD_ID_CHAR
            cid_inputs[i, inst_size:, :] = PAD_ID_CHAR
            # pos ids
            pid_inputs[i, :inst_size] = pids
            pid_inputs[i, inst_size:] = PAD_ID_TAG
            # chunk ids
            chid_inputs[i, :inst_size] = chids
            chid_inputs[i, inst_size:] = PAD_ID_TAG
            # heads
            nid_inputs[i, :inst_size] = nids
            nid_inputs[i, inst_size:] = PAD_ID_TAG
            # masks
            masks[i, :inst_size] = 1.0
            if unk_replace:
                for j, wid in enumerate(wids):
                    if word_alphabet.is_singleton(wid):
                        single[i, j] = 1

        if unk_replace:
            noise = np.random.binomial(1, unk_replace, size=[bucket_size, bucket_length])
            wid_inputs = wid_inputs * (1 - noise * single)

        indices = None
        if shuffle:
            indices = np.arange(bucket_size)
            np.random.shuffle(indices)
        for start_idx in range(0, bucket_size, batch_size):
            if shuffle:
                excerpt = indices[start_idx:start_idx + batch_size]
            else:
                excerpt = slice(start_idx, start_idx + batch_size)
            yield wid_inputs[excerpt], cid_inputs[excerpt], pid_inputs[excerpt], chid_inputs[excerpt], \
                  nid_inputs[excerpt], masks[excerpt]


def read_data_to_variable(source_path, word_alphabet, char_alphabet, pos_alphabet, chunk_alphabet, ner_alphabet,
                          max_size=None, normalize_digits=True, use_gpu=False, volatile=False, elmo_ee=None, data_reduce=1.0):
    data, max_char_length = read_data(source_path, word_alphabet, char_alphabet, pos_alphabet,
                                      chunk_alphabet, ner_alphabet,
                                      max_size=max_size, normalize_digits=normalize_digits,data_reduce=data_reduce)
    bucket_sizes = [len(data[b]) for b in range(len(_buckets))]

    data_variable = []

    sys.stdout.flush()

    for bucket_id in range(len(_buckets)):
        bucket_size = bucket_sizes[bucket_id]
        if bucket_size == 0:
            data_variable.append((1, 1))
            continue

        bucket_length = _buckets[bucket_id]
        char_length = min(utils.MAX_CHAR_LENGTH, max_char_length[bucket_id] + utils.NUM_CHAR_PAD)
        wid_inputs = np.empty([bucket_size, bucket_length], dtype=np.int64)
        if elmo_ee:
            elmo_embeddings = np.empty([bucket_size, bucket_length, 1024], dtype=np.float32)
        cid_inputs = np.empty([bucket_size, bucket_length, char_length], dtype=np.int64)
        pid_inputs = np.empty([bucket_size, bucket_length], dtype=np.int64)
        chid_inputs = np.empty([bucket_size, bucket_length], dtype=np.int64)
        nid_inputs = np.empty([bucket_size, bucket_length], dtype=np.int64)

        masks = np.zeros([bucket_size, bucket_length], dtype=np.float32)
        single = np.zeros([bucket_size, bucket_length], dtype=np.int64)

        lengths = np.empty(bucket_size, dtype=np.int64)
        if elmo_ee:
            ee = elmo_ee

        word_str_batch = []
        process_info_len = 0
        for i, inst in enumerate(data[bucket_id]):
            wids, cid_seqs, pids, chids, nids = inst
            inst_size = len(wids)
            lengths[i] = inst_size
            # word ids
            wid_inputs[i, :inst_size] = wids
            wid_inputs[i, inst_size:] = PAD_ID_WORD
            for c, cids in enumerate(cid_seqs):
                cid_inputs[i, c, :len(cids)] = cids
                cid_inputs[i, c, len(cids):] = PAD_ID_CHAR
            cid_inputs[i, inst_size:, :] = PAD_ID_CHAR
            # elmo embeddeings
            if elmo_ee:
                word_str = [word_alphabet.get_instance(wid) for wid in wids]
                # while (len(word_str) < bucket_length):
                #     word_str.append("_PAD")
                word_str_batch.append(word_str)
            # pos ids
            pid_inputs[i, :inst_size] = pids
            pid_inputs[i, inst_size:] = PAD_ID_TAG
            # chunk ids
            chid_inputs[i, :inst_size] = chids
            chid_inputs[i, inst_size:] = PAD_ID_TAG
            # ner ids
            nid_inputs[i, :inst_size] = nids
            nid_inputs[i, inst_size:] = PAD_ID_TAG
            # masks
            masks[i, :inst_size] = 1.0
            for j, wid in enumerate(wids):
                if word_alphabet.is_singleton(wid):
                    single[i, j] = 1

            if i % 100 == 99 or i == len(data[bucket_id]) - 1:
                if elmo_ee:
                    # print(word_str_batch)
                    embeddings, mask = ee.batch_to_embeddings(word_str_batch) # embeddings: [batch_size, 3, sentence_len, 1024]
                    embeddings = embeddings.data.cpu().numpy()
                    batch_size, _, sentence_len, _ = embeddings.shape

                    if sentence_len < bucket_length:
                        zeros = np.zeros((batch_size, 3, bucket_length - sentence_len, 1024), dtype=embeddings.dtype)
                        embeddings = np.concatenate((embeddings, zeros), axis=2) # embeddings: [batch_size, 3, bucket_length, 1024]
                        
                    embeddings = embeddings[:, 2, :, :] # embeddings: [batch_size, bucket_length, 1024]
                    elmo_embeddings[(i + 1 - batch_size):(i + 1), :, :] = embeddings
                    word_str_batch = []

                sys.stdout.write("\b" * process_info_len)
                sys.stdout.write(" " * process_info_len)
                sys.stdout.write("\b" * process_info_len)
                process_info = ("sentence %d / %d, bucket %d / %d" % (i + 1, len(data[bucket_id]), bucket_id + 1, len(_buckets)))
                sys.stdout.write(process_info)
                sys.stdout.flush()
                process_info_len = len(process_info)

        sys.stdout.write("\b" * process_info_len)
        sys.stdout.write(" " * process_info_len)
        sys.stdout.write("\b" * process_info_len)
        words = Variable(torch.from_numpy(wid_inputs), volatile=volatile)
        chars = Variable(torch.from_numpy(cid_inputs), volatile=volatile)
        pos = Variable(torch.from_numpy(pid_inputs), volatile=volatile)
        chunks = Variable(torch.from_numpy(chid_inputs), volatile=volatile)
        ners = Variable(torch.from_numpy(nid_inputs), volatile=volatile)
        masks = Variable(torch.from_numpy(masks), volatile=volatile)
        single = Variable(torch.from_numpy(single), volatile=volatile)
        lengths = torch.from_numpy(lengths)
        if elmo_ee:
            elmo_embeddings = Variable(torch.from_numpy(elmo_embeddings), volatile=volatile)
        if use_gpu:
            words = words.cuda()
            chars = chars.cuda()
            pos = pos.cuda()
            chunks = chunks.cuda()
            ners = ners.cuda()
            masks = masks.cuda()
            single = single.cuda()
            lengths = lengths.cuda()
            if elmo_ee:
                elmo_embeddings = elmo_embeddings.cuda()

        data_variable.append((words, chars, pos, chunks, ners, masks, single, lengths, elmo_embeddings if elmo_ee else None))

    return data_variable, bucket_sizes


def get_batch_variable(data, batch_size, unk_replace=0.):
    data_variable, bucket_sizes = data
    total_size = float(sum(bucket_sizes))
    # A bucket scale is a list of increasing numbers from 0 to 1 that we'll use
    # to select a bucket. Length of [scale[i], scale[i+1]] is proportional to
    # the size if i-th training bucket, as used later.
    buckets_scale = [sum(bucket_sizes[:i + 1]) / total_size for i in range(len(bucket_sizes))]

    # Choose a bucket according to data distribution. We pick a random number
    # in [0, 1] and use the corresponding interval in train_buckets_scale.
    random_number = np.random.random_sample()
    bucket_id = min([i for i in range(len(buckets_scale)) if buckets_scale[i] > random_number])
    bucket_length = _buckets[bucket_id]

    words, chars, pos, chunks, ners, masks, single, lengths, elmo_embedding = data_variable[bucket_id]
    bucket_size = bucket_sizes[bucket_id]
    batch_size = min(bucket_size, batch_size)
    index = torch.randperm(bucket_size).long()[:batch_size]
    if words.is_cuda:
        index = index.cuda()

    words = words[index]
    if unk_replace:
        ones = Variable(single.data.new(batch_size, bucket_length).fill_(1))
        noise = Variable(masks.data.new(batch_size, bucket_length).bernoulli_(unk_replace).long())
        words = words * (ones - single[index] * noise)

    if elmo_embedding:
        return words, chars[index], pos[index], chunks[index], ners[index], masks[index], lengths[index], elmo_embedding[index]
    return words, chars[index], pos[index], chunks[index], ners[index], masks[index], lengths[index], None


def iterate_batch_variable(data, batch_size, unk_replace=0., shuffle=False):
    data_variable, bucket_sizes = data

    bucket_indices = np.arange(len(_buckets))
    if shuffle:
        np.random.shuffle((bucket_indices))

    for bucket_id in bucket_indices:
        bucket_size = bucket_sizes[bucket_id]
        bucket_length = _buckets[bucket_id]
        if bucket_size == 0:
            continue

        words, chars, pos, chunks, ners, masks, single, lengths, elmo_embedding = data_variable[bucket_id]
        if unk_replace:
            ones = Variable(single.data.new(bucket_size, bucket_length).fill_(1))
            noise = Variable(masks.data.new(bucket_size, bucket_length).bernoulli_(unk_replace).long())
            words = words * (ones - single * noise)

        indices = None
        if shuffle:
            indices = torch.randperm(bucket_size).long()
            if words.is_cuda:
                indices = indices.cuda()
        for start_idx in range(0, bucket_size, batch_size):
            if shuffle:
                excerpt = indices[start_idx:start_idx + batch_size]
            else:
                excerpt = slice(start_idx, start_idx + batch_size)
            if elmo_embedding:
                yield words[excerpt], chars[excerpt], pos[excerpt], chunks[excerpt], ners[excerpt], \
                  masks[excerpt], lengths[excerpt], elmo_embedding[excerpt]
            else:
                yield words[excerpt], chars[excerpt], pos[excerpt], chunks[excerpt], ners[excerpt], \
                  masks[excerpt], lengths[excerpt], None
