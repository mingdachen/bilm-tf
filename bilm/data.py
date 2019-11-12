# originally based on https://github.com/tensorflow/models/tree/master/lm_1b
import glob
import random
import pickle

import numpy as np

from typing import List


class Vocabulary(object):
    '''
    A token vocabulary.  Holds a map from token to ids and provides
    a method for encoding text to a sequence of ids.
    '''
    def __init__(self, filename, validate_file=False):
        '''
        filename = the vocabulary file.  It is a flat text file with one
            (normalized) token per line.  In addition, the file should also
            contain the special tokens <S>, </S>, <UNK> (case sensitive).
        '''
        self._id_to_word = []
        self._word_to_id = {}
        self._unk = -1
        self._desc_bos = -1
        self._cont_bos = -1
        self._eos = -1

        with open(filename) as f:
            idx = 0
            for line in f:
                word_name = line.strip()
                if word_name == '<DESC>':
                    self._desc_bos = idx
                elif word_name == '<CONTEXT>':
                    self._cont_bos = idx
                elif word_name == '</S>':
                    self._eos = idx
                elif word_name == '<UNK>':
                    self._unk = idx
                if word_name == '!!!MAXTERMID':
                    continue

                self._id_to_word.append(word_name)
                self._word_to_id[word_name] = idx
                idx += 1

        # check to ensure file has special tokens
        if validate_file:
            if self._desc_bos == -1 or self._cont_bos == -1 \
                    or self._eos == -1 or self._unk == -1:
                raise ValueError("Ensure the vocabulary file has "
                                 "<DESC>, <CONTEXT>, </S>, <UNK> tokens")

    @property
    def desc_bos(self):
        return self._desc_bos

    @property
    def cont_bos(self):
        return self._cont_bos

    @property
    def eos(self):
        return self._eos

    @property
    def unk(self):
        return self._unk

    @property
    def size(self):
        return len(self._id_to_word)

    def word_to_id(self, word):
        if word in self._word_to_id:
            return self._word_to_id[word]
        return self.unk

    def id_to_word(self, cur_id):
        return self._id_to_word[cur_id]

    def decode(self, cur_ids):
        """Convert a list of ids to a sentence, with space inserted."""
        return ' '.join([self.id_to_word(cur_id) for cur_id in cur_ids])

    def encode(self, sentence, if_context, reverse=False, split=True):
        """Convert a sentence to a list of ids, with special tokens added.
        Sentence is a single string with tokens separated by whitespace.

        If reverse, then the sentence is assumed to be reversed, and
            this method will swap the BOS/EOS tokens appropriately."""

        bos_token = self._cont_bos if if_context else self._desc_bos
        if split:
            word_ids = [
                self.word_to_id(cur_word) for cur_word in sentence.split()
            ]
        else:
            word_ids = [self.word_to_id(cur_word) for cur_word in sentence]

        if reverse:
            return np.array(
                [self.eos] + word_ids + [bos_token], dtype=np.int32)
        else:
            return np.array(
                [bos_token] + word_ids + [self.eos], dtype=np.int32)

    def encode_bow(self, sentence, num_steps):
        sent_ids = np.random.permutation(
            [self.word_to_id(w) for w in sentence])[:num_steps]
        return sent_ids


class UnicodeCharsVocabulary(Vocabulary):
    """Vocabulary containing character-level and word level information.

    Has a word vocabulary that is used to lookup word ids and
    a character id that is used to map words to arrays of character ids.

    The character ids are defined by ord(c) for c in word.encode('utf-8')
    This limits the total number of possible char ids to 256.
    To this we add 5 additional special ids: begin sentence, end sentence,
        begin word, end word and padding.

    WARNING: for prediction, we add +1 to the output ids from this
    class to create a special padding id (=0).  As a result, we suggest
    you use the `Batcher`, `TokenBatcher`, and `LMDataset` classes instead
    of this lower level class.  If you are using this lower level class,
    then be sure to add the +1 appropriately, otherwise embeddings computed
    from the pre-trained model will be useless.
    """
    def __init__(self, filename, max_word_length, **kwargs):
        super(UnicodeCharsVocabulary, self).__init__(filename, **kwargs)
        self._max_word_length = max_word_length

        # char ids 0-255 come from utf-8 encoding bytes
        # assign 256-300 to special chars
        self.bod_char = 256  # <begin description>
        self.boc_char = 261  # <begin context>
        self.eos_char = 257  # <end sentence>
        self.bow_char = 258  # <begin word>
        self.eow_char = 259  # <end word>
        self.pad_char = 260  # <padding>

        num_words = len(self._id_to_word)

        self._word_char_ids = np.zeros(
            [num_words, max_word_length],
            dtype=np.int32)

        # the charcter representation of the begin/end of sentence characters
        def _make_bos_eos(c):
            r = np.zeros([self.max_word_length], dtype=np.int32)
            r[:] = self.pad_char
            r[0] = self.bow_char
            r[1] = c
            r[2] = self.eow_char
            return r
        self.bod_chars = _make_bos_eos(self.bod_char)
        self.boc_chars = _make_bos_eos(self.boc_char)
        self.eos_chars = _make_bos_eos(self.eos_char)

        for i, word in enumerate(self._id_to_word):
            self._word_char_ids[i] = self._convert_word_to_char_ids(word)

        self._word_char_ids[self.desc_bos] = self.bod_chars
        self._word_char_ids[self.cont_bos] = self.boc_chars
        self._word_char_ids[self.eos] = self.eos_chars
        # TODO: properly handle <UNK>

    @property
    def word_char_ids(self):
        return self._word_char_ids

    @property
    def max_word_length(self):
        return self._max_word_length

    def _convert_word_to_char_ids(self, word):
        code = np.zeros([self.max_word_length], dtype=np.int32)
        code[:] = self.pad_char

        word_encoded = word.encode('utf-8', 'ignore')[:(self.max_word_length-2)]
        code[0] = self.bow_char
        for k, chr_id in enumerate(word_encoded, start=1):
            code[k] = chr_id
        code[len(word_encoded) + 1] = self.eow_char

        return code

    def word_to_char_ids(self, word):
        if word in self._word_to_id:
            return self._word_char_ids[self._word_to_id[word]]
        else:
            return self._convert_word_to_char_ids(word)

    def encode_chars(self, sentence, if_context, reverse=False, split=True):
        '''
        Encode the sentence as a white space delimited string of tokens.
        '''
        bos_chars = self.boc_chars if if_context else self.bod_chars
        if split:
            chars_ids = [self.word_to_char_ids(cur_word)
                         for cur_word in sentence.split()]
        else:
            chars_ids = [self.word_to_char_ids(cur_word)
                         for cur_word in sentence]
        if reverse:
            return np.vstack([self.eos_chars] + chars_ids + [bos_chars])
        else:
            return np.vstack([bos_chars] + chars_ids + [self.eos_chars])


class Batcher(object):
    '''
    Batch sentences of tokenized text into character id matrices.
    '''
    def __init__(self, lm_vocab_file: str, max_token_length: int):
        '''
        lm_vocab_file = the language model vocabulary file (one line per
            token)
        max_token_length = the maximum number of characters in each token
        '''
        self._lm_vocab = UnicodeCharsVocabulary(
            lm_vocab_file, max_token_length
        )
        self._max_token_length = max_token_length

    def batch_sentences(self, sentences: List[List[str]]):
        '''
        Batch the sentences as character ids
        Each sentence is a list of tokens without <s> or </s>, e.g.
        [['The', 'first', 'sentence', '.'], ['Second', '.']]
        '''
        n_sentences = len(sentences)
        max_length = max(len(sentence) for sentence in sentences) + 2

        X_char_ids = np.zeros(
            (n_sentences, max_length, self._max_token_length),
            dtype=np.int64
        )

        for k, sent in enumerate(sentences):
            length = len(sent) + 2
            char_ids_without_mask = self._lm_vocab.encode_chars(
                sent, split=False)
            # add one so that 0 is the mask value
            X_char_ids[k, :length, :] = char_ids_without_mask + 1

        return X_char_ids


class TokenBatcher(object):
    '''
    Batch sentences of tokenized text into token id matrices.
    '''
    def __init__(self, lm_vocab_file: str):
        '''
        lm_vocab_file = the language model vocabulary file (one line per
            token)
        '''
        self._lm_vocab = Vocabulary(lm_vocab_file)

    def batch_sentences(self, sentences: List[List[str]]):
        '''
        Batch the sentences as character ids
        Each sentence is a list of tokens without <s> or </s>, e.g.
        [['The', 'first', 'sentence', '.'], ['Second', '.']]
        '''
        n_sentences = len(sentences)
        max_length = max(len(sentence) for sentence in sentences) + 2

        X_ids = np.zeros((n_sentences, max_length), dtype=np.int64)

        for k, sent in enumerate(sentences):
            length = len(sent) + 2
            ids_without_mask = self._lm_vocab.encode(sent, split=False)
            # add one so that 0 is the mask value
            X_ids[k, :length] = ids_without_mask + 1

        return X_ids


##### for training
def _get_batch(generator, batch_size, num_steps, bow_size, max_word_length):
    """Read batches of input."""
    no_more_data = False
    while True:
        cur_stream = [None] * batch_size

        fw_cont_token_ids = np.zeros([batch_size, num_steps], np.int32)
        bk_cont_token_ids = np.zeros([batch_size, num_steps], np.int32)
        fw_desc_token_ids = np.zeros([batch_size, num_steps], np.int32)
        bk_desc_token_ids = np.zeros([batch_size, num_steps], np.int32)

        fw_cont_token_mask = np.zeros([batch_size, num_steps], np.int32)
        bk_cont_token_mask = np.zeros([batch_size, num_steps], np.int32)
        fw_desc_token_mask = np.zeros([batch_size, num_steps], np.int32)
        bk_desc_token_mask = np.zeros([batch_size, num_steps], np.int32)

        entity_mask = np.zeros([batch_size, num_steps + 1], np.int32)
        if max_word_length is not None:
            fw_cont_char_ids = np.zeros(
                [batch_size, num_steps, max_word_length],
                np.int32)
            bk_cont_char_ids = np.zeros(
                [batch_size, num_steps, max_word_length],
                np.int32)
            fw_desc_char_ids = np.zeros(
                [batch_size, num_steps, max_word_length],
                np.int32)
            bk_desc_char_ids = np.zeros(
                [batch_size, num_steps, max_word_length],
                np.int32)
        else:
            fw_cont_char_ids = bk_cont_char_ids = \
                fw_desc_char_ids = bk_desc_char_ids = None

        fw_cont_target_ids = np.zeros([batch_size, num_steps], np.int32)
        bk_cont_target_ids = np.zeros([batch_size, num_steps], np.int32)
        fw_desc_target_ids = np.zeros([batch_size, num_steps], np.int32)
        bk_desc_target_ids = np.zeros([batch_size, num_steps], np.int32)

        fw_cont_target_mask = np.zeros([batch_size, num_steps], np.int32)
        bk_cont_target_mask = np.zeros([batch_size, num_steps], np.int32)
        fw_desc_target_mask = np.zeros([batch_size, num_steps], np.int32)
        bk_desc_target_mask = np.zeros([batch_size, num_steps], np.int32)

        cont_bow_ids = np.zeros([batch_size, bow_size], np.int32)
        desc_bow_ids = np.zeros([batch_size, bow_size], np.int32)

        cont_bow_mask = np.zeros([batch_size, bow_size], np.int32)
        desc_bow_mask = np.zeros([batch_size, bow_size], np.int32)
        for i in range(batch_size):
            fw_cont_cur_pos = bk_cont_cur_pos = \
                fw_desc_cur_pos = bk_desc_cur_pos = 0

            if cur_stream[i] is None:
                try:
                    cur_stream[i] = list(next(generator))
                except StopIteration:
                    # No more data, exhaust current streams and quit
                    no_more_data = True
                    break
            fw_desc_words, fw_desc_chars, bk_desc_words, \
                bk_desc_chars, fw_cont_words, fw_cont_chars, \
                bk_cont_words, bk_cont_chars, desc_bow, cont_bow, \
                start, end = cur_stream[i]

            entity_mask[i, start + 1:] = 1.
            entity_mask[i, end + 1:] = 0.

            desc_bow_ids[i, :len(desc_bow)] = desc_bow
            cont_bow_ids[i, :len(cont_bow)] = cont_bow

            desc_bow_mask[i, :len(desc_bow)] = 1
            cont_bow_mask[i, :len(cont_bow)] = 1

            # how_many = min(
            #     len(fw_desc_words) - 1, num_steps - fw_desc_cur_pos)
            # fw_desc_next_pos = fw_desc_cur_pos + how_many
            # fw_desc_token_mask[i, fw_desc_cur_pos:fw_desc_next_pos] = 1

            # how_many = min(
            #     len(bk_desc_words) - 1, num_steps - bk_desc_cur_pos)
            # bk_desc_next_pos = bk_desc_cur_pos + how_many
            # bk_desc_token_mask[i, bk_desc_cur_pos:bk_desc_next_pos] = 1

            # how_many = min(
            #     len(fw_cont_words) - 1, num_steps - fw_cont_cur_pos)
            # fw_cont_next_pos = fw_cont_cur_pos + how_many
            # fw_cont_token_mask[i, fw_cont_cur_pos:fw_cont_next_pos] = 1

            # how_many = min(
            #     len(bk_cont_words) - 1, num_steps - bk_cont_cur_pos)
            # bk_cont_next_pos = bk_cont_cur_pos + how_many
            # bk_cont_token_mask[i, bk_cont_cur_pos:bk_cont_next_pos] = 1
            # while fw_desc_cur_pos < num_steps:
            # forward description
            how_many = min(
                len(fw_desc_words) - 1, num_steps - fw_desc_cur_pos)
            # if how_many == 0:
            #     break
            fw_desc_next_pos = fw_desc_cur_pos + how_many

            fw_desc_token_ids[i, fw_desc_cur_pos:fw_desc_next_pos] = \
                fw_desc_words[:how_many]
            fw_desc_token_mask[i, fw_desc_cur_pos:fw_desc_next_pos] = 1

            if max_word_length is not None:
                fw_desc_char_ids[i, fw_desc_cur_pos:fw_desc_next_pos] = \
                    fw_desc_chars[:how_many]

            fw_desc_target_ids[i, fw_desc_cur_pos:fw_desc_next_pos] = \
                fw_desc_words[1:how_many + 1]
            fw_desc_target_mask[i, fw_desc_cur_pos:fw_desc_next_pos] = 1

            # fw_desc_cur_pos = fw_desc_next_pos

            # fw_desc_words = fw_desc_words[how_many:]
            # if max_word_length is not None:
            #     fw_desc_chars = fw_desc_chars[how_many:]

            # while bk_desc_cur_pos < num_steps:
            # backward description
            how_many = min(
                len(bk_desc_words) - 1, num_steps - bk_desc_cur_pos)
            # if how_many == 0:
            #     break
            bk_desc_next_pos = bk_desc_cur_pos + how_many

            bk_desc_token_ids[i, bk_desc_cur_pos:bk_desc_next_pos] = \
                bk_desc_words[:how_many]
            bk_desc_token_mask[i, bk_desc_cur_pos:bk_desc_next_pos] = 1

            if max_word_length is not None:
                bk_desc_char_ids[i, bk_desc_cur_pos:bk_desc_next_pos] = \
                    bk_desc_chars[:how_many]
            bk_desc_target_ids[i, bk_desc_cur_pos:bk_desc_next_pos] = \
                bk_desc_words[1:how_many + 1]
            bk_desc_target_mask[i, bk_desc_cur_pos:bk_desc_next_pos] = 1

            # bk_desc_cur_pos = bk_desc_next_pos

            # bk_desc_words = bk_desc_words[how_many:]
            # if max_word_length is not None:
            #     bk_desc_chars = bk_desc_chars[how_many:]

            # while fw_cont_cur_pos < num_steps:
                # forward context
            how_many = min(
                len(fw_cont_words) - 1, num_steps - fw_cont_cur_pos)
            # if how_many == 0:
                # break
            fw_cont_next_pos = fw_cont_cur_pos + how_many

            fw_cont_token_ids[i, fw_cont_cur_pos:fw_cont_next_pos] = \
                fw_cont_words[:how_many]
            fw_cont_token_mask[i, fw_cont_cur_pos:fw_cont_next_pos] = 1
            if max_word_length is not None:
                fw_cont_char_ids[i, fw_cont_cur_pos:fw_cont_next_pos] = \
                    fw_cont_chars[:how_many]
            fw_cont_target_ids[i, fw_cont_cur_pos:fw_cont_next_pos] = \
                fw_cont_words[1:how_many + 1]
            fw_cont_target_mask[i, fw_cont_cur_pos:fw_cont_next_pos] = 1

            # fw_cont_cur_pos = fw_cont_next_pos

            # fw_cont_words = fw_cont_words[how_many:]
            # if max_word_length is not None:
            #     fw_cont_chars = fw_cont_chars[how_many:]

            # while bk_cont_cur_pos < num_steps:
            # backward context
            how_many = min(
                len(bk_cont_words) - 1, num_steps - bk_cont_cur_pos)
            # if how_many == 0:
                # break
            bk_cont_next_pos = bk_cont_cur_pos + how_many

            bk_cont_token_ids[i, bk_cont_cur_pos:bk_cont_next_pos] = \
                bk_cont_words[:how_many]
            bk_cont_token_mask[i, bk_cont_cur_pos:bk_cont_next_pos] = 1
            if max_word_length is not None:
                bk_cont_char_ids[i, bk_cont_cur_pos:bk_cont_next_pos] = \
                    bk_cont_chars[:how_many]
            bk_cont_target_ids[i, bk_cont_cur_pos:bk_cont_next_pos] = \
                bk_cont_words[1:how_many + 1]
            bk_cont_target_mask[i, bk_cont_cur_pos:bk_cont_next_pos] = 1

            # bk_cont_cur_pos = bk_cont_next_pos

            # bk_cont_words = bk_cont_words[how_many:]
            # if max_word_length is not None:
            #     bk_cont_chars = bk_cont_chars[how_many:]

        if no_more_data:
            # There is no more data.  Note: this will not return data
            # for the incomplete batch
            break

        X = {'fw_cont_token_ids': fw_cont_token_ids,
             'fw_cont_token_mask': fw_cont_token_mask,
             'fw_cont_char_ids': fw_cont_char_ids,
             'fw_cont_target_ids': fw_cont_target_ids,
             'fw_cont_target_mask': fw_cont_target_mask,
             'bk_cont_token_ids': bk_cont_token_ids,
             'bk_cont_token_mask': bk_cont_token_mask,
             'bk_cont_char_ids': bk_cont_char_ids,
             'bk_cont_target_ids': bk_cont_target_ids,
             'bk_cont_target_mask': bk_cont_target_mask,
             'cont_bow_ids': cont_bow_ids,
             'cont_bow_mask': cont_bow_mask,
             'fw_desc_token_ids': fw_desc_token_ids,
             'fw_desc_token_mask': fw_desc_token_mask,
             'fw_desc_char_ids': fw_desc_char_ids,
             'fw_desc_target_ids': fw_desc_target_ids,
             'fw_desc_target_mask': fw_desc_target_mask,
             'bk_desc_token_ids': bk_desc_token_ids,
             'bk_desc_token_mask': bk_desc_token_mask,
             'bk_desc_char_ids': bk_desc_char_ids,
             'bk_desc_target_ids': bk_desc_target_ids,
             'bk_desc_target_mask': bk_desc_target_mask,
             'desc_bow_ids': desc_bow_ids,
             'desc_bow_mask': desc_bow_mask,
             'entity_mask': entity_mask}

        yield X


class LMDataset(object):
    """
    Hold a language model dataset.

    A dataset is a list of tokenized files.  Each file contains one sentence
        per line.  Each sentence is pre-tokenized and white space joined.
    """
    def __init__(self, filepattern, vocab, reverse=False, test=False,
                 shuffle_on_load=False):
        '''
        filepattern = a glob string that specifies the list of files.
        vocab = an instance of Vocabulary or UnicodeCharsVocabulary
        reverse = if True, then iterate over tokens in each sentence in reverse
        test = if True, then iterate through all data once then stop.
            Otherwise, iterate forever.
        shuffle_on_load = if True, then shuffle the sentences after loading.
        '''
        self._vocab = vocab
        self._all_shards = glob.glob(filepattern)
        print('Found %d shards at %s' % (len(self._all_shards), filepattern))
        self._shards_to_choose = []

        self._reverse = reverse
        self._test = test
        self._shuffle_on_load = shuffle_on_load
        self._use_char_inputs = hasattr(vocab, 'encode_chars')

        self._ids = self._load_random_shard()

    def _choose_random_shard(self):
        if len(self._shards_to_choose) == 0:
            self._shards_to_choose = list(self._all_shards)
            random.shuffle(self._shards_to_choose)
        shard_name = self._shards_to_choose.pop()
        return shard_name

    def _load_random_shard(self):
        """Randomly select a file and read it."""
        if self._test:
            if len(self._all_shards) == 0:
                # we've loaded all the data
                # this will propogate up to the generator in get_batch
                # and stop iterating
                raise StopIteration
            else:
                shard_name = self._all_shards.pop()
        else:
            # just pick a random shard
            shard_name = self._choose_random_shard()

        ids = self._load_shard(shard_name)
        self._i = 0
        self._nids = len(ids)
        return ids

    def _load_shard(self, shard_name):
        """Read one file and convert to ids.

        Args:
            shard_name: file path.

        Returns:
            list of (id, char_id) tuples.
        """
        print('Loading data from: %s' % shard_name)
        with open(shard_name) as f:
            sentences_raw = f.readlines()

        if self._reverse:
            sentences = []
            for sentence in sentences_raw:
                splitted = sentence.split()
                splitted.reverse()
                sentences.append(' '.join(splitted))
        else:
            sentences = sentences_raw

        if self._shuffle_on_load:
            random.shuffle(sentences)

        ids = [self.vocab.encode(sentence, self._reverse)
               for sentence in sentences]
        if self._use_char_inputs:
            chars_ids = [self.vocab.encode_chars(sentence, self._reverse)
                         for sentence in sentences]
        else:
            chars_ids = [None] * len(ids)

        print('Loaded %d sentences.' % len(ids))
        print('Finished loading')
        return list(zip(ids, chars_ids))

    def get_sentence(self):
        while True:
            if self._i == self._nids:
                self._ids = self._load_random_shard()
            ret = self._ids[self._i]
            self._i += 1
            yield ret

    @property
    def max_word_length(self):
        if self._use_char_inputs:
            return self._vocab.max_word_length
        else:
            return None

    def iter_batches(self, batch_size, num_steps):
        for X in _get_batch(self.get_sentence(), batch_size, bow_size,
                            num_steps, self.max_word_length):

            # token_ids = (batch_size, num_steps)
            # char_inputs = (batch_size, num_steps, 50) of character ids
            # targets = word ID of next word (batch_size, num_steps)
            yield X

    @property
    def vocab(self):
        return self._vocab


class BidirectionalLMDataset(object):
    def __init__(self, filepattern, vocab, test=False, shuffle_on_load=False):
        '''
        bidirectional version of LMDataset
        '''
        self._data_forward = LMDataset(
            filepattern, vocab, reverse=False, test=test,
            shuffle_on_load=shuffle_on_load)
        self._data_reverse = LMDataset(
            filepattern, vocab, reverse=True, test=test,
            shuffle_on_load=shuffle_on_load)

    def iter_batches(self, batch_size, num_steps):
        max_word_length = self._data_forward.max_word_length

        for X, Xr in zip(
            _get_batch(self._data_forward.get_sentence(), batch_size,
                       num_steps, max_word_length),
            _get_batch(self._data_reverse.get_sentence(), batch_size,
                       num_steps, max_word_length)
        ):

            for k, v in Xr.items():
                X[k + '_reverse'] = v

            yield X


class WikiLinkDataset:
    def __init__(self, vocab, filepattern, path2ent2def, num_steps, bow_size):
        self.num_steps = num_steps
        self.bow_size = bow_size
        self._vocab = vocab
        self._all_shards = glob.glob(filepattern)
        print('Found %d shards at %s' % (len(self._all_shards), filepattern))
        self._shards_to_choose = list(self._all_shards).copy()
        random.shuffle(self._shards_to_choose)
        self._ids = self._load_random_shard()

    def _choose_random_shard(self):
        if len(self._shards_to_choose) == 0:
            raise StopIteration()
            # self._shards_to_choose = list(self._all_shards)
            # random.shuffle(self._shards_to_choose)
        shard_name = self._shards_to_choose.pop()
        return shard_name

    def _load_random_shard(self):
        """Randomly select a file and read it."""
        shard_name = self._choose_random_shard()

        ids = self._load_shard(shard_name)
        self._i = 0
        self._nids = len(ids)
        return ids

    def _add_ent_to_sent(self, ent, sent):
        new_curr_sent = sent.split(" ")
        ent_idx = new_curr_sent.index("<placeholder>")
        pw_curr_sent = new_curr_sent[: ent_idx]
        nw_curr_sent = new_curr_sent[ent_idx + 1:]
        new_curr_sent = pw_curr_sent + ent.split(" ") + nw_curr_sent
        return new_curr_sent, len(pw_curr_sent), \
            len(pw_curr_sent) + len(ent.split(" "))

    def _load_shard(self, shard_name):
        """Read one file and convert to ids.

        Args:
            shard_name: file path.

        Returns:
            list of (id, char_id) tuples.
        """
        print('Loading data from: %s' % shard_name)
        with open(shard_name) as f:
            fw_curr_words, fw_curr_chars = [], []
            bk_curr_words, bk_curr_chars = [], []
            fw_desc_words, fw_desc_chars = [], []
            bk_desc_words, bk_desc_chars = [], []

            desc_bows, curr_bows = [], []
            starts = []
            ends = []
            for line in f:
                try:
                    tgt_doc, ent, _, curr_sent, _, desc_sent = \
                        line.strip().lower().split("\t")
                except ValueError:
                    continue
                if len(ent) == 0:
                    continue
                new_curr_sent, start, end = \
                    self._add_ent_to_sent(ent, curr_sent)
                if len(new_curr_sent) + 1 >= self.num_steps:
                    continue
                fw_desc_words.append(
                    self._vocab.encode(
                        desc_sent, if_context=False, reverse=False))
                fw_desc_chars.append(
                    self._vocab.encode_chars(
                        desc_sent, if_context=False, reverse=False))

                rev_desc_sent = desc_sent.split().copy()
                rev_desc_sent.reverse()
                bk_desc_words.append(
                    self._vocab.encode(" ".join(rev_desc_sent),
                                       if_context=False, reverse=True))
                bk_desc_chars.append(
                    self._vocab.encode_chars(
                        " ".join(rev_desc_sent),
                        if_context=False, reverse=True))

                desc_bows.append(
                    self._vocab.encode_bow(desc_sent, self.bow_size))

                fw_curr_words.append(
                    self._vocab.encode(" ".join(new_curr_sent),
                                       if_context=True, reverse=False))
                fw_curr_chars.append(
                    self._vocab.encode_chars(" ".join(new_curr_sent),
                                             if_context=True, reverse=False))

                rev_curr_sent = new_curr_sent.copy()
                rev_curr_sent.reverse()
                bk_curr_words.append(
                    self._vocab.encode(" ".join(rev_curr_sent),
                                       if_context=True, reverse=True))
                bk_curr_chars.append(
                    self._vocab.encode_chars(
                        " ".join(rev_curr_sent),
                        if_context=True, reverse=True))

                curr_bows.append(
                    self._vocab.encode_bow(" ".join(curr_sent), self.bow_size))

                starts.append(start)
                ends.append(end)

        idx = np.random.permutation(len(fw_desc_words))
        fw_desc_words = np.array(fw_desc_words)[idx]
        fw_desc_chars = np.array(fw_desc_chars)[idx]
        bk_desc_words = np.array(bk_desc_words)[idx]
        bk_desc_chars = np.array(bk_desc_chars)[idx]

        fw_curr_words = np.array(fw_curr_words)[idx]
        fw_curr_chars = np.array(fw_curr_chars)[idx]
        bk_curr_words = np.array(bk_curr_words)[idx]
        bk_curr_chars = np.array(bk_curr_chars)[idx]

        desc_bows = np.array(desc_bows)[idx]
        curr_bows = np.array(curr_bows)[idx]

        starts = np.array(starts)[idx]
        ends = np.array(ends)[idx]

        print('Loaded %d sentences.' % len(starts))
        print('Finished loading (#finish: {}, #todo: {})'.format(
            len(self._all_shards) - len(self._shards_to_choose),
            len(self._shards_to_choose)))
        return list(zip(fw_desc_words, fw_desc_chars, bk_desc_words,
                        bk_desc_chars, fw_curr_words, fw_curr_chars,
                        bk_curr_words, bk_curr_chars, desc_bows, curr_bows,
                        starts, ends))

    def get_sentence(self):
        while True:
            # try:
            if self._i == self._nids:
                self._ids = self._load_random_shard()
                while len(self._ids) == 0:
                    self._ids = self._load_random_shard()
            ret = self._ids[self._i]
            self._i += 1
            yield ret

    @property
    def max_word_length(self):
        return self._vocab.max_word_length

    def iter_batches(self, batch_size):
        for X in _get_batch(self.get_sentence(), batch_size, self.num_steps,
                            self.bow_size, self.max_word_length):

            # token_ids = (batch_size, num_steps)
            # char_inputs = (batch_size, num_steps, 50) of character ids
            # targets = word ID of next word (batch_size, num_steps)
            yield X


class InvalidNumberOfCharacters(Exception):
    pass
