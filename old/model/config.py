import os


from .general_utils import get_logger
from .data_utils import get_trimmed_glove_vectors, load_vocab, \
        get_processing_word

os_type = "tinymind"
#os_type = "windows"

class Config():
    def __init__(self, load=True):
        """Initialize hyperparameters and load vocabs

        Args:
            load_embeddings: (bool) if True, load embeddings into
                np array, else None

        """
        # directory for training outputs
        if not os.path.exists(self.dir_output):
            os.makedirs(self.dir_output)

        # create instance of logger
        self.logger = get_logger(self.path_log)

        # load if requested (default)
        if load:
            self.load()


    def load(self):
        """Loads vocabulary, processing functions and embeddings

        Supposes that build_data.py has been run successfully and that
        the corresponding files have been created (vocab and trimmed GloVe
        vectors)

        """
        # 1. vocabulary
        self.vocab_words = load_vocab(self.filename_words)
        self.vocab_tags  = load_vocab(self.filename_tags)
        self.vocab_chars = load_vocab(self.filename_chars)

        self.nwords     = len(self.vocab_words)
        self.nchars     = len(self.vocab_chars)
        self.ntags      = len(self.vocab_tags)

        # 2. get processing functions that map str -> id
        self.processing_word = get_processing_word(self.vocab_words,
                self.vocab_chars, lowercase=True, chars=self.use_chars)
        self.processing_tag  = get_processing_word(self.vocab_tags,
                lowercase=False, allow_unk=False)

        # 3. get pre-trained embeddings
        self.embeddings = (get_trimmed_glove_vectors(self.filename_trimmed)
                if self.use_pretrained else None)

    if "tinymind" == os_type:
        dir_output = "/output"
        dir_model = dir_output
        path_log = dir_output
    else:
        # general config
        dir_output = "results/test/"
        dir_model = dir_output + "model.weights/"
        path_log = dir_output + "log.txt"

    # embeddings
    dim_word = 300
    dim_char = 100

    # dataset
    if "tinymind" == os_type:
        # glove files
        filename_glove = "/data/HataFeng/data-ner-1998-2014/glove.6B/glove.6B.{}d.txt".format(dim_word)
        # trimmed embeddings (created from glove_filename with build_data.py)
        filename_trimmed = "/data/HataFeng/data-ner-1998-2014/glove.6B.{}d.trimmed.npz".format(dim_word)
        use_pretrained = True

        filename_dev = "/data/HataFeng/data-ner-1998-2014/dev.txt"
        filename_test = "/data/HataFeng/data-ner-1998-2014/test.txt"
        filename_train = "/data/HataFeng/data-ner-1998-2014/train.txt"

        # vocab (created from dataset with build_data.py)
        filename_words = "/data/HataFeng/data-ner-1998-2014/words.txt"
        filename_tags = "/data/HataFeng/data-ner-1998-2014/tags.txt"
        filename_chars = "/data/HataFeng/data-ner-1998-2014/chars.txt"
    else:
        # glove files
        filename_glove = "data/glove.6B/glove.6B.{}d.txt".format(dim_word)
        # trimmed embeddings (created from glove_filename with build_data.py)
        filename_trimmed = "data/glove.6B.{}d.trimmed.npz".format(dim_word)
        use_pretrained = True

        filename_train = 'G:/test_data/NLP/词性标注/人民日报语料库2014/result/train.txt'
        filename_test = 'G:/test_data/NLP/词性标注/人民日报语料库2014/result/test.txt'
        filename_dev = 'G:/test_data/NLP/词性标注/人民日报语料库2014/result/dev.txt'

        # vocab (created from dataset with build_data.py)
        filename_words = "data/words.txt"
        filename_tags = "data/tags.txt"
        filename_chars = "data/chars.txt"


    max_iter = None # if not None, max number of examples in Dataset



    # training
    train_embeddings = False
    nepochs          = 5
    dropout          = 0.5
    batch_size       = 20
    lr_method        = "adam"
    lr               = 0.001
    lr_decay         = 0.9
    clip             = -1 # if negative, no clipping
    nepoch_no_imprv  = 3

    # model hyperparameters
    hidden_size_char = 100 # lstm on chars
    hidden_size_lstm = 300 # lstm on word embeddings

    # NOTE: if both chars and crf, only 1.6x slower on GPU
    use_crf = True # if crf, training is 1.7x slower on CPU
    use_chars = False # if char embedding, training is 3.5x slower on CPU
