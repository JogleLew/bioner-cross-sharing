class entries:
    class data: 
        # data_dir: Datasets directory
        data_dir = "data"

        # data_names: Dataset names, e.g. 
        # ["BC2GM-IOB", "Ex-PTM-IOB", "NCBI-disease-IOB", "linnaeus-IOB"]
        # ["JNLPBA", "BC5CDR-IOB", "BioNLP11ID-IOB", "BioNLP13PC-IOB", "CRAFT-IOB", "BC4CHEMD"]
        data_names = ["linnaeus-IOB", "BC5CDR-IOB"]

        # data_reduce: Data size reduce, value is keeping rate
        data_reduce = 1.0
    
    class embedding:
        # embedding: Embedding for words, one of ["glove", "senna", "sskip", "polyglot"]
        embedding_type = "glove"

        # embedding_dict: Path for embedding dict
        embedding_dict = "data/glove/glove.6B/glove.6B.100d.gz"

        # unk_replace: The rate to replace a singleton word with UNK
        unk_replace = 0.0
    
    class rnn: 
        # mode: Architecture of RNN, one of ["RNN", "LSTM", "GRU", "CNN"]
        mode = "LSTM"

        # hidden_size: Number of hidden units in RNN
        hidden_size = 256

        # hidden_size_private: Number of hidden units in RNN
        hidden_size_private = 256

        # char_dim: Dimension of character embeddings
        char_dim = 30

        # num_filters: Number of filters in CNN
        num_filters = 30

        # tag_space: Dimension of tag space
        tag_space = 128

        # dropout: Type of dropout, one of ["std", "variational"]
        dropout = "std"

        # p: Dropout rate
        p = 0.5

        # bigram: Enable bi-gram parameter for CRF
        bigram = True

        # attention: Attetion mode, one of ["none", "mlp", "fine"]
        attention = "none"
    
        # char_level_rnn: Enable Character Level RNN
        char_level_rnn = False
    
    class multitask:
        # adv_loss_coef: Coefficient of adversarial loss
        adv_loss_coef = 0.01

        # diff_loss_coef: Coefficient of diff loss
        diff_loss_coef = 0.001
    
    class training:
        # num_epoches: Number of training epochs
        num_epochs = 80

        # batch_size: Number of sentences in each batch
        batch_size = 16

        # learning_rate: Learning rate
        learning_rate = 0.001

        # momentum: Momentum
        momentum = 0

        # alpha: Alpha value of rmsprop
        alpha = 0.95 

        # lr_decay: Decay rate of learning rate
        lr_decay = 0.97

        # schedule: Schedule for learning rate decay
        schedule = 1

        # gamma: Weight for regularization
        gamma = 0.0
