def load_parameters():
    """
        Loads the defined parameters
    """
    # Input data params
    DATA_ROOT_PATH = '/home/fantonio/datasets/ted/enfr/lowercased/joint_bpe' #'/home/fantonio/software/nmt-keras/examples/EuTrans' \
    SRC_LAN = 'en'  # Language of the source text
    TRG_LAN = 'fr'  # Language of the target text

    TASK_NAME  = 'ted'
    # preprocessed features
    DATASET_NAME = TASK_NAME+'_'+SRC_LAN+TRG_LAN+'-linked'  # Dataset name (add '-linked' suffix for using
                                                            # dataset with temporally-linked training data)
                                                            #
                                                            #    -linked
                                                            #    -linked-upperbound
                                                            #    -linked-upperbound-copy
                                                            #    -linked-upperbound-prev
                                                            #    -linked-upperbound-nocopy
                                                            #    -linked-video
                                                            #    -linked-vidtext
                                                            #    -vidtext-embed

    PRE_TRAINED_DATASET_NAME = None  #'MSVD_features'     # Dataset name for reusing vocabulary of pre-trained model (set to None for disabling)
                                                          # (only applicable if we are using a pre-trained model, default None)
    VOCABULARIES_MAPPING = {'description': 'description',
                            'state_below': 'description',
                            'prev_sentence': 'description'}

    # SRC_LAN or TRG_LAN will be added to the file names
    TEXT_FILES = {'train': 'training.',  # Data files
                  'val': 'dev.',
                  'test': 'test.'}

    PRE_TRAINED_VOCABULARY_NAME = None  #'1BillionWords_vocabulary'      # Dataset name for reusing vocabulary of pre-trained model

    # Input data
    #INPUT_DATA_TYPE = 'video-features'                          # 'video-features' or 'video'
    INPUT_DATA_TYPE = 'text'                                     # 'video-features' or 'video'
    STEP_LINKS = True  # Whether precalculate link indexes or go acroos the whole dataset
    PREV_SENT_LAN = TRG_LAN                                      # Whether if we use the previous source sentence
                                                                 #
    UPPERBOUND = False or '-upperbound' in DATASET_NAME
                                                                 # or the previous translated sentence(target)
    NUM_FRAMES = 26                                              # fixed number of input frames per video

    # Output data
    DESCRIPTION_FILES = {'train': 'Annotations/train_descriptions.txt',                 # Description files
                         'val': 'Annotations/val_descriptions.txt',
                         'test': 'Annotations/test_descriptions.txt',}
    DESCRIPTION_COUNTS_FILES = { 'train': 'Annotations/train_descriptions_counts.npy',  # Description counts files
                                 'val': 'Annotations/val_descriptions_counts.npy',
                                 'test': 'Annotations/test_descriptions_counts.npy',}

    # Dataset parameters
    INPUTS_IDS_DATASET = ['source_text', 'state_below']  # Corresponding inputs of the dataset
    OUTPUTS_IDS_DATASET = ['description']  # Corresponding outputs of the dataset
    INPUTS_IDS_MODEL = ['source_text', 'state_below']  # Corresponding inputs of the built model
    OUTPUTS_IDS_MODEL = ['description']  # Corresponding outputs of the built model


    if '-linked' in DATASET_NAME:

        LINK_SAMPLE_FILES = {'train': 'Annotations/train_link_samples.txt',
                             'val': 'Annotations/val_link_samples.txt',
                             'test': 'Annotations/test_link_samples.txt',
                            } # Links index files

        INPUTS_IDS_DATASET.append('prev_sentence')
        INPUTS_IDS_MODEL.append('prev_sentence')

        # Previous sentence indexes
        if not UPPERBOUND:
            INPUTS_IDS_DATASET.append('link_index')
            INPUTS_IDS_MODEL.append('link_index')


    # Evaluation params
    METRICS = ['coco']  # Metric used for evaluating model after each epoch (leave empty if only prediction is required)

    EVAL_ON_SETS = ['val', 'test'] if UPPERBOUND else ['val'] # Possible values: 'train', 'val' and 'test' (external evaluator)
    EVAL_ON_SETS_KERAS = []                       # Possible values: 'train', 'val' and 'test' (Keras' evaluator)
    START_EVAL_ON_EPOCH = 0                       # First epoch where the model will be evaluated
    EVAL_EACH_EPOCHS = False                      # Select whether evaluate between N epochs or N updates
    EVAL_EACH = 1650                               # Sets the evaluation frequency (epochs or updates)

    ALIGN_FROM_RAW = True  # Align using the full vocabulary or the short_list

    # Search parameters
    SAMPLING = 'max_likelihood'                   # Possible values: multinomial or max_likelihood (recommended)
    TEMPERATURE = 1                               # Multinomial sampling parameter
    if not '-vidtext-embed' in DATASET_NAME:
        BEAM_SEARCH = True                        # Switches on-off the beam search procedure
    else:
        BEAM_SEARCH = False
    BEAM_SIZE = 12                                 # Beam size (in case of BEAM_SEARCH == True)
    BEAM_SEARCH_COND_INPUT = 1                     # Index of the conditional input used in beam search (i.e., state_below)
    OPTIMIZED_SEARCH = True                       # Compute annotations only a single time per sample
    NORMALIZE_SAMPLING = False                    # Normalize hypotheses scores according to their length
    ALPHA_FACTOR = .6                             # Normalization according to length**ALPHA_FACTOR
                                                  # (see: arxiv.org/abs/1609.08144)

    # Sampling params: Show some samples during training
    SAMPLE_ON_SETS = ['val', 'test']             # Possible values: 'train', 'val' and 'test'

    N_SAMPLES = 5                                 # Number of samples generated
    START_SAMPLING_ON_EPOCH = 1                   # First epoch where the model will be evaluated
    SAMPLE_EACH_UPDATES = 2200                     # Sampling frequency (default 450)

    # Word representation params
    TOKENIZATION_METHOD = 'tokenize_none'        # Select which tokenization we'll apply:
                                                  #  tokenize_basic, tokenize_aggressive, tokenize_soft,
                                                  #  tokenize_icann or tokenize_questions
    APPLY_DETOKENIZATION = True
    DETOKENIZATION_METHOD = 'detokenize_bpe'

    FILL = 'end'                                  # whether we fill the 'end' or the 'start' of the sentence with 0s
    PAD_ON_BATCH = True                           # Whether we take as many timesteps as the longes sequence of the batch
                                                  # or a fixed size (MAX_OUTPUT_TEXT_LEN)

    # Input image parameters
    DATA_AUGMENTATION = False                      # Apply data augmentation on input data (noise on features)
    DATA_AUGMENTATION_TYPE = ['random_selection']  # 'random_selection', 'noise'
    IMG_FEAT_SIZE = 0 #1024                           # Size of the image features

    # Input text parameters
    INPUT_VOCABULARY_SIZE = 0                     # Size of the input vocabulary. Set to 0 for using all,
    MAX_INPUT_TEXT_LEN = 80
    MAX_INPUT_TEXT_LEN_TEST = MAX_INPUT_TEXT_LEN * 2
    MIN_OCCURRENCES_INPUT_VOCAB = 0

    # Output text parameters
    OUTPUT_VOCABULARY_SIZE = 0                    # Size of the input vocabulary. Set to 0 for using all,
                                                  # otherwise it will be truncated to these most frequent words.
    MAX_OUTPUT_TEXT_LEN = 80                      # Maximum length of the output sequence
                                                  # set to 0 if we want to use the whole answer as a single class
    MAX_OUTPUT_TEXT_LEN_TEST = 80                 # Maximum length of the output sequence during test time
    MIN_OCCURRENCES_VOCAB = 0                     # Minimum number of occurrences allowed for the words in the vocabulay.

    # Optimizer parameters (see model.compile() function)
    LOSS = 'categorical_crossentropy' #sparse_
    CLASSIFIER_ACTIVATION = 'softmax'

    OPTIMIZER = 'Adam'                            # Optimizer
    LR = 0.0002                                   # Learning rate. Recommended values - Adam 0.001 - Adadelta 1.0
    CLIP_C = 10.                                  # During training, clip gradients to this norm
    SAMPLE_WEIGHTS = True                     # Select whether we use a weights matrix (mask) for the data outputs
    LR_DECAY = None                               # Minimum number of epochs before the next LR decay. Set to None if don't want to decay the learning rate
    LR_GAMMA = 0.995                              # Multiplier used for decreasing the LR

    # Training parameters
    MAX_EPOCH = 200                               # Stop when computed this number of epochs
    BATCH_SIZE = 20                              # ABiViRNet trained with BATCH_SIZE = 64

    HOMOGENEOUS_BATCHES = False                         # Use batches with homogeneous output lengths for every minibatch (Possibly buggy!)
    PARALLEL_LOADERS = 1                                # Parallel data batch loaders
    EPOCHS_FOR_SAVE = 1 if EVAL_EACH_EPOCHS else None   # Number of epochs between model saves (None for disabling epoch save)
    WRITE_VALID_SAMPLES = True                          # Write valid samples in file
    SAVE_EACH_EVALUATION = True if not EVAL_EACH_EPOCHS else False   # Save each time we evaluate the model

    # Early stop parameters
    EARLY_STOP = True                             # Turns on/off the early stop protocol
    PATIENCE = 15                                 # We'll stop if the val STOP_METRIC does not improve after this
                                                  # number of evaluations

    if not '-vidtext-embed' in DATASET_NAME:
        STOP_METRIC = 'Bleu_4'                        # Metric for the stop
    else:
        STOP_METRIC = 'accuracy'

    # Model parameters
    MODEL_TYPE = 'LinkedTranslation3'                               # 'ArcticVideoCaptionWithInit'
                                                                   # 'ArcticVideoCaptionNoLSTMEncWithInit'
                                                                   # 'TemporallyLinkedVideoDescriptionNoAtt'
                                                                   # 'TemporallyLinkedVideoDescriptionAtt'
                                                                   # 'TemporallyLinkedVideoDescriptionAttDoublePrev'
                                                                   # 'VideoTextEmbedding'
                                                                   # 'DeepSeek'
                                                                   # '--------------------NMT------------------------'
                                                                   # 'LinkedTranslation'
                                                                   # 'DoubleLinkedTranslation'


    ENCODER_RNN_TYPE = 'LSTM'  # RNN unit type ('LSTM' supported)
    DECODER_RNN_TYPE = 'ConditionalLSTM'  # RNN unit type ('LSTM' supported)
    # Added parameters with respect original config
    SOURCE_TEXT_EMBEDDING_SIZE = 420
    SRC_PRETRAINED_VECTORS = None  # Path to pretrained vectors. (e.g. DATA_ROOT_PATH + '/DATA/word2vec.%s.npy' % TRG_LAN)
    INIT_FUNCTION = 'glorot_uniform'  # General initialization function for matrices.
    INNER_INIT = 'orthogonal'  # Initialization function for inner RNN matrices.
    INIT_ATT = 'glorot_uniform'  # Initialization function for attention mechism matrices

    # Input text parameters
    TARGET_TEXT_EMBEDDING_SIZE = 420              # Source language word embedding size (ABiViRNet 301)
    TRG_PRETRAINED_VECTORS = None                 # Path to pretrained vectors. (e.g. DATA_ROOT_PATH + '/DATA/word2vec.%s.npy' % TRG_LAN)
                                                  # Set to None if you don't want to use pretrained vectors.
                                                  # When using pretrained word embeddings, the size of the pretrained word embeddings must match with the word embeddings size.
    TRG_PRETRAINED_VECTORS_TRAINABLE = True       # Finetune or not the target word embedding vectors.

    # Encoder configuration
    ENCODER_HIDDEN_SIZE = 420                     # For models with RNN encoder (ABiViRNet 717)
    BIDIRECTIONAL_ENCODER = True                  # Use bidirectional encoder
    N_LAYERS_ENCODER = 1                          # Stack this number of encoding layers (default 1)
    BIDIRECTIONAL_DEEP_ENCODER = True             # Use bidirectional encoder in all encoding layers


    # Previous sentence encoder
    PREV_SENT_ENCODER_HIDDEN_SIZE = 300           # For models with previous sentence RNN encoder (484)
    BIDIRECTIONAL_PREV_SENT_ENCODER = True        # Use bidirectional encoder
    N_LAYERS_PREV_SENT_ENCODER = 1                # Stack this number of encoding layers
    BIDIRECTIONAL_DEEP_PREV_SENT_ENCODER = True   # Use bidirectional encoder in all encoding layers

    DECODER_HIDDEN_SIZE = 420                     # For models with LSTM decoder (ABiViRNet 484)
    SKIP_VECTORS_HIDDEN_SIZE = TARGET_TEXT_EMBEDDING_SIZE
    ADDITIONAL_OUTPUT_MERGE_MODE = 'Add'          # Merge mode for the skip connections
    WEIGHTED_MERGE = False                        # Wether we want to apply a conventional or a weighted merge


    #AFFINE_LAYERS_DIM = 250     # Dimensionality of the affine layers in 'DeepSeek' model

    IMG_EMBEDDING_LAYERS = []  # FC layers for visual embedding
                               # Here we should specify the activation function and the output dimension
                               # (e.g IMG_EMBEDDING_LAYERS = [('linear', 1024)]

    # Fully-Connected layers for initializing the first RNN state
    #       Here we should only specify the activation function of each layer
    #       (as they have a potentially fixed size)
    #       (e.g INIT_LAYERS = ['tanh', 'relu'])
    INIT_LAYERS = ['tanh']

    # Additional Fully-Connected layers's sizes applied before softmax.
    #       Here we should specify the activation function and the output dimension
    #       (e.g DEEP_OUTPUT_LAYERS = [('tanh', 600), ('relu', 400), ('relu', 200)])
    DEEP_OUTPUT_LAYERS = [('linear', TARGET_TEXT_EMBEDDING_SIZE)]

    # Regularizers
    WEIGHT_DECAY = 1e-4                           # L2 regularization
    RECURRENT_WEIGHT_DECAY = 0.                   # L2 regularization in recurrent layers

    USE_DROPOUT = False                           # Use dropout
    DROPOUT_P = 0.                                # Percentage of units to drop

    USE_RECURRENT_DROPOUT = False                 # Use dropout in recurrent layers # DANGEROUS!
    RECURRENT_DROPOUT_P = 0.                      # Percentage of units to drop in recurrent layers
    RECURRENT_INPUT_DROPOUT_P = 0.                # Percentage of units to drop in recurrent layers

    USE_NOISE = True                              # Use gaussian noise during training
    NOISE_AMOUNT = 0.01                           # Amount of noise

    USE_BATCH_NORMALIZATION = True                # If True it is recommended to deactivate Dropout
    BATCH_NORMALIZATION_MODE = 1                  # See documentation in Keras' BN

    USE_PRELU = False                             # use PReLU activations as regularizer
    USE_L2 = False                                # L2 normalization on the features

    # Results plot and models storing parameters
    EXTRA_NAME = ''                    # This will be appended to the end of the model name
    MODEL_NAME = DATASET_NAME + '_' + MODEL_TYPE +\
                 '_txtemb_' + str(TARGET_TEXT_EMBEDDING_SIZE) + \
                 '_imgemb_' + '_'.join([layer[0] for layer in IMG_EMBEDDING_LAYERS]) + \
                 '_lstmenc_' + str(ENCODER_HIDDEN_SIZE) + \
                 '_lstm_' + str(DECODER_HIDDEN_SIZE) + \
                 '_additional_output_mode_' + str(ADDITIONAL_OUTPUT_MERGE_MODE) + \
                 '_deepout_' + '_'.join([layer[0] for layer in DEEP_OUTPUT_LAYERS]) + \
                 '_' + OPTIMIZER + '_decay_' + str(LR_DECAY) + '-' + str(LR_GAMMA)

    MODEL_NAME += '_' + EXTRA_NAME + '_TMA'

    # Name and location of the pre-trained model (only if RELOAD > 0)
    PRE_TRAINED_MODELS = [MODEL_NAME]
    PRE_TRAINED_MODEL_STORE_PATHS = map(lambda x: 'trained_models/' + x  + '/', PRE_TRAINED_MODELS) if isinstance(PRE_TRAINED_MODELS, list) else 'trained_models/'+PRE_TRAINED_MODELS+'/'
    LOAD_WEIGHTS_ONLY = True                           # Load weights of pre-trained model or complete Model_Wrapper instance

    STORE_PATH = 'trained_models/' + MODEL_NAME  + '/' # Models and evaluation results will be stored here

    DATASET_STORE_PATH = 'datasets/'                   # Dataset instance will be stored here

    SAMPLING_SAVE_MODE = 'list'                        # 'list' or 'vqa'
    VERBOSE = 1                                        # Verbosity level
    RELOAD = 0                                         # If 0 start training from scratch, otherwise the model
                                                      # Saved on epoch 'RELOAD' will be used
    REBUILD_DATASET = True                             # Build again or use stored instance
    MODE = 'training'                                  # 'training' or 'sampling' (if 'sampling' then RELOAD must
                                                       # be greater than 0 and EVAL_ON_SETS will be used)
    RELOAD_PATH = 'trained_models/' + MODEL_NAME  + '/'
    #RELOAD_PATH = 'trained_models/xer_test/'
    SAMPLING_RELOAD_EPOCH = False
    SAMPLING_RELOAD_POINT = 0
    # Extra parameters for special trainings
    TRAIN_ON_TRAINVAL = False  # train the model on both training and validation sets combined
    FORCE_RELOAD_VOCABULARY = False                    # force building a new vocabulary from the training samples applicable if RELOAD > 1

    # ============================================
    parameters = locals().copy()
    return parameters
