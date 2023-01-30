import argparse

def createParser():
    parser = argparse.ArgumentParser(description='Input parameters to train the model.')

    # device
    parser.add_argument ('-n_gpu', '--n_gpu', type=int, default=0,
                        help='Number of GPU core.')

    # dataset
    parser.add_argument ('-afile', '--annotation_file', default="/workdir/annotations_file_short_SF.csv",
                        help='Path to the annotation_file.')
    parser.add_argument ('-ddir', '--dataset_dir', default="/workdir/data_v2",
                        help='Path to the dataset directory.')

    # train_dataloader
    parser.add_argument ('-n_b', '--n_batch', type=int, default=100, 
                        help='Number of batches in train dataloader')
    parser.add_argument ('-n_w', '--n_ways', type=int, default=10,
                        help='Number of classes inside one batch. \
                            Maximum number of n_ways is 100, \
                            since maximum number of classes in train set is 100.')
    parser.add_argument ('-n_s', '--n_support', type=int, default=1,
                        help='Number of support vectors inside one class. \
                            The following inequality should be satisfied: \
                            n_support + n_query < 78, since minimum number of \
                            utterances per person in SpeakingFaces Dataset in train set is 78.')
    parser.add_argument ('-n_q', '--n_query', type=int, default=1,
                        help='Number of query vectors inside one class. \
                            The following inequality should be satisfied: \
                            n_support + n_query < 78, since minimum number of \
                            utterances per person in SpeakingFaces Dataset in train set is 78.')

    # train
    parser.add_argument ('-n_epochs', '--num_epochs', type=int, default=100,
                        help='Number of epochs in training process.')
    parser.add_argument ('-save_dir', '--save_dir', default="/workdir/results",
                        help='Path to the directory where output data will be saved.')

    return parser