import argparse

def createParser():
    parser = argparse.ArgumentParser(description='Input parameters to train the model.')

    # device
    parser.add_argument ('-n_gpu', '--n_gpu', type=int, default=0,
                        help='Number of GPU core.')

    # dataset
    parser.add_argument ('-afile', '--annotation_file', default="annotations_file_short_SF.csv",
                        help='Path to the annotation_file.')
    parser.add_argument ('-ddir', '--dataset_dir', default="/workdir/sf_pv/data_v2",
                        help='Path to the dataset directory.')

    # train_dataloader
    parser.add_argument ('-n_b', '--n_batch', type=int, default=100, 
                        help='Number of batches in train dataloader')
    parser.add_argument ('-n_w', '--n_ways', type=int, default=30,
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
    # valid_dataloader
    parser.add_argument ('-valid_batch_size', '--valid_batch_size', type=int, default=64, 
                        help='Batch size in valid dataloader')
    
    # loss
    parser.add_argument ('-dist_type', '--dist_type', default='squared_euclidean',
                        help='Distance type to calculate in Prototypical Loss function \
                              (default: "squared_euclidean"). Can be either "squared_euclidean" or "cosine_similarity"')

    # model
    parser.add_argument ('-model_choice', '--model_choice', default="resnet1",
                        help='if resnet1: resnet (from pytorch), \
                              if resnet2: resnet (from timm), \
                              if vit1: vit (from timm)')
    parser.add_argument ('-fine_tune', '--fine_tune', type=int, choices=(0,1), default=0,
                        help='Allows to choose between two types of transfer learning: fine tuning and feature extraction. \
                            If "1" it means fine tuning mode, i.e. train all weights,  \
                            If "0", it means feature extraction mode, i.e. train last classifier layer. \
                            For more details of the description of each mode, \
                            read https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html')
    parser.add_argument ('-exp_name', '--exp_name', default="exp1",
                        help='Experiment name.')
    


    # train
    parser.add_argument ('-n_epochs', '--num_epochs', type=int, default=100,
                        help='Number of epochs in training process.')
    parser.add_argument ('-save_dir', '--save_dir', default="/workdir/results",
                        help='Path to the directory where output data will be saved.')
    parser.add_argument ('-modality', '--modality', choices=("rgb","thr", "wav"), default="rgb",
                        help='Allows to choose modality, it can be either "rgb" or "thrm". \
                            If "rgb" it means model train and evaluate on rgb images, \
                            If "thr", it means model train and evaluate on thermal images, \
                            If "wav", it means model train and evaluate on wav files.')

    return parser