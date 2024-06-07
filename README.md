=======
One Model to Rule Them All: Unified Transformer for Biometric Matching
==
![One Model to Rule Them All: Unified Transformer for Biometric Matching](data/train_pipeline.png)

We present a transformer-based model for biometric verification, leveraging the adaptability of transformer architectures. Our approach involves joint training on audio, visual, and thermal data within a multimodal framework. By converting all three data types into an image format, we construct a unified system utilizing the Vision Transformer (ViT) architecture, with fully shared model parameters. Additionally, we extend the prototypical loss to accommodate the multimodal data, enabling the model to learn from all possible combinations of input modalities. 


## Dataset

The preprocessed data used for our experiments can be downloaded from [Google Drive](https://drive.google.com/drive/folders/16T3FKwBbCkrgaJhEGFDw8pqR_z30eP7U?usp=sharing).

The *SpeakingFaces* directory contains the compressed version of the preprocessed data used for the reported experiments on SpeakingFaces dataset. The train set is split into 5 parts that should be extracted into the same location. For each utterance in the train split, only the first frame (visual and thermal) is selected. For each utterance in the test and validation splits, 10 equidistant frames (visual and thermal) are selected. All 7 parts of the data, should be extracted to the same folder. 

The *SpeakingFaces/metadata* contains lists prepared for the train, validation, and test sets:
1) *train_list.txt* contains the paths to the recordings and the corresponding subject identifiers present in SpeakingFaces. 
2) The *valid_list.txt* and *test_list.txt* consist of randomly generated positive and negative pairs taken from the validation and test splits of SpeakingFaces, respectively. For each subject, the same number of positive and negative pairs were selected. In total, the numbers of pairs in the validation and test sets are 38,000 and 46,200, respectively.

The *VoxCeleb1* directory contains the compressed version of the preprocessed data used for the reported experiments on the VoxCeleb1 test split. The test set is split into 2 parts that should be extracted into the same location. For each utterance, 10 equidistant frames (visual and thermal) are selected. 
The *VoxCeleb1/metadata* contains the *test_list.txt*, provided first by original authors.

## Train
The following command launches training of our trimodal unified transformer on SpeakingFaces dataset
```
python main.py --data_type rgb wav thr --annotation_file annotations/annotations_file_SF_train_cleaned.csv --path_to_train_dataset $data_dir --path_to_valid_dataset $valid_dir --path_to_valid_list $valid_list --save_dir results --exp_name exp1 --num_epochs $n_epochs --n_ways 40 --n_batch 300 --lr 0.000004--weight_decay 0.01
```
The following command launches training of our audio-visual unified transformer on VoxCeleb dataset

## Reference

