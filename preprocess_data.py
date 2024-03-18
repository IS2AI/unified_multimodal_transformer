import numpy as np
import os
import shutil
import argparse

# Parse arguments
parser = argparse.ArgumentParser(description='Copy files for testing and validation from datasets.')
parser.add_argument('--dataset', '-ds', type=str, help='Dataset: voxceleb or speaking_faces')
parser.add_argument('--mode', '-m', type=str, help='Mode: validation/test for speaking_faces')
parser.add_argument('--num_files','-n', type=int, help='Number of files copy in thr and img modality')
parser.add_argument('--source_path','-source_p', type=str, help='Specifies the path where the files are stored.')
parser.add_argument('--dest_path','-dest_p', type=str, help='Specifies the path where the files will be copied')
args = parser.parse_args()

# Example paths:
# Speaking_faces paths (dataset = "speaking_faces"):
#   source_path = "/mnt/storage/datasets/sf_pv/data_v2/",
#   dest_path = "/mnt/storage/speaking_faces/test" (mode = test)
#   dest_path = "/mnt/storage/speaking_faces/val" (mode = valid)
# Voxceleb paths (dataset = "voxceleb"):
#   source_path = "/mnt/storage/datasets/VoxCeleb1/test"
#   dest_path = "/mnt/storage/Voxceleb1_abriged/test"

def copy_files_sf(source, dest, mode="test", num_files=10):
    modality_types = ['rgb', 'thr', 'wav']
    sub_begin_ind, sub_end_ind = (120, 142) if mode == "test" else (100, 119)
    
    for ind in range(sub_begin_ind, sub_end_ind+1):
        sub_folder = f"{source}sub_{ind}"
        for session_folder in os.listdir(sub_folder):
            session_path = os.path.join(sub_folder, session_folder)
            copy_files(modality_types, session_path, os.path.join(dest, f"sub_{ind}", session_folder), num_files)

def copy_files_voxceleb(source, dest, num_files=10):
    modality_types = ['rgb', 'thr', 'wav']
    for id_folder in sorted(os.listdir(source)):
        id_path = os.path.join(source, id_folder)
        for youtube_tag in os.listdir(id_path):
            if youtube_tag == '.ipynb_checkpoints':
                continue
            tag_path = os.path.join(id_path, youtube_tag)
            dest_path = os.path.join(dest, id_folder, youtube_tag)
            os.makedirs(dest_path, exist_ok=True)
            
            copy_files(modality_types, tag_path, dest_path, num_files)
                                
def copy_files(modality_types, source_path, dest_base_path, num_files):
    for modality in modality_types:
        # Construct paths
        modality_path = os.path.join(source_path, modality)
        dest_path = os.path.join(dest_base_path, modality)
        
        # Ensure destination path exists
        os.makedirs(dest_path, exist_ok=True)
        
        if os.path.exists(modality_path):
            if modality == 'wav':
                # Copy .wav files directly
                wav_files = [f for f in os.listdir(modality_path) if f.lower().endswith('.wav')]
                for file_name in wav_files:
                    shutil.copy(os.path.join(modality_path, file_name), os.path.join(dest_path, file_name))
            else:
                # Handle other modalities, considering both session folders and id_folders
                session_or_id_folders = [f for f in os.listdir(modality_path) if os.path.isdir(os.path.join(modality_path, f))]
                for folder in session_or_id_folders:
                    folder_path = os.path.join(modality_path, folder)
                    files = [f for f in os.listdir(folder_path) if f.lower().endswith('.jpg')]
                    sorted_files = np.array(sorted(files, key= lambda x: int(x.split(".")[0])))
                    indices = np.linspace(0, len(files) - 1, num=num_files, dtype=int)
                    selected_files = sorted_files[indices]
                    destination_folder_path = os.path.join(dest_path, folder)
                    os.makedirs(destination_folder_path, exist_ok=True)
                    for file_name in selected_files:
                        shutil.copy(os.path.join(folder_path, file_name), os.path.join(destination_folder_path, file_name))
                        
                        
# Main execution
if __name__ == "__main__":
    if args.dataset in ["sf", "speaking_faces"]:
        source = args.source_path
        dest = args.dest_path
        copy_files_sf(source, dest, args.mode, num_files=args.num_files)
    elif args.dataset in ["voxceleb1", "VC1"]:
        source = args.source_path
        dest = args.dest_path
        copy_files_voxceleb(source, dest, num_files=args.num_files)
