import pandas as pd
import os
from ag_vision.constants import paths

def generate_rf_img_id(in_str):
    a = in_str.split('.rf.')[0]
    a = a.split('_')[:-1]
    return '_'.join(a)


def generate_classification_df(folder_location: str, project_path: str, annotation_type: str, task_name: str = 'roboflow'):
    df_list = []
    for split in ['train', 'test', 'valid']:
        split_dir = f"{folder_location}/{split}"
        if os.path.isdir(split_dir):
            print(f"Generating annotations from {split_dir}")
            a = os.listdir(f"{folder_location}/{split}")
            for label in a:
                tmp_dir = f"{folder_location}/{split}/{label}"
                rf_files =  os.listdir(tmp_dir)
                df = pd.DataFrame({'rf_file_name': rf_files})
                df['tmp_path'] = tmp_dir + '/' + df['rf_file_name']
                df.loc[:, 'class'] = label
                df.loc[:, 'split'] = split

                df['img_id'] = df['rf_file_name'].apply(lambda x: generate_rf_img_id(x))
                df['ext'] = df['rf_file_name'].apply(lambda x: os.path.splitext(x)[1])
                df['img_id_len'] = df['img_id'].apply(lambda x: len(x))
                # uuid's have a len of 36 this will only affect images that do not have a image saved alread on the FG side.
                df['img_id'] = df['img_id'].apply(lambda x: x if len(x) < 36 else x[-36:])
                df['save_img_name'] = df['img_id'] + df['ext']

                df['save_path'] = df.apply(lambda x:
                                           paths.annotation_image_path(project=project_path,
                                                                       annotation_type=annotation_type,
                                                                       task_name=task_name,
                                                                       batch_name='roboflow',
                                                                       f_name=f'{x.save_img_name}'),
                                           axis=1)

                df_list.append(df)

    final_df = pd.concat(df_list).reset_index(drop=True)
    print(f"There were {len(final_df)} annotation found.")

    return final_df