import pandas as pd
import os

def generate_classification_df(folder_location: str, img_list: list):
    df_list = []
    for split in ['train', 'test', 'valid']:
        a = os.listdir(f"{folder_location}/{split}/")
        for label in a:
            rf_files =  os.listdir(f"{folder_location}/{split}/{label}/")
            df = pd.DataFrame({'rf_file_name': rf_files})
            df.loc[:, 'class'] = label
            df.loc[:, 'split'] = split
            df.loc[:, 'image_id'] = df['rf_file_name'].apply(lambda x: x.split('_')[0])
            df_list.append(df)

    id_list = [os.path.splitext(x)[0] for x in img_list]
    final_df = pd.concat(df_list).reset_index(drop=True)

    # only keeps annotations for images that are in the img_list
    final_df = final_df[final_df['image_id'].isin(id_list)]

    return final_df