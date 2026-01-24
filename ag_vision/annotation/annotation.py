import pandas as pd
import os

def generate_classification_df(folder_location: str, img_list: list, downloaded_images: bool=False):
    df_list = []
    for split in ['train', 'test', 'valid']:
        split_dir = f"{folder_location}/{split}"
        if os.path.isdir(split_dir):
            print(f"Generating annotations from {split_dir}")
            a = os.listdir(f"{folder_location}/{split}")
            for label in a:
                rf_files =  os.listdir(f"{folder_location}/{split}/{label}")
                df = pd.DataFrame({'rf_file_name': rf_files})
                df.loc[:, 'class'] = label
                df.loc[:, 'split'] = split
                if downloaded_images:
                    df.loc[:, 'rf_file_name'] = df['rf_file_name'].apply(lambda x: x.replace('.rf.', '_rf_'))
                    df.loc[:, 'image_id'] = df['rf_file_name'].apply(lambda x: os.path.splitext(x)[0])
                else:
                    # if we pushed the images up the first part of the roboflow id is the original name.
                    df.loc[:, 'image_id'] = df['rf_file_name'].apply(lambda x: x.split('_')[0])

                df_list.append(df)

    id_list = [os.path.splitext(x)[0] for x in img_list]

    final_df = pd.concat(df_list).reset_index(drop=True)
    print(f"There were {len(final_df)} annotation found.")
    # only keeps annotations for images that are in the img_list
    final_df = final_df[final_df['image_id'].isin(id_list)]
    print(f"{len(final_df)} annotation belong to this batch.")

    return final_df