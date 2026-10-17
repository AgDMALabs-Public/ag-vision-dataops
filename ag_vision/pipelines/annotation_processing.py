import os
import pandas as pd


def generate_annotations_table(anno_list: list, project_index: int = 8) -> pd.DataFrame:
    """
    Build the annotations table from a list of annotation JSON paths.

    Expected path layout (project_index points at <project>):
    /Volumes/<catalog>/production/data/<category>/annotations/object_detection/<project>/v1/<download_date>/<anno_id>.json
       0        1          2         3       4          5             6            7        8        9            10
    """
    anno_list = list(anno_list)
    anno_df = pd.DataFrame({'annotation_path': anno_list})

    anno_df['annotation_id'] = ['.'.join(os.path.basename(x).split('.')[:-1]) for x in anno_list]
    anno_df['annotation_task'] = [x.split('/')[project_index] for x in anno_list]
    anno_df['annotation_type'] = [x.split('/')[project_index - 1] for x in anno_list]
    anno_df['project'] = [x.split('/')[project_index - 3] for x in anno_list]
    anno_df['annotation_batch'] = [x.split('/')[project_index + 1] for x in anno_list]
    anno_df['download_date'] = [x.split('/')[project_index + 2] for x in anno_list]

    # image_path: same prefix up to <annotation_batch>, then images/<annotation_id>.jpg
    anno_df['image_path'] = [
        '/'.join(x.split('/')[:project_index + 2] + ['images', '.'.join(os.path.basename(x).split('.')[:-1]) + '.jpg'])
        for x in anno_list
    ]
    anno_df['download_date'] = pd.to_datetime(anno_df['download_date'].str.replace('_', '-'), format='%Y-%m-%d')
    
    return anno_df[['image_path', 'annotation_id', 'project', 'annotation_task',
                    'annotation_batch', 'annotation_type', 'annotation_path', 'download_date']]