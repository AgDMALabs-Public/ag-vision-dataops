import logging
from pydantic import BaseModel

logger = logging.getLogger(__name__)


def _check_for_model_in_model(model, key: str, key_model):
    if not hasattr(model, key):
        setattr(model, key, key_model)
    elif getattr(model, key) is None:
        setattr(model, key, key_model)


def add_data_to_model(model_obj, key: str, value: float or str, overwrite: bool):
    if overwrite:
        print(f"Adding {key} to metadata.")
        setattr(model_obj, key, value)

    elif hasattr(model_obj, key):
        if getattr(model_obj, key) is not None:
            print(f'{key} key already exists in the metadata. Set Overwrite to True if you want to replace it.')
        else:
            print(f"Adding {key} to metadata.")
            setattr(model_obj, key, value)

    return model_obj


def add_first_level_nested_metadata(metadata_model: BaseModel, root_key: str, first_level_key: str,
                                    first_level_value: any, root_model: BaseModel,
                                    overwrite: bool = False) -> BaseModel:
    try:
        # add the root level if missing or is none
        _check_for_model_in_model(model=metadata_model,
                                  key=root_key,
                                  key_model=root_model)

        root_obj = getattr(metadata_model, root_key)

        root_obj = add_data_to_model(model_obj=root_obj,
                                     key=first_level_key,
                                     value=first_level_value,
                                     overwrite=overwrite)

    except Exception as e:
        metadata_model = metadata_model
        print(f"An unexpected error occurred: {e}")

    return metadata_model


def add_second_level_nested_metadata(metadata_model: BaseModel, root_model: BaseModel, root_key: str,
                                     first_level_model: BaseModel,
                                     first_level_key: str, second_level_key: str, second_level_val: any,
                                     overwrite: bool = False) -> BaseModel:
    try:
        _check_for_model_in_model(model=metadata_model,
                                  key=root_key,
                                  key_model=root_model)

        root_obj = getattr(metadata_model, root_key)

        _check_for_model_in_model(model=root_obj,
                                  key=first_level_key,
                                  key_model=first_level_model)

        fl_obj = getattr(root_obj, first_level_key)

        fl_obj = add_data_to_model(model_obj=fl_obj,
                                   key=second_level_key,
                                   value=second_level_val,
                                   overwrite=overwrite)

    except Exception as e:
        print(f"An unexpected error occurred: {e}")

    return metadata_model


def add_first_level_nested_model_metadata(metadata_model: BaseModel, root_model: BaseModel, root_key: str,
                                          first_level_model: BaseModel, first_level_key: str, pred: any,
                                          confidence: float,
                                          version: str, model_id: str, overwrite: bool = False) -> BaseModel:
    metadata_model = add_second_level_nested_metadata(metadata_model=metadata_model,
                                                      root_model=root_model,
                                                      root_key=root_key,
                                                      first_level_model=first_level_model,
                                                      first_level_key=first_level_key,
                                                      second_level_key='pred',
                                                      second_level_val=pred,
                                                      overwrite=overwrite)

    metadata_model = add_second_level_nested_metadata(metadata_model=metadata_model,
                                                      root_model=root_model,
                                                      root_key=root_key,
                                                      first_level_model=first_level_model,
                                                      first_level_key=first_level_key,
                                                      second_level_key='confidence',
                                                      second_level_val=confidence,
                                                      overwrite=overwrite)

    metadata_model = add_second_level_nested_metadata(metadata_model=metadata_model,
                                                      root_model=root_model,
                                                      root_key=root_key,
                                                      first_level_model=first_level_model,
                                                      first_level_key=first_level_key,
                                                      second_level_key='model_version',
                                                      second_level_val=version,
                                                      overwrite=overwrite)

    metadata_model = add_second_level_nested_metadata(metadata_model=metadata_model,
                                                      root_model=root_model,
                                                      root_key=root_key,
                                                      first_level_model=first_level_model,
                                                      first_level_key=first_level_key,
                                                      second_level_key='model_id',
                                                      second_level_val=model_id,
                                                      overwrite=overwrite)

    return metadata_model


def add_notes_to_metadata(metadata_dict: dict, msg: any, author: str) -> dict:
    if 'notes' not in metadata_dict:
        metadata_dict['notes'] = {}

    for x in range(100):
        note_name = f'note_{x}'
        if note_name not in metadata_dict['notes']:
            metadata_dict['notes'][note_name] = {}
            logger.info(f'Adding {note_name} ')
            break

    metadata_dict['notes'][note_name]['author'] = author
    metadata_dict['notes'][note_name]['msg'] = msg

    return metadata_dict


def add_flat_schema_data_to_metadata(metadata_dict: dict, flat_key: str, val: any, delimiter: str = ':',
                                     overwrite=False) -> dict:
    parsed_key = flat_key.split(delimiter)
    depth = len(parsed_key)
    if depth == 1:
        metadata_dict[parsed_key[0]] = val
        return metadata_dict

    elif depth == 2:
        metadata_dict = add_first_level_nested_metadata(metadata_dict=metadata_dict,
                                                        root_key=parsed_key[0],
                                                        first_level_key=parsed_key[1],
                                                        first_level_value=val,
                                                        overwrite=overwrite)
        return metadata_dict

    elif depth == 3:
        metadata_dict = add_second_level_nested_metadata(metadata_dict=metadata_dict,
                                                         root_key=parsed_key[0],
                                                         first_level_key=parsed_key[1],
                                                         second_level_key=parsed_key[2],
                                                         val=val,
                                                         overwrite=overwrite)
        return metadata_dict
    else:
        logger.warning(f"This function currently does not support nested dicts greater than 3.")


def flatten_schema(schema_model: BaseModel, parent_key: str = "", sep: str = ":") -> dict:
    """

    """
    schema_dict = schema_model.model_dump()

    items = []
    for key, value in schema_dict.items():
        new_key = f"{parent_key}{sep}{key}" if parent_key else key  # Concatenate parent and current key
        if isinstance(value, dict):  # If the value is another dictionary, recurse
            items.extend(flatten_schema(value, new_key, sep=sep).items())
        else:
            items.append((new_key, value))  # Add the flattened key-value pair

    return dict(items)


def generate_nested_schema_from_flat(schema_dict: dict, sep: str = ":") -> dict:
    """

    """
    out_dict = {}
    for key, value in schema_dict.items():
        out_dict = add_flat_schema_data_to_metadata(metadata_dict=out_dict,
                                                    flat_key=key,
                                                    val=value,
                                                    delimiter=sep)
    return out_dict
