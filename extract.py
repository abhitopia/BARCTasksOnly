#%%
import json
from pathlib import Path
import jsonlines
from tqdm import tqdm

def convert_to_arc_format(pair_list):
    if not isinstance(pair_list, list):
        raise ValueError('pair_list is not a list.')
    train_list = []
    test_list = []
    if len(pair_list) < 10:
        number_of_tests = 1
    elif len(pair_list) < 20:
        number_of_tests = 2
    else:
        number_of_tests = 3
    for pair_index, pair_item in enumerate(pair_list):
        # print(f'Item {pair_index}: {pair_item} {len(pair_item)}')
        if len(pair_item) != 2:
            raise ValueError(f'Item {pair_index} does not have 2 elements.')
        input_image = pair_item[0]
        output_image = pair_item[1]
        dict = {'input': input_image, 'output': output_image}
        if pair_index >= len(pair_list) - number_of_tests:
            test_list.append(dict)
        else:
            train_list.append(dict)
    arc_dict = {'train': train_list, 'test': test_list}
    return arc_dict


def extract_from_jsonl(path, folder_name='extracted'):
    progress_bar = tqdm(desc='Extracting', unit='files', total=100000)
    output_folder = Path(path).parent / folder_name
    output_folder.mkdir(exist_ok=True)

    with jsonlines.open(path) as reader:
        for idx, obj in enumerate(reader):
            progress_bar.update(1)
            output_file = output_folder / f'{idx}.json'

            if output_file.exists():
                continue

            # Process each object
            examples = obj['examples']
            arc_dict = convert_to_arc_format(examples)
            json.dump(arc_dict, output_file.open('w'), indent=2)

# %%
# path = "data/100k_gpt4o-mini_generated_problems.jsonl"
# extract_from_jsonl(path, folder_name='100k_gpt4o-mini_generated_problems')

# path = "/data/100k-gpt4-description-gpt4omini-code_generated_problems.jsonl"
# extract_from_jsonl(path, folder_name='100k-gpt4-description-gpt4omini-code_generated_problems')

# path = 'data/data_100k.jsonl'
# extract_from_jsonl(path, folder_name='data_100k')

path = 'data/data_suggestfunction_100k.jsonl'
extract_from_jsonl(path, folder_name='data_suggestfunction_100k')
# %%
