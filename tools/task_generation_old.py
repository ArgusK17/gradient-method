import csv
import json
import random
from typing import List, Optional, Tuple
import os

class icl_task:
   def __init__(self):
      self.zero_shot=None
      self.few_shot=None
      self.answers=None
      self.info={}

def generate_index_lists(cand_list, list_size, number_of_lists, mode=None):
    base_list = list(cand_list)
    generated_lists = []
    
    if mode == "same_task":
        # Choose the first element to be the same in all lists
        common_first_element = random.choice(base_list)
        for _ in range(number_of_lists):
            # Ensure the common element is always included, then fill the rest randomly
            list_with_common_first = [common_first_element] + random.sample([x for x in base_list if x != common_first_element], list_size - 1)
            generated_lists.append(list_with_common_first)
            
    elif mode == "same_examples":
        # Choose the elements to be common (except the first element)
        common_elements = random.sample(base_list, list_size - 1)
        for _ in range(number_of_lists):
            # Ensure each list has a unique first element, then add the common elements
            unique_first_element = random.choice([x for x in base_list if x not in common_elements])
            list_with_common_elements = [unique_first_element] + common_elements
            generated_lists.append(list_with_common_elements)
            
    else:  # mode is None or any other value, generate completely random lists
        for _ in range(number_of_lists):
            generated_list = random.sample(base_list, list_size)
            generated_lists.append(generated_list)
    
    return generated_lists

def gen_translation_task(prompt_num=1, example_num=3, seed=0, lg_from=1, lg_to=0, mode: Optional[str]=None):    

    '''
    This function generates "language_from -> language_to" form in-context learning prompts.
    Language indecies:
    0: English
    1: French
    2: Spanish
    3: Italian
    '''
    random.seed(seed)
    task=icl_task()

    file_path = os.path.abspath(__file__)
    folder_path = os.path.dirname(file_path)
    vocabulary = []
    with open(folder_path+'/data/vocabulary.csv', mode='r', newline='', encoding='utf-8') as file:
        reader = csv.reader(file)
        header = next(reader)
        for row in reader:
            vocabulary.append(row)

    candidate_list = range(len(vocabulary))

    prompts_few_shot = []
    prompts_zero_shot = []
    answers = []

    pair_lists = generate_index_lists(candidate_list, example_num+1, prompt_num, mode)
    for pair_list in pair_lists:
        task_id = pair_list[0]
        index_list = pair_list[1:]
        prompt=""
        for pair_id in index_list:
            prompt=prompt+vocabulary[pair_id][lg_from]+"->"+vocabulary[pair_id][lg_to]+"\n"
        prompt=prompt+vocabulary[task_id][lg_from]+"->"
        prompts_few_shot.append(prompt)
        prompts_zero_shot.append(vocabulary[task_id][lg_from]+"->")

        answers.append(vocabulary[task_id][lg_to])

        task.few_shot = prompts_few_shot
        task.zero_shot = prompts_zero_shot
        task.answers = answers

    return task


def gen_cc_task(prompt_num=1, example_num=3, seed=0, train=True, mode: Optional[str]=None):
    '''
    This function generates "country -> capital" form in-context learning prompts.
    '''

    random.seed(seed)
    task=icl_task()

    file_path = os.path.abspath(__file__)
    folder_path = os.path.dirname(file_path)
    with open(folder_path+'/data/country_capital.json', mode='r', encoding='utf-8') as file:
        data = json.load(file)
    vocabulary=list(data.items())

    candidate_list = range(len(vocabulary))

    # train_test_split=100
    # if train:
    #     candidate_list = range(train_test_split)
    # else:
    #     candidate_list = range(train_test_split,len(vocabulary))

    prompts_few_shot = []
    prompts_zero_shot = []
    answers = []

    pair_lists = generate_index_lists(candidate_list, example_num+1, min(prompt_num,len(vocabulary)), mode)
    for pair_list in pair_lists:
        task_id = pair_list[0]
        index_list = pair_list[1:]
        prompt=""
        for pair_id in index_list:
            prompt=prompt+vocabulary[pair_id][0]+"->"+vocabulary[pair_id][1]+"\n"
        prompt=prompt+vocabulary[task_id][0]+"->"
        prompts_few_shot.append(prompt)
        prompts_zero_shot.append(vocabulary[task_id][0]+"->")

        answers.append(vocabulary[task_id][1])

        task.few_shot = prompts_few_shot
        task.zero_shot = prompts_zero_shot
        task.answers = answers

    return task

