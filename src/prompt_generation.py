from utlis import *



def my_generate_prompt_CoT_bs(TG, EK, Q):
    '''
    Generate the prompt for the model
    
    Args:
    TG: list of strings, temporal graph
    EK: list of strings, external knowledge
    Q: string, the question
    
    Returns:
    prompt: string, the prompt for the model
    '''
    prompt = f'Input:\n{{\n"Timeline":\n{json.dumps(TG)},\n"Question": {json.dumps(Q)},'

    if EK is not None:
        prompt += f'\n"Useful information":\n{json.dumps(EK)},'

    prompt += '\n"Instruction": "Let\'s think step by step. Only return me json."\n}\nOutput:\n```json\n'

    return prompt



def my_generate_prompt_TG_Reasoning(dataset_name, split_name, story, TG, EK, Q, CoT, A, f_ICL, Q_type=None, mode=None, eos_token="", f_no_TG=False):
    '''
    Generate the prompt for the model.

    args:
        dataset_name: string, dataset name
        split_name: string, split name
        story: string, story
        TG: list of strings or string, temporal graph
        EK: list of strings or string, exteral knowledge
        Q: string, question
        CoT: list of strings, chain of thought
        A: string, answer
        Q_type: string, question type
        mode: string, mode
        eos_token: string, eos token
        f_no_TG: bool, whether to use the temporal graph or original story as context

    return:
        prompt: string, the prompt
    '''
    if f_ICL and mode == 'test':
        if dataset_name == 'TGQA':
            split_name = f'_Q{Q_type}'

        strategy = 'TGR' if not f_no_TG else 'storyR'        
        file_path = f'../materials/{dataset_name}/prompt_examples_{strategy}{split_name}.txt'

        with open(file_path) as txt_file:
            prompt_examples = txt_file.read()


    context = f'"Timeline":\n{json.dumps(TG)},\n' if not f_no_TG else f'"Story":\n{json.dumps(story)},\n'

    if f_ICL and mode == 'test':
        prompt = f'Example:\n\n{prompt_examples}\n\nTest:\n\nInput:\n{{\n{context}"Question": {json.dumps(Q)},'
    else:
        prompt = f'Input:\n{{\n{context}"Question": {json.dumps(Q)},'

    if EK is not None:
        prompt += f'\n"Useful information":\n{json.dumps(EK)},'

    prompt += '\n"Instruction": "Let\'s think step by step. Only return me json."\n}\nOutput:\n```json\n'

    if CoT is not None and A is not None:
        if isinstance(CoT, list):
            CoT = CoT[0]
        CoT = CoT.replace('\n', ' ')
        prompt += f'{{\n"Thought": {json.dumps(CoT)},\n"Answer": {json.dumps(A)}\n}}\n```'
    
    prompt += eos_token
    return prompt



def my_generate_prompt_ICL(dataset_name, split_name, learning_setting, story, Q, C, f_ICL, f_shorten_story, f_using_CoT, Q_type=None):
    '''
    Gnerate the prompt for the model

    args:
    story: the story, str
    Q: the question, str
    C: the candidates, list
    Q_type: the question type, str

    return:
    prompt: the generated prompt, str
    '''
    if f_ICL: # use in-context learning
        if dataset_name == 'TGQA':
            Q_type = f'Q{Q_type}'

        if Q_type is None:
            file_path = f'../materials/{dataset_name}/prompt_examples_ICL_{learning_setting}{split_name}.txt'
        else:
            file_path = f'../materials/{dataset_name}/prompt_examples_ICL_{learning_setting}{split_name}_{Q_type}.txt'

        with open(file_path) as txt_file:
            prompt_examples = txt_file.read()

    if f_shorten_story: # shorten the story
        story = shorten_story(story)

    C = add_brackets(C)
    Q += ' Choose from ' + ', '.join(C) + '.'
    
    prompt = f"Example:\n\n{prompt_examples}\n\n\n\nTest:\n\nStory: {story}\n\nQuestion: {Q}" if f_ICL else f"Story: {story}\n\nQuestion: {Q}"
    prompt += "\n\nAnswer: Let's think step by step.\n\n" if f_using_CoT else "\n\nAnswer: "

    return prompt



def my_generate_prompt_TG_trans(dataset_name, story, TG, entities, relation, times, f_ICL, f_shorten_story, f_hard_mode, 
                                transferred_dataset_name, mode=None, eos_token="</s>", max_story_len=1500):
    '''
    Generate the prompt for text to TG translation (given context and keywords, generate the relevant TG)

    Args:
    - story: str or list, the story
    - TG: str or list, the TG
    - entities: str or list, the entities
    - relation: str, the relation
    - times: str or list, the times
    - f_ICL: bool, whether to use ICL
    - f_shorten_story: bool, whether to shorten the story
    - f_hard_mode: bool, whether to use hard mode
    - transferred_dataset_name: str, the name of the transferred dataset
    - mode: train or test
    - eos_token: str, the end of sentence token
    - max_story_len: int, the maximum length of the story (only valid when f_shorten_story is True)

    Returns:
    - prompt: str, the prompt
    '''

    def add_examples_in_prompt(prompt):
        if f_ICL and mode == 'test':
            file_path = f'../materials/{dataset_name}/prompt_examples_text_to_TG_Trans.txt' if (not f_hard_mode) else \
                        f'../materials/{dataset_name}/prompt_examples_text_to_TG_Trans_hard.txt'
            with open(file_path) as txt_file:
                prompt_examples = txt_file.read()
            prompt = f"\n\n{prompt_examples}\n\nTest:\n{prompt}"
        return prompt.strip() + '\n'


    # Convert the list to string
    entities = ' , '.join(add_brackets(entities)) if entities is not None else None
    times = ' , '.join(add_brackets(times)) if times is not None else None

    if f_shorten_story:
        story = shorten_story(story, max_story_len)
 
    if relation is None:
        # If we do not have such information extracted from the questions, we will translate the whole story.
        prompt = add_examples_in_prompt(f'### Input:\n{{\n"Story": {json.dumps(story)},\n"Instruction": "Summary all the events as a timeline. Only return me json."\n}}\n ### Output: \n```json')
    else:
        if f_hard_mode or entities is None or times is None:
            prompt = add_examples_in_prompt(f'### Input:\n{{\n"Story": {json.dumps(story)},\n"Instruction": "Summary {relation} as a timeline. Only return me json."\n}}\n ### Output: \n```json')
        else:
            prompt = add_examples_in_prompt(f'### Input:\n{{\n"Story": {json.dumps(story)},\n"Instruction": "Given the time periods: {times}, summary {relation} as a timeline. Choose from {entities}. Only return me json."\n}}\n ### Output: \n```json')
             
    # For training data, we provide the TG as label.
    if TG is not None:
        # If we want to test the transfer learning performance, we can change the format of the TG in TGQA to other datasets.
        TG = TG_formating_change(TG, dataset_name, transferred_dataset_name)

        prompt += f'{{\n"Timeline":\n{json.dumps(TG)}\n}}\n```'

    prompt += eos_token
    return prompt