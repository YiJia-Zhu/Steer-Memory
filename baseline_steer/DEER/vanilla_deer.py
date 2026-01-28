
import warnings
warnings.filterwarnings("ignore")  # Ignore all warnings
from transformers import AutoModelForCausalLM, AutoTokenizer
import pdb
import torch
import torch.nn.functional as F
import json
import re
from tqdm import tqdm
import argparse
import os
from transformers.cache_utils import DynamicCache
from copy import deepcopy

def append_jsonl(data, file_path):
    """
    Append results from list to a .jsonl file.

    Parameters:
        data (list): List containing Python dictionaries to append.
        file_path (str): Target .jsonl file path.
    """
    with open(file_path, 'a', encoding='utf-8') as f:
        for item in data:
            json_line = json.dumps(item, ensure_ascii=False)
            f.write(json_line + '\n')

def write_jsonl(data, file_path):
    """
    Write results from list to a .jsonl file.

    Parameters:
        data (list): List containing Python dictionaries to write.
        file_path (str): Output .jsonl file path.
    """
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            # Convert each dictionary to JSON string and write to file
            json_line = json.dumps(item, ensure_ascii=False)
            f.write(json_line + '\n')

def read_jsonl(file_path):
    """
    Read .jsonl file and return a list of dictionaries.

    Parameters:
        file_path (str): Path to .jsonl file.

    Returns:
        data (list): List containing all JSON objects.
    """
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            # Parse JSON object from each line
            json_obj = json.loads(line.strip())
            data.append(json_obj)
    return data

def calcu_max_probs_w_kv(model, pred_input_ids, kv_cache, tokenizer, method=1):
    list1 = tokenizer([' }', '}', '}.', '}.\n', '}\\', '}}', ')}', ')}.', ')}\n', '</think>'])['input_ids']
    stop_ids = sum(list1, [])
    total_steps = 0
    if method == 0:
        total_prob_max = 1.0
    else:
        total_prob_max = 0.0

    pred_tokens = []
    last_token = -1
   
    backup_cache = deepcopy(kv_cache)
   
    with torch.no_grad():
        while last_token not in stop_ids:
            
            if last_token == -1:
                output_dicts = model(input_ids=pred_input_ids, past_key_values=backup_cache)
            else:
                output_dicts = model(input_ids=torch.tensor([last_token]).unsqueeze(0).to(pred_input_ids.device), past_key_values=backup_cache)
            logits = output_dicts['logits'][0][-1]
            past_key_values = output_dicts['past_key_values']
            probs = F.softmax(logits, dim=-1)

            max_value, max_index = torch.max(probs, dim=0)

            
            if last_token == -1:
                total_prob_max = total_prob_max
            else:

                if method == 0:
                    total_prob_max *= max_value
                else:
                    total_prob_max += max_value
            
            pred_tokens.append(max_index)
            last_token = max_index
            total_steps += 1
            if total_steps > 20:
                break

    
    if method != 0:
        total_prob_max = (total_prob_max - max_value) / (total_steps - 2)

    del backup_cache, past_key_values
    torch.cuda.empty_cache() 
    
    return total_prob_max.item()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', type=str, default="/mnt/data1/ycx/DeepSeek-R1-Distill-Qwen-32B", help="model directory")
    parser.add_argument('--threshold', type=float, default=0.95)
    parser.add_argument('--max_len', type=int, default=16384)
    parser.add_argument('--dataset', type=str, default='math')
    parser.add_argument('--output_path', type=str, default='./outputs')
    parser.add_argument('--log', type=bool, default=False)
    parser.add_argument('--think_ratio', type=float, default=0.9)

    args = parser.parse_args()
    return args

args = parse_args()

think_len = int(args.max_len * args.think_ratio)
answer_len = args.max_len - think_len

# model
model_name = args.model_name_or_path
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="bfloat16",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# data
sys_prompts = ['Please reason step by step, and put your final answer within \\boxed{}.']
questions_json = read_jsonl('./data/' + args.dataset + '/test.jsonl')

os.makedirs(f'{args.output_path}/{model_name}/{args.dataset}', exist_ok=True)
output_list = []

for i in tqdm(range(0, len(questions_json))):
    output_dict = {}
    sys_prompt = sys_prompts[0]
    
    prompt = questions_json[i]['problem'] #+ 'start ' * 30000
    answer = str(questions_json[i]['answer'])

    messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": prompt}
            ]

    text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
    )

    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    input_ids = model_inputs["input_ids"]
    input_length = len(input_ids[0])
    
    last_token_ids = tokenizer("**", add_special_tokens=False)["input_ids"] + tokenizer("</think>", add_special_tokens=False)["input_ids"] + [tokenizer.eos_token_id]
    continue_ids = tokenizer("Wait", add_special_tokens=False)["input_ids"]
    stop_ids = continue_ids + last_token_ids

    answer_prompt_ids = tokenizer("\n**Final Answer**\n\nThe final answer is \\boxed", add_special_tokens=False)["input_ids"]
    answer_ids = tokenizer(answer, add_special_tokens=False)["input_ids"]
    
    past_key_values = DynamicCache()
    first_round = True
    too_long = False
    while 1:
        if first_round:
           
            generated_dicts = model.generate(
                input_ids,
                #max_new_tokens=200,
                max_new_tokens=think_len-len(input_ids[0]),
                do_sample=False,
                eos_token_id=stop_ids,
                return_dict_in_generate=True, 
                output_logits=True,
                tokenizer=tokenizer,
                past_key_values=past_key_values,
                )
           
        else:
            generated_dicts = model.generate(
                input_ids, 
                max_new_tokens=think_len-len(input_ids[0]),
                do_sample=False,
                return_dict_in_generate=True, 
                output_logits=True,
                tokenizer=tokenizer,
                eos_token_id=stop_ids,
                past_key_values=past_key_values,
                )
            
        generated_ids = [
            output_ids[len(input_ids):-1] for input_ids, output_ids in zip(input_ids, generated_dicts['sequences'])
        ]
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        logits = generated_dicts['logits'][-1]
        probs = F.softmax(logits, dim=-1)[0]

        max_value, max_index = torch.max(probs, dim=0)

        if max_index in last_token_ids:
            real_stop = 1
        else:
            real_stop = 0

        pred_input_ids = torch.cat((input_ids, generated_ids[0].unsqueeze(0)), dim=1)

        if len(pred_input_ids[0]) >= think_len - 100:
            too_long = True
       
        pred_prob = calcu_max_probs_w_kv(model, torch.tensor(answer_prompt_ids).to(generated_ids[0].device).unsqueeze(0), past_key_values, tokenizer, 1)
       
        torch.cuda.empty_cache() 
        
        if pred_prob > args.threshold or real_stop or too_long:
            input_ids = torch.cat((pred_input_ids, torch.tensor(tokenizer('\n</think>\n\n')['input_ids']).to(generated_ids[0].device).unsqueeze(0)), dim=1) # with wait

            generated_dicts = model.generate(
                input_ids,
                max_new_tokens=answer_len,
                do_sample=False,
                return_dict_in_generate=True, 
                past_key_values=past_key_values,
            )

            generated_ids = [
                output_ids[len(input_ids):-1] for input_ids, output_ids in zip(input_ids, generated_dicts['sequences'])
            ]
            final_output_ids = torch.cat((input_ids[0], generated_ids[0]), dim=-1)
            response = tokenizer.batch_decode([final_output_ids[input_length:]], skip_special_tokens=True)[0]
            
            if args.log:
                log_file_path = args.output_path + "./outputs/log" + str(args.threshold) + ".txt"
                with open(log_file_path, "a") as file:
                    file.write(response + "\n")
            
            break
        
        else:
            tmp = torch.cat((generated_ids[0], torch.tensor(continue_ids).to(generated_ids[0].device)), dim=0)
            input_ids = torch.cat((input_ids, tmp.unsqueeze(0)), dim=1) # with wait
            torch.cuda.empty_cache() 
        
    output_dict['question'] = questions_json[i]['problem']
    output_dict['generated_responses'] = [response]
    output_dict['gold_answer'] = questions_json[i]['answer']

    append_jsonl([output_dict], args.output_path  + model_name + '/' + args.dataset + '/greedy_p' + str(args.threshold) + '_len' + str(args.max_len) + '.jsonl')