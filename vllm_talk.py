from datasets import load_dataset
from litellm import acompletion
import asyncio
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm.asyncio import tqdm
from typing import Optional, Dict
from utils import get_circle_list
from datasets import Dataset

import click

from dotenv import load_dotenv
load_dotenv()
import re

def extract_first_number(text):
    # 正则表达式匹配第一个数字
    match = re.search(r'[+-]?\d+(\.\d+)?', text)
    if match:
        return match.group(0)  # 返回匹配到的第一个数字
    else:
        return None  # 如果没有找到数字
SHARD_SIZE = 1000000
MAX_QUEUE_SIZE = 100

import litellm
litellm.enable_json_schema_validation=True
#TASK：
#1. answer_questionaire
#2. extract_answer

letter2int = {
        'A':0,
        'B':1,
        'C':2,
        'D':3,
        'E':4,
        'F':5
    }
options = [
            "Y not like X at all",
            "Y not like X",
            "Y a little like X",
            "Y moderately like X",
            "Y like X",
            "Y very much like X"
        ]
new_options = get_circle_list(options)

async def async_completion(
    inputs, model: str, api_base: Optional[str] = None
):
    try:
        results = []
        letters = []
        #inputs: (question_id, person, question, origin_response)
        #outputs: (question_id, person, question, origin_response, response, results, letters, max_num_letters)
        question_id, person, question, origin_response = inputs
        prompt = "Let's role-play. I will ask you a question and you must give me an answer. I want you to act as the person described below. Think from the person's perspective. \n\n"
        prompt += f"{person} \n\n"
        prompt += "Use the given information to answer the question below. \n\n"
        prompt += f"{question} \n Answer:"
        messages = [
            {"role": "user", "content": prompt}
        ]
        if 'Meta' in model and '3' in model:
            response = await acompletion(
                model=model,
                messages=messages,
                api_base=api_base,
                num_retries=3,
                stop=["<|eot_id|>", "<|end_of_text|>"],
                temperature=0,
                # max_completion_tokens=gen_kwargs['max_completion_tokens']
            )
        else:
            response = await acompletion(
                model=model,
                messages=messages,
                api_base=api_base,
                num_retries=3,
                temperature=0,
            )
        response = response.choices[0].message.content
        i = 0
        for op in new_options:
            extract_answer_example =  f"""
            Question: {question}
            There is a person X whose answer is {origin_response}.
            There is a person Y whose answer is {response}.
            Disregarding the fact that X has not provided a detailed explanation, please assess the similarity between Y's and X's responses. Below are the options for similarity:
            Use the above information to determine which option B's response is more consistent with A's response.
            Options: A. {op[0]} B. {op[1]} C. {op[2]} D. {op[3]} E. {op[4]} F. {op[5]}
            Please give me the choice letter first and then give me the reason.
            Please return the results in the following format:
            Letter of option. [Reason.]
            """
            messages = []
            messages.append({"role":"user", "content": extract_answer_example})
            if 'Llama' in model and '3' in model:
                new_response = await acompletion(
                    model=model,
                    messages=messages,
                    api_base=api_base,
                    num_retries=3,
                    stop=["<|eot_id|>", "<|end_of_text|>"],
                    temperature=0,
                    # **gen_kwargs,
                )
            else:
                new_response = await acompletion(
                    model=model,
                    messages=messages,
                    api_base=api_base,
                    num_retries=3,
                    temperature=0,
                    # **gen_kwargs,
                )
            result = new_response.choices[0].message.content
            results.append(result)
            # 提取方法：首字母;The final answer is E;
            if result[0] in letter2int.keys():
                letters.append((letter2int[result[0]]+i)%6)
            else:
                if 'The final answer is' in result:
                    match = re.search(r'\b[A-F]\b', result)
                    letters.append((letter2int[match.group()]+i)%6)
                elif 'The answer is:' in result:
                    match = re.search(r'\b[A-F]\b', result)
                    letters.append((letter2int[match.group()]+i)%6)
                elif 'The answer is ' in result:
                    match = re.search(r'\b[A-F]\b', result)
                    letters.append((letter2int[match.group()]+i)%6)
                elif 'The correct answer is:' in result:
                    match = re.search(r'\b[A-F]\b', result)
                    letters.append((letter2int[match.group()]+i)%6)
                else:
                    import pdb; pdb.set_trace()
                # letters.append('')
            i+=1
        print(letters)
        return (question_id, person, question, origin_response, response, results, letters)
    except Exception as e:
        print(e)
        results = None

    return results


async def producer(queue, examples):
    for example in examples:
        await queue.put(example)
    print("Producer: Finished adding examples to the queue.")


async def save_shard(data, output_path):
    """Asynchronously save a shard of data to parquet format"""
    # import pdb
    # pdb.set_trace()
    df = pd.DataFrame(data)
    table = pa.Table.from_pandas(df)
    shard_file_path = output_path

    # Use run_in_executor to handle file I/O asynchronously
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, pq.write_table, table, shard_file_path)


async def shard_manager(queue, output_path):
    """Manages saving shards of data"""
    current_shard = []

    while True:
        try:
            item = await queue.get()

            if item is None:  # Sentinel value to indicate completion
                if current_shard:  # Save final partial shard
                    await save_shard(current_shard, output_path)
                break

            current_shard.append(item)

            if len(current_shard) >= SHARD_SIZE:
                await save_shard(current_shard, output_path)
                current_shard = []

            queue.task_done()

        except Exception as e:
            print(f"Error in shard manager: {e}")
            queue.task_done()


async def consumer(queue, shard_queue, model, api_base, pbar):
    while True:
        try:
            question_id, person, question, origin_response = await queue.get()
            inputs = (question_id, person, question, origin_response)
            (question_id, person, question, origin_response, response, results, letters) = await async_completion(inputs, model, api_base=api_base)
            data = {
                'question_id':question_id, 
                'person_info':person, 
                'question':question, 
                'origin_response':origin_response,
                'response':response,
                'results':str(results),
                'letters':str(letters)
                }
            await shard_queue.put(data)
            pbar.update(1)
            queue.task_done()
        except asyncio.CancelledError:
            print("Consumer: Task cancelled.")
            break


async def run(
    examples,
    model,
    output_path: str,
    concurrency: int = 5,
    api_base: Optional[str] = None,
):
    input_queue = asyncio.Queue(maxsize=MAX_QUEUE_SIZE)
    shard_queue = asyncio.Queue()

    pbar = tqdm(total=len(examples), desc='Processing examples', position=0, unit='examples')

    # Start the producer, consumers, and shard manager
    producer_task = asyncio.create_task(producer(input_queue, examples))
    shard_manager_task = asyncio.create_task(shard_manager(shard_queue, output_path))

    consumer_tasks = [
        asyncio.create_task(consumer(input_queue, shard_queue, model, api_base, pbar))
        for _ in range(concurrency)
    ]

    # Wait for producer and consumers to complete
    await producer_task
    await input_queue.join()

    # Cancel consumer tasks
    for task in consumer_tasks:
        task.cancel()

    # Signal shard manager to finish
    await shard_queue.put(None)
    await shard_manager_task

    pbar.close()


@click.command()
@click.argument('dataset')
@click.argument('model_name')
@click.option('--concurrency', default=3, help='The number of concurrent requests to make to the model.')
@click.option('--api_base', default="http://localhost:8813/v1", help='The base URI for the model.')
@click.option('--seed', default=2024, help='The random seed.')
@click.option('--output_path', default='Datasets/test', help='The path to save the output dataset.')
def main(dataset, model_name, concurrency, api_base, seed, output_path):
    # seed the random number generator
    import random
    import numpy as np
    
    random.seed(seed)
    np.random.seed(seed)

    #=========数据集加载===========
    print(f'==> input_dataset: {dataset}')
    df = pd.read_parquet(dataset)
    ds = Dataset.from_pandas(df)
    #===========================
    examples = []
    tqb = tqdm(total=len(ds), desc='person + question examples', position=0, unit='examples')
    # import pdb
    # pdb.set_trace()
    for i, example in enumerate(ds):
        # if i==5:
        #     break
        examples.append((example['question_id'], example['person_infor'], example['question'], example['origin_response']))
        tqb.update(1)
            
    asyncio.run(
        run(
            examples,
            f'openai/{model_name}' if (model_name.startswith('Qwen') or "Llama" in model_name or "gemma") else model_name,
            output_path=output_path,
            concurrency=concurrency,
            api_base=api_base,
        )
    )


if __name__ == '__main__':
    main()