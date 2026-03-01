import argparse
import os
import re
import shutil

import datasets
from datasets import concatenate_datasets


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="/seu_share2/home/fenglei/101013989/verl_agent/datasets/math/")
    parser.add_argument("--hdfs_dir", default=None)

    args = parser.parse_args()

    train_dataset = datasets.load_dataset("aaabiao/dapo_filter", split="train")
    test_dataset_aime24 = datasets.load_dataset("HuggingFaceH4/aime_2024", split="train")
    test_dataset_aime25 = datasets.load_dataset("MathArena/aime_2025", split="train")
    test_dataset_math500 = datasets.load_dataset("HuggingFaceH4/MATH-500", split="test")
    test_dataset_math500_first50 = test_dataset_math500.select(range(50))

    test_dataset_amc23 = datasets.load_dataset("knoveleng/AMC-23", split="train")
    test_dataset_minerva = datasets.load_dataset("zwhe99/simplerl-minerva-math", split="test")
    test_dataset_olympiadbench = datasets.load_dataset("realtreetune/olympiadbench", split="test")


    # add a row to each data item that represents a unique id
    def make_map_fn(split, data_source):
        _rm_suffix = re.compile(
            r"(\r?\n|\\n)*\s*Let'?s\s+think\s+step\s+by\s+step\s+and\s+output\s+the\s+final\s+answer\s+within\s*\\boxed\{\}\.\s*$",
            re.IGNORECASE
        )
        def process_fn(example, idx):
            prompt = example.pop("prompt")
            reward_model = example.pop("reward_model")
            ability = example.pop("ability")
            assert len(prompt) == 1
            prompt = prompt[0]["content"]
            instruction_following = 'Let\'s think step by step and output the final answer after "####".'
            question = _rm_suffix.sub("", prompt)+ " " + instruction_following
            #assert "Let's think step by step and output" not in question
            ground_truth = reward_model["ground_truth"]
            data = {
                "data_source": 'new_math_dapo',
                "data_source2": data_source,
                "ability": ability,
                #"reward_model": reward_model,
                "reward_model": {"style": "rule", "ground_truth": ground_truth},
                "prompt": [
                    {
                        "role": "user",
                        "content": question,
                    }
                ],
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "answer": ground_truth,
                    "question": question,
                },
                "env_kwargs": {"ground_truth": ground_truth, "question": question, "data_source": 'new_math_dapo'}

            }
            return data

        return process_fn

    def make_map_fn_test(split, data_source):
        def process_fn(example, idx):
            question = example.pop("problem")
            ability = "math"
            ground_truth = example.pop("answer")
            # change type of ground_truth to string if it is not
            if not isinstance(ground_truth, str):
                ground_truth = str(ground_truth)
            data = {
                "data_source": 'new_math_dapo',
                "data_source2": data_source,
                "ability": ability,
                "reward_model": {"style": "rule", "ground_truth": ground_truth},
                "prompt": [
                    {
                        "role": "user",
                        "content": question,
                    }
                ],
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "answer": ground_truth,
                    "question": question,
                },
                "env_kwargs": {"ground_truth": ground_truth, "question": question, "data_source": 'new_math_dapo'}

            }
            return data

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn("train", "dapo_filter"), with_indices=True)
    test_dataset_aime24 = test_dataset_aime24.map(function=make_map_fn_test("test", "aime24"), with_indices=True)
    test_dataset_aime25 = test_dataset_aime25.map(function=make_map_fn_test("test", "aime25"), with_indices=True)
    test_dataset_math500 = test_dataset_math500.map(function=make_map_fn_test("test", "math500"), with_indices=True)
    test_dataset_math500_first50 = test_dataset_math500_first50.map(function=make_map_fn_test("test", "math500_first50"), with_indices=True)
    test_dataset_amc23 = test_dataset_amc23.map(function=make_map_fn_test("test", "amc23"), with_indices=True)
    test_dataset_minerva = test_dataset_minerva.map(function=make_map_fn_test("test", "minerva"), with_indices=True)
    test_dataset_olympiadbench = test_dataset_olympiadbench.map(function=make_map_fn_test("test", "olympiadbench"), with_indices=True)

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    test_dataset_sampled = concatenate_datasets([test_dataset_math500_first50, test_dataset_aime24, test_dataset_aime25])
    print(f"Combined test_dataset_sampled length: {len(test_dataset_sampled)}")

    test_dataset_full = concatenate_datasets([test_dataset_math500,test_dataset_aime24, test_dataset_aime25, test_dataset_olympiadbench, test_dataset_amc23, test_dataset_minerva])
    print(f"Combined test_dataset_full length: {len(test_dataset_full)}")

    train_dataset.to_parquet(os.path.join(local_dir, "train.parquet"))
    test_dataset_sampled.to_parquet(os.path.join(local_dir, "test_sampled.parquet"))
    test_dataset_full.to_parquet(os.path.join(local_dir, "test.parquet"))

    if hdfs_dir is not None:
        os.makedirs(hdfs_dir, exist_ok=True)
        dest = os.path.join(hdfs_dir, os.path.basename(local_dir.rstrip(os.sep)))
        if os.path.exists(dest):
            shutil.rmtree(dest)
        shutil.copytree(local_dir, dest)