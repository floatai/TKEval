import argparse
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from utils import construct_reponse, evaluate_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="./models/Llama-2-7b-hf/")
    parser.add_argument("--data_path", type=str, default="./data/identify_math_theorems_all_data_0123_shots")
    parser.add_argument("--eval_metric", type=str, choices=["multiple_choice", "generation"])
    parser.add_argument("--output_path", type=str)
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()

    if args.eval_metric == "multiple_choice":
        batch_size = 1
    else:
        batch_size = args.batch_size

    model = AutoModelForCausalLM.from_pretrained(args.model_path,
                                                 device_map="auto",
                                                 torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path,
                                              padding_side="left")

    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token

    with open(args.data_path, "r") as f:
        task_data = json.load(f)

    pred_res = {}
    for num_shot, data in task_data.items():
        data_with_result = construct_reponse(model,
                                             tokenizer,
                                             data,
                                             metric_type=args.eval_metric,
                                             batch_size=batch_size)
        pred_res[num_shot] = data_with_result
        eval_result = evaluate_metrics(data_with_result, args.eval_metric)
        print("shot: ", num_shot, ", res: ", eval_result)

    with open(args.output_path, "w") as fpout:
        json.dump(pred_res, fpout, indent=4)
