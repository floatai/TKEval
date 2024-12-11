import json
import sys
import re

input_filepath, output_filepath = sys.argv[1:]

result_list = []
with open(input_filepath, "r") as fpin:
    for line in fpin:
        info = json.loads(line.strip())
        gpt4_output = info["gpt4-turbo_prediction"]
        pattern = re.compile(r'```python\n.*```', re.DOTALL)
        function_code_idx = re.search(pattern, gpt4_output).span()
        function_code = gpt4_output[function_code_idx[0]: function_code_idx[1]]
        result_list.append([function_code[len("```python\n"): -len("```")]])
with open(output_filepath, "w") as fpout:
    json.dump(result_list, fpout)