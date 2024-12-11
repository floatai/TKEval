import pdb
import time
import re
import logging
import numpy as np
import scipy
from typing import Dict, List, Any, Union, Optional
import hashlib
import torch
from tqdm import tqdm
"""
We have two functions for the model to be evaluated on these tasks:
1) generate_text()
2) cond_log_prob()
One for generative evaluation and one for multiple-choice evaluation
"""


def argmax(array):
    """argmax with deterministic pseudorandom tie breaking."""
    max_indices = np.arange(len(array))[array == np.max(array)]
    idx = int(hashlib.sha256(np.asarray(array).tobytes()).hexdigest(),
              16) % len(max_indices)
    return max_indices[idx]


def get_prompt(
    letter_set: List[str],
    correct_words: List[str],
    incorrect_words: List[str],
    score: int,
    current_round: int,
    total_rounds: int,
) -> str:
    """Generate the prompt text for a single game round."""
    prompt = (
        f"Instructions: Please give as many examples as possible of words with "
        f"four or more letters, using only the letters: "
        f'{" ".join(letter_set).upper()}. Letters may be repeated. You will '
        f"have {total_rounds} rounds to enter words. Please separate words with "
        f"a comma. i.e. Guess: word1, word2, ... More points will be given to "
        f"longer words.\n\n")

    prompt += (f'Correct words: {", ".join(correct_words)}.\n'
               f'Incorrect words: {", ".join(incorrect_words)}.\n'
               f"Score: {score}.\n"
               f"Round: {current_round+1} of {total_rounds}.\n\n")

    prompt += f"Guess: "

    return prompt


def _process_attempt(attempt: str) -> List[str]:
    """Converts attempt to a list of words."""
    words = [x for x in re.split("[^a-z]", attempt.lower()) if len(x) > 0]
    return words


def _valid_scorer(word: str) -> int:
    """Gives the score of a word assumed to be valid."""
    if len(word) == 4:
        score = 1
    else:
        score = len(word)
        if len(set(word)) == 7:
            score += 7
    return score


def word_scorer(word_list: List[str], word: str) -> int:
    if word in word_list:
        return _valid_scorer(word)
    else:
        return 0


class SpellingBeeGame(object):
    """A single instance of the spelling bee game."""

    def __init__(self, game_data, num_rounds=20):
        """Game based on NYT Spelling Bee.

        Args:
          game_data: A dictionary specifying a single game.
          num_rounds: The total number of rounds of guessing allowed.
        """
        self.letter_set = game_data["letter_set"]
        self.word_list = game_data["word_list"]
        self.pangrams = game_data["pangrams"]
        self.max_score = game_data["max_score"]
        self.word_count = game_data["word_count"]

        self.correct_words = []
        self.incorrect_words = []
        self.score = 0
        self.round = 0
        self.total_rounds = num_rounds

    def _score_round(self, attempt: str) -> int:
        """Extract words, score and add to correct and incorrect lists."""
        words = _process_attempt(attempt)
        for word in words:
            score = word_scorer(self.word_list, word)
            if score > 0:
                self.score += score
                self.correct_words.append(word)
            else:
                self.incorrect_words.append(word)

    def play_round(self, model, tokenizer) -> None:
        """Play a single round of the game."""
        prompt = get_prompt(
            self.letter_set,
            self.correct_words,
            self.incorrect_words,
            self.score,
            self.round,
            self.total_rounds,
        )
        logging.info(prompt)
        attempt = generate_text(model,
                                tokenizer,
                                inputs=prompt,
                                max_length=2046,
                                stop_string=None,
                                output_regex=None)
        logging.info(attempt)
        self._score_round(attempt)
        self.round += 1

    def play(self, model, tokenizer) -> Dict[str, int]:
        """Play a whole game."""
        for _ in range(self.round, self.total_rounds):
            self.play_round(model, tokenizer)

        score_dict = {
            "score": self.score,
            "num_correct": len(self.correct_words),
            "num_incorrect": len(self.incorrect_words),
            "max_score": self.max_score,
            "word_count": self.word_count,
        }

        return score_dict


def _aggregate_scores(game_scores: List[Dict[str, int]]):
    """Aggregate scores over multiple games into a ScoreData object."""

    relative_score = []
    recall = []
    num_incorrect = []
    precision = []

    for score_dict in game_scores:

        relative_score.append(score_dict["score"] / score_dict["max_score"])
        recall.append(score_dict["num_correct"] / score_dict["word_count"])
        num_incorrect.append(score_dict["num_incorrect"])

        # Precision -- defined to be 1 if no words are guessed.
        num_guesses = score_dict["num_correct"] + score_dict["num_incorrect"]
        if num_guesses > 0:
            precision.append(score_dict["num_correct"] / num_guesses)
        else:
            precision.append(1.0)

    relative_score = np.mean(relative_score)
    recall = np.mean(recall)
    num_incorrect = np.mean(num_incorrect)
    precision = np.mean(precision)

    return relative_score

    return score_data


def postprocess_output(text, max_length, stop_string, output_regex):
    """
    Modify model output to satisfy stop_string and output_regex keywords.

    Args:
      text: A string or list of strings containing model outputs.
      max_length: Model output will be truncated to be at most this length.
      stop_string: Model output will be truncated to the shortest string
        which includes stop_string. If None, no truncation will be performed.
        e.g. if stop_string='.', the model output "The apple is on the
        ground. What is" will be truncated to "The apple is on the ground."
        before being returned.
      output_regex: Rather than returning the full output, return the first
        match to the python regular expression output_regex in the output
        string. If there is no match, an empty string will be returned. If
        output_regex=None, no regular expression matching will be performed.
        e.g. if output_regex=r"\\d+", and the model output is "= 42. 7*7 =
        49. 7*8 =", then "42" will be returned.

    Returns: A string or list of strings, all processed as described for each
      argument.
    """

    if isinstance(text, list):
        return [
            postprocess_output(mo, max_length, stop_string, output_regex)
            for mo in text
        ]

    # Ensure it is a string (will convert from bytes, ... as needed)
    if not isinstance(text, str):
        text = str(text, "utf-8")

    # truncate at max_length
    if max_length:
        text = text[:max_length]

    # Remove all text after any stop_string
    if stop_string:
        index = text.find(stop_string)
        if index > 0:
            text = text[:index + len(stop_string)]

    # extract substring matching regex (empty string for no match)
    if output_regex:
        _text = text
        text = next(iter(re.findall(output_regex, text)), "")
        assert (
            not type(text) is tuple
        ), f'Regex {output_regex} returned multiple matching groups when applied to string {_text}. Try using non-capturing groups, by starting regex groups with ?: (e.g. "(stuff)" -> "(?:stuff)").'

    return text


def flatten_multiple_choice_examples(inputs, targets):
    flat_idx = []
    flat_inputs = []
    flat_choices = []
    for example_id, (example_input, choices) in enumerate(zip(inputs,
                                                              targets)):
        for choice_id, choice in enumerate(choices):
            flat_idx.append((example_id, choice_id))
            flat_inputs.append(example_input)
            flat_choices.append(choice)

    return flat_idx, flat_inputs, flat_choices


def generate(
    model,
    tokenizer,
    context: str,
    max_length: int = 2046,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 0.9,
    num_outputs: int = 1,
    do_sample: bool = True,
) -> List[str]:
    """Generates outputs from language model.

    Args:
    context: input context
    max_length: maximum output length
    temperature: 1.0 has no effect, lower tend toward greedy sampling
    top_k: tokens to consider at each step as context
    top_p: nucleus sampling threshold
    num_outputs: number of outputs to generate

    Returns:
    list of generated responses

    Raises:
    ValueError if max_length is invalid
    """
    if max_length < 1:
        raise ValueError("max_length must be > 0")

    input_ids = tokenizer(context,
                          return_tensors="pt",
                          padding=True,
                          truncation=False)["input_ids"].to('cuda')

    if do_sample:
        output_sequences = model.generate(
            input_ids=input_ids,
            max_new_tokens=max_length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=1.0,
            do_sample=do_sample,
            num_return_sequences=num_outputs,
        )
    else:
        output_sequences = model.generate(
            input_ids=input_ids,
            max_new_tokens=max_length,
            temperature=0.0,
            do_sample=do_sample,
            num_return_sequences=num_outputs,
        )

    # Remove the batch dimension when returning multiple sequences
    if len(output_sequences.shape) > 2:
        output_sequences.squeeze_()

    generated_sequences = []

    for generated_sequence_idx, generated_sequence in enumerate(
            output_sequences):
        # Decode text
        text = tokenizer.decode(
            generated_sequence,
            clean_up_tokenization_spaces=False,
            skip_special_tokens=True,
        )

        # remove prompt
        text = text[len(context):]
        generated_sequences.append(text)

    return generated_sequences


def exact_str_match_fn(targets: List[str],
                       responses: List[str]) -> Dict[str, Any]:
    """Exact match between targets and responses."""
    # aitor: might want to use these metrics after removing irrelevant words.
    # they do something like this in see t5.data.postprocessors.wsc_simple
    count = 0

    for target, response in zip(targets, responses):
        if not isinstance(target, list):
            target = [target]
        if response in target:
            count += 1

    return {"exact_str_match": count / len(targets)}


def multiple_choice_grade(samples):
    """Scoring based on argmax of `log_prob`.

    Args:
      samples: List of dictionaries with 'target_scores' field.

    Returns:
      Average score on task.
    """
    count = 0

    for sample in samples:
        choice = sample["choice"][argmax(sample["log_prob"])]
        count += sample["target_scores"][choice]

    return count / len(samples)


def spelling_bee_relative_score_grade(samples):
    """Scoring based on argmax of `log_prob`.

    Args:
      samples: List of dictionaries with 'target_scores' field.

    Returns:
      Average score on task.
    """
    results = {}

    count = 0
    all_relative_scores = []
    all_recall = []
    all_num_incorrect = []
    all_precision = []
    for sample in samples:
        all_relative_scores.append(sample["score_data"]["score"] /
                                   sample["score_data"]["max_score"])
        all_recall.append(sample["score_data"]["num_correct"] /
                          sample["score_data"]["word_count"])
        all_num_incorrect.append(sample["score_data"]["num_incorrect"])

        # Precision -- defined to be 1 if no words are guessed.
        num_guesses = sample["score_data"]["num_correct"] + sample[
            "score_data"]["num_incorrect"]
        if num_guesses > 0:
            all_precision.append(sample["score_data"]["num_correct"] /
                                 num_guesses)
        else:
            all_precision.append(1.0)

    relative_score = np.mean(all_relative_scores)
    recall = np.mean(all_recall)
    num_incorrect = np.mean(all_num_incorrect)
    precision = np.mean(all_precision)

    results.update({
        "relative_score": relative_score,
        "recall": recall,
        "num_incorrect": num_incorrect,
        "precision": precision
    })

    return results


def measure_generative_exact_match(
        samples: List[Dict[str, Any]]) -> Dict[str, float]:
    """Measure of generative metrics."""

    results = {}

    responses = [sample["generative_response"] for sample in samples]
    targets = [sample["target"] for sample in samples]

    result = exact_str_match_fn(targets, responses)
    results.update(result)

    return results


def measure_multiple_choice(samples: List[Dict[str, Any]]) -> Dict[str, float]:
    """Measures multiple choice metrics from samples."""

    results = {}

    result = multiple_choice_grade(samples)
    results["multiple_choice_grade"] = result

    return results


def measure_spelling_bee(
        samples: List[Dict[str, Any]],
        multiple_choice_metrics: List[str]) -> Dict[str, float]:
    """Measures multiple choice metrics from samples."""

    results = {}

    result = spelling_bee_relative_score_grade(samples)
    results["relative_score"] = result

    return results


def _maybe_truncate_input(input_ids: np.ndarray,
                          max_len: int,
                          verbose: bool = True) -> np.ndarray:
    context_len = len(input_ids[0])

    if max_len <= context_len:
        if verbose:
            print(f"context len: {context_len}, max_model_len: {max_len}")
            print(f"input context will be truncated to length {max_len - 1}")
        return input_ids[:, -(max_len - 1):]

    return input_ids


####  we gonna use this function to replace the others.
def score_(model, tokenizer, inputs, targets):
    # print("input: ", inputs)
    # print("targets: ", targets)
    # exit(1)
    if isinstance(inputs, str):
        input_list = [inputs]
        target_list = [targets]
    else:
        input_list = inputs
        target_list = targets

    ### concatenate input and target and parse them.
    # concatenated_input = [inp+tar for inp, tar in zip(input_list, target_list)]

    input_ids = tokenizer.encode(input_list[0], truncation=False)
    output_ids = tokenizer.encode(target_list[0],
                                  truncation=False,
                                  add_special_tokens=False)
    labels = torch.tensor([-100] * (len(input_ids) - 1) + output_ids + [-100],
                          dtype=torch.int64).cuda()

    concatenated_input = torch.tensor([input_ids + output_ids],
                                      dtype=torch.long).cuda()
    # print(concatenated_input)

    # encoding = tokenizer(concatenated_input, padding="max_length", truncation=True)
    # tokenized = tokenizer.batch_encode_plus(concatenated_input, return_tensors="pt", padding=True, truncation=False)
    # print(tokenized)
    # exit(1)
    logits = model(input_ids=concatenated_input).logits
    # print(logits.shape)
    # print(logits.dtype)
    ### losses will be the log probablity of the whole sequence which we could use for comparison;
    #TODO: how to calculate the log probablity of the whole sequence?? We have padded and want to calcualte masked loss.
    # Below two lines are problematic. It gives -inf and nan values; ASK how ??
    # extended_mask = tokenized["attention_mask"].unsqueeze(2).expand_as(logits)
    # masked_logits = torch.mul(logits, extended_mask)
    # losses = torch.nn.CrossEntropyLoss(reduction='none')(masked_logits.permute(0, 2, 1), tokenized["input_ids"])
    losses = torch.nn.functional.cross_entropy(logits.view(
        -1, logits.shape[-1]),
                                               labels,
                                               reduction="mean")

    return [-losses.item()]


def generate_text(model, tokenizer, inputs, max_length, stop_string,
                  output_regex, batch_size):
    if isinstance(inputs, str):
        input_list = [inputs]
    else:
        input_list = inputs

    generated = []
    last = start = time.time()

    for idx in range(0, len(input_list), batch_size):
        start = time.time()
        inpt = input_list[idx:idx + batch_size]
        # print("inpt: ", inpt)
        model_inputs = tokenizer(inpt, return_tensors="pt", padding=True).to(model.device)
        # print("model_inputs: ", model_inputs)
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=max_length,
            temperature=1.0,
            top_k=1,
            top_p=0,
        )
        # print("model_inputs: ", model_inputs)
        # print("generated_ids: ", generated_ids)
        text = tokenizer.batch_decode(
            generated_ids[:, model_inputs["input_ids"].shape[1]:],
            skip_special_tokens=True)

        end = time.time()
        # print("Input {} finished, took {} seconds".format(idx, end - start))
        generated.extend(text)
        # print("output: ", text)
        # exit(1)

        if time.time() - last > 60:
            print(
                f"generated {idx} of {len(input_list)} outputs in {int(time.time() - start)} secs..."
            )
            last = time.time()

    if isinstance(inputs, str):
        generated = generated[0]

    generated = postprocess_output(generated, max_length, stop_string,
                                   output_regex)
    return generated


def cond_log_prob(
    model,
    tokenizer,
    inputs,
    targets,
    batch_size,
    absolute_normalization=False,
):
    """Computes conditional log probabilities of targets given inputs.

    Args:
      `inputs`: A single string input or a list of string inputs.

      `targets`: Possible string outputs for each input. If input is a
         string, this is a list `[t_1, t_2, ..., t_n]` of possible string
         outputs. If input is a list of strings, then this is a nested
         list `[[t_1, t_2, ..., t_n], ...]` with length equal to `len(inputs)`.

       `absolute_normalization`: When True, the function returns the log
         probability of unconstrained generation or the target sequence. When
         False (default), log probabilities are normalized so that the probabilities
         of generating `targets` sum to 1. Note that setting `absolute_normalization`
         to True restricts the class of models that can be evaluated to those that
         can assign absolute probabilities to sequences.

       Returns:
         If a single string input is provided, returns a list of
         log-probabilities `[lp_1, lp_2, ..., lp_n]` predicted by the model,
         where  `lp_i = log(prob(t_i | input)`  is the conditional log-prob
         to generate target `t_i` given input. If a list of string inputs
         was provided, returns a list of such elements of the form
         `[[lp_1, lp_2, ..., lp_n], ...]`, where each element contains the
         log-probabilities for the corresponding input and targets.
         In this case, the length of the returned list is `len(input)`.
    """

    if isinstance(inputs, str):
        input_list = [inputs]
        target_list = [targets]
    else:
        input_list = inputs
        target_list = targets

    flat_idx, flat_inputs, flat_choices = flatten_multiple_choice_examples(
        inputs=input_list, targets=target_list)
    num_examples = len(flat_idx)
    flat_scores = []
    for idx in range(0, num_examples, batch_size):
        batch_idx = flat_idx[idx:min(idx + batch_size, num_examples)]
        batch_inputs = flat_inputs[idx:min(idx + batch_size, num_examples)]
        batch_choices = flat_choices[idx:min(idx + batch_size, num_examples)]

        start = time.time()
        batch_scores = score_(
            model, tokenizer, batch_inputs, batch_choices
        )  ### TODO: we need to implement this score function!!!!
        end = time.time()
        print("Batch {} finished, took {} seconds".format(idx, end - start))
        flat_scores += batch_scores

    scores = [[] for _ in range(len(input_list))]

    for idx, score in zip(flat_idx, flat_scores):
        if score == 0:
            # all tokens were masked. Setting score to -inf.
            logging.warning(
                'Found score identical to zero. Probably from empty target. '
                'Setting score to -inf.')
            scores[idx[0]].append(-np.inf)
        else:
            scores[idx[0]].append(score)

    if not absolute_normalization:
        scores = [
            list(score_row - scipy.special.logsumexp(score_row))
            for score_row in scores
        ]

    if isinstance(inputs, str):
        scores = scores[0]

    return scores


def construct_reponse(model,
                      tokenizer,
                      samples,
                      metric_type,
                      batch_size=8,
                      absolute_normalization=False,
                      num_rounds=20):
    inputs = [sample["input"] for sample in samples]

    if metric_type == "multiple_choice":
        targets = [sample["choice"] for sample in samples]
        print(
            f"Example of targets for multiple choice evaluation: {','.join(targets[0])}"
        )
        absolute_log_probs = cond_log_prob(model,
                                           tokenizer,
                                           inputs=inputs,
                                           targets=targets,
                                           batch_size=batch_size,
                                           absolute_normalization=True)
        if len(absolute_log_probs[0]) == 1 and len(inputs) == 1:
            absolute_log_probs = [absolute_log_probs]

        normalized_log_probs = [
            list(np.array(x) - scipy.special.logsumexp(x))
            for x in absolute_log_probs
        ]

        if absolute_normalization:
            log_probs = absolute_log_probs
        else:
            log_probs = normalized_log_probs

    if metric_type == "generation":
        generative_responses = generate_text(model,
                                             tokenizer,
                                             inputs=inputs,
                                             stop_string=None,
                                             output_regex=r"[a-zA-Z]+",
                                             max_length=20,
                                             batch_size=batch_size)
        if isinstance(generative_responses, str) and len(inputs) == 1:
            generative_responses = [generative_responses]

    if metric_type == "spelling_bee":
        game_scores = []
        for input in tqdm(inputs):
            game = SpellingBeeGame(game_data=input, num_rounds=num_rounds)
            game_scores.append(game.play(model=model, tokenizer=tokenizer))

            # pdb.set_trace()
        # score_data = _aggregate_scores(game_scores)

    for i, sample in enumerate(samples):
        if metric_type == "multiple_choice":
            sample["absolute_log_prob"] = absolute_log_probs[i]
            sample["normalized_log_prob"] = normalized_log_probs[i]
            sample["log_prob"] = log_probs[i]
        if metric_type == "generation":
            sample["generative_response"] = generative_responses[i]
        # if self.scoring_metrics and generate_probs_for_targets:
        #     sample["log_prob_gen"] = log_probs_gen[i]
        if metric_type == "spelling_bee":
            sample["score_data"] = game_scores[i]

    return samples


def evaluate_metrics(samples, metric_type):

    results = {}
    if metric_type == "multiple_choice":
        result = measure_multiple_choice(samples)
        results.update(result)

    if metric_type == "generation":
        result = measure_generative_exact_match(samples)
        results.update(result)

    if metric_type == "spelling_bee":
        result = measure_spelling_bee(samples)
        results.update(result)

    return results
