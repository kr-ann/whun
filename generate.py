#!/usr/bin/env python3
# coding=utf-8
import argparse
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import logging

logging.getLogger("transformers").setLevel(logging.CRITICAL)
MAX_LENGTH = int(10000)


def adjust_length_to_model(length, max_sequence_length):
    if length < 0 and max_sequence_length > 0:
        length = max_sequence_length
    elif 0 < max_sequence_length < length:
        length = max_sequence_length
    elif length < 0:
        length = MAX_LENGTH
    return length

# generation parameters
def generate(prompt, model_name_or_path, length=256):
    stop_token = "<end>"

    temperature = 1.0
    repetition_penalty = 1.0
    num_beams = 0
    do_sample = False
    k = 50
    p = 0.95
    prefix = ""
    num_return_sequences = 1

    # Initialize the model and tokenizer
    model_class, tokenizer_class = (GPT2LMHeadModel, GPT2Tokenizer)

    tokenizer = tokenizer_class.from_pretrained(model_name_or_path)
    model = model_class.from_pretrained(model_name_or_path)

    length = adjust_length_to_model(length,
                                    max_sequence_length= \
                                    model.config.max_position_embeddings)

    prompt_text = prompt if prompt else input("Model prompt >>> ")

    prefix = prefix
    encoded_prompt = tokenizer.encode(prefix + prompt_text,
                                      add_special_tokens=False,
                                      return_tensors="pt")

    if encoded_prompt.size()[-1] == 0:
        input_ids = None
    else:
        input_ids = encoded_prompt

    output_sequences = model.generate(
        input_ids=input_ids,
        max_length=length + len(encoded_prompt[0]),
        temperature=temperature,
        top_k=k,
        top_p=p,
        repetition_penalty=repetition_penalty,
        do_sample=True,
        num_return_sequences=num_return_sequences,
    )

    # Remove the batch dimension when returning multiple sequences
    if len(output_sequences.shape) > 2:
        output_sequences.squeeze_()

    generated_sequences = []

    for generated_sequence_idx, generated_sequence in \
        enumerate(output_sequences):
        generated_sequence = generated_sequence.tolist()

        text = tokenizer.decode(generated_sequence,
                                clean_up_tokenization_spaces=True)

        # Remove all text after the stop token
        text = text[: text.find(stop_token)]

        # Add the prompt at the beginning of the sequence.
        # Remove the excess text that was used for pre-processing
        total_sequence = (
                prompt_text + " " + text[len(tokenizer.decode(
                    encoded_prompt[0], clean_up_tokenization_spaces=True)):]
        )

        generated_sequences.append(total_sequence)

    return " ".join(generated_sequences)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, help='Which model to use. \
    Possible options: 1, 2, 3, 4, 5, 6, or "gpt2" for the base model.')
    args = parser.parse_args()

    if args.model == '1':
        model_name = "./first_run/model"
    elif args.model == '2':
        model_name = "./first_run/model_2"
    elif args.model == '3':
        model_name = "./first_run/model_3"
    elif args.model == '4':
        model_name = "./model_4"
    elif args.model == '5':
        model_name = "./model_5"
    elif args.model == '6':
        model_name = "./model_6"
    elif args.model == 'gpt2':
        model_name = "gpt2"
    else:
        raise ValueError(f"Incorrect model path or name: {args.model}. See help for possible options.")


    prompt = input("Enter start of the joke:\n")
    print(generate(prompt, model_name))
