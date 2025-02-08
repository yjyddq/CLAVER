# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from typing import List, Optional

import fire

from llama import Llama

import pandas as pd
import csv


def main(
        ckpt_dir: str,
        tokenizer_path: str,
        temperature: float = 0.9,
        top_p: float = 0.95,
        max_seq_len: int = 128,
        max_batch_size: int = 8,
        max_gen_len: Optional[int] = 77,
):
    """
    Entry point of the program for generating text using a pretrained model.

    Args:
        ckpt_dir (str): The directory containing checkpoint files for the pretrained model.
        tokenizer_path (str): The path to the tokenizer model used for text encoding/decoding.
        temperature (float, optional): The temperature value for controlling randomness in generation.
            Defaults to 0.6.
        top_p (float, optional): The top-p sampling parameter for controlling diversity in generation.
            Defaults to 0.9.
        max_seq_len (int, optional): The maximum sequence length for input prompts. Defaults to 512.
        max_batch_size (int, optional): The maximum batch size for generating sequences. Defaults to 8.
        max_gen_len (int, optional): The maximum length of generated sequences. If None, it will be
            set to the model's max sequence length. Defaults to None.
    """
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )
    label_path = '/PATH/TO/labels.csv'
    label_list = pd.read_csv(label_path)
    actions = label_list['name']
    repeat = 5
    descriptions_action_decomposition = []
    descriptions_body = []
    descriptions_synonyms = []
    descriptions_scene = []

    for i in range(len(actions)):
        for j in range(repeat):
            prompts_action_decomposition: List[str] = [
                # Few shot prompt (providing a few examples before asking model to complete more);
                """Generate a simple and detailed description (explanation) about the action decomposition of a human action video (Please ensure that the length of description for each action class does not exceed 77 words),
            
                abseiling => Abseiling combines several actions to descend a vertical surface with a rope. Climbers secure themselves with a harness and utilize a descender device for controlled descent. Simple actions, like maintaining a straight body position and regulating rope tension, form the basis. Abseiling demands proper training, safety measures, and is popular in adventure sports and rescue operations, allowing individuals to experience controlled descent in various settings.
            
                air drumming => Air drumming is a rhythmic expression where individuals simulate playing drums without physical instruments. Simple actions, like mimicking drumming motions in the air, combine to create this imaginative and playful activity. Enthusiasts use their hands and feet to imitate drumming patterns, syncing with music. It's a spontaneous, enjoyable gesture often done during music listening or live performances, showcasing one's connection to the rhythm without the need for actual drums or drumsticks.
            
                answering questions => Answering questions involves providing responses to queries posed by others. Simple actions like active listening, comprehension, and concise articulation combine in this communicative process. It is fundamental in various contexts, facilitating information exchange and problem-solving. Respondents draw on their knowledge and expertise to address inquiries, contributing to effective communication and fostering understanding between individuals or groups.
             
                {} =>""".format(actions[i])
            ]
            prompts_body: List[str] = [
                # Few shot prompt (providing a few examples before asking model to complete more);
                """Generate text descriptions to describe an action video based on its actions nouns and possible body parts involved. (Please ensure that your response does not exceed 77 words),

                Cutting in the kitchen => Using a sharp knife, fingers gripping the handle, hand guiding the blade through ingredients on a cutting board, wrist controlling the motion, fingers curling slightly to hold the food steady, precision applied to achieve desired shapes or sizes, ensuring safety and efficiency during food preparation.

                driving car => Gripping the steering wheel, hands adjusting position, fingers pressing pedals for acceleration and braking, eyes scanning surroundings for obstacles, feet coordinating between clutch, brake, and accelerator, body positioned comfortably in the driver's seat, mind focused on navigation and traffic signals, reacting swiftly to changing road conditions.

                Walking With Dog => Leash in hand, fingers securing grip, arm relaxed as it swings alongside the body, legs moving in tandem with the dog's pace, feet stepping forward with purpose, eyes attentive to the dog's behavior and surroundings, occasional stops for sniffing or marking, a bond of companionship evident in synchronized movement.

                {} =>""".format(actions[i])
            ]
            prompts_synonyms: List[str] = [
                # Few shot prompt (providing a few examples before asking model to complete more);
                """Please generate synonyms or synonymous phrases about action nouns or phrases, they share the same central concept but have more diverse expressions. (Please ensure that your response does not exceed 77 words),

                Cutting in the kitchen => Slicing, dicing, chopping, mincing, cleaving, carving, trimming, preparing ingredients.

                driving car => Operating a vehicle, maneuvering behind the wheel, navigating the road, piloting an automobile, steering, cruising, commuting by car, motoring.

                Walking With Dog => Strolling with a canine companion, ambling with a pet, promenading with a pup, hiking with a furry friend, sauntering alongside a dog, wandering with a four-legged buddy, leash-walking, trotting with a pooch.

                {} =>""".format(actions[i])
            ]
            prompts_scene: List[str] = [
                # Few shot prompt (providing a few examples before asking model to complete more);
                """Please generate some descriptions based on the content of action and the possible scene in which it may occur. (Please ensure that your response does not exceed 77 words),

                Cutting in the kitchen => In the bustling kitchen, amidst the rhythmic clang of utensils and the sizzle of pans, a chef deftly wields a knife, cutting through fresh ingredients with precision. The air fills with the aroma of herbs and spices as vegetables yield to the blade's edge. Amidst the organized chaos, the chef orchestrates culinary mastery, transforming raw components into culinary delights, each slice a testament to skill and finesse in the art of cooking.

                driving car => Behind the wheel, the driver embarks on a journey along winding roads and urban thoroughfares. The rhythmic purr of the engine accompanies their focused gaze ahead, navigating the ever-changing landscape of traffic lights and bustling intersections. With each curve and straightaway, they maintain a steady pace, their hands deftly maneuvering the wheel, guiding the vehicle towards its destination with confidence and purpose.

                Walking With Dog => Amidst the serenity of a neighborhood park, a person strides alongside their loyal companion, a dog by their side. The leash stretches and relaxes as the duo explores nature's tapestry, leaves crunching beneath their feet. The dog's tail wags in rhythm with each step, its eager gaze darting from tree to tree, nose twitching with curiosity. Together, they embark on a shared adventure, forging a bond of companionship amidst the tranquility of the great outdoors.

                {} =>""".format(actions[i])
            ]

            results_action_decomposition = generator.text_completion(
                prompts_action_decomposition,
                max_gen_len=max_gen_len,
                temperature=temperature,
                top_p=top_p,
            )
            for prompt, result in zip(prompts_action_decomposition, results_action_decomposition):
                print(prompt)
                print(f"> {result['generation']}")
                print("\n==================================\n")
            descriptions_action_decomposition.append(results_action_decomposition[0]['generation'].strip(' '))

            results_body = generator.text_completion(
                prompts_body,
                max_gen_len=max_gen_len,
                temperature=temperature,
                top_p=top_p,
            )
            for prompt, result in zip(prompts_body, results_body):
                print(prompt)
                print(f"> {result['generation']}")
                print("\n==================================\n")
            descriptions_body.append(results_body[0]['generation'].strip(' '))

            results_synonyms = generator.text_completion(
                prompts_synonyms,
                max_gen_len=max_gen_len,
                temperature=temperature,
                top_p=top_p,
            )
            for prompt, result in zip(prompts_synonyms, results_synonyms):
                print(prompt)
                print(f"> {result['generation']}")
                print("\n==================================\n")
            descriptions_synonyms.append(results_synonyms[0]['generation'].strip(' '))

            results_scene = generator.text_completion(
                prompts_scene,
                max_gen_len=max_gen_len,
                temperature=temperature,
                top_p=top_p,
            )
            for prompt, result in zip(prompts_scene, results_scene):
                print(prompt)
                print(f"> {result['generation']}")
                print("\n==================================\n")
            descriptions_scene.append(results_scene[0]['generation'].strip(' '))

    data_action_decomposition = [[i // repeat, content] for i, content in enumerate(descriptions_action_decomposition)]
    df_action_decomposition = pd.DataFrame(data_action_decomposition, columns=['id', 'name'])
    df_action_decomposition.to_csv('./action_decomposition.csv', index=False, quoting=csv.QUOTE_NONE, escapechar='\\')

    data_body = [[i // repeat, content] for i, content in enumerate(descriptions_body)]
    df_body = pd.DataFrame(data_body, columns=['id', 'name'])
    df_body.to_csv('./body.csv', index=False, quoting=csv.QUOTE_NONE, escapechar='\\')

    data_synonyms = [[i // repeat, content] for i, content in enumerate(descriptions_synonyms)]
    df_synonyms = pd.DataFrame(data_synonyms, columns=['id', 'name'])
    df_synonyms.to_csv('./synonyms.csv', index=False, quoting=csv.QUOTE_NONE, escapechar='\\')
    
    data_scene = [[i// repeat, content] for i, content in enumerate(descriptions_scene)]
    df_scene = pd.DataFrame(data_scene, columns=['id', 'name'])
    df_scene.to_csv('./scene.csv', index=False, quoting=csv.QUOTE_NONE, escapechar='\\')


if __name__ == "__main__":
    fire.Fire(main)
