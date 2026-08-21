"""
Few-shot LLM-based rhetorical sarcasm classification evaluation.

Evaluation protocols:
    - top1
    - top2
    - any-match


Environment variables:

    ARK_API_KEY
    MODEL_NAME
    ARK_API_URL(optional)


Example:

export ARK_API_KEY="your_api_key"
export MODEL_NAME="your_model"

"""


import os
import json
import time
import argparse
import random
import requests


from typing import List, Dict
from collections import defaultdict



# ============================================================
# Rhetorical taxonomy
# ============================================================


RHETORICAL_DEVICES = [

    "Irony",

    "Echoic Mention",

    "Hyperbole",

    "Rhetorical Question",

    "Self-Deprecation",

    "Presupposition",

    "Innuendo",

    "Intentional Reenactment",

    "Unexpected Twist"

]



DEVICE_DEFINITION = {


    "Irony":
        "Surface meaning opposite to intent.",


    "Echoic Mention":
        "Mocking repetition of a prior utterance.",


    "Hyperbole":
        "Extreme exaggeration.",


    "Rhetorical Question":
        "A question implying an assertion.",


    "Self-Deprecation":
        "Self-mockery targeting oneself.",


    "Presupposition":
        "Implicit shared assumption.",


    "Innuendo":
        "Indirect negative implication.",


    "Intentional Reenactment":
        "Exaggerated replay of events.",


    "Unexpected Twist":
        "Sudden narrative contrast or reversal."

}



DEVICE_LOOKUP = {

    k.lower(): k

    for k in RHETORICAL_DEVICES

}



# ============================================================
# Few-shot Prompt
# ============================================================


def build_prompt(item: Dict) -> str:


    dialogue = "\n".join(
        item["sarcastic dialogue"]
    )


    segment = item["sarcastic segment"]



    labels = "\n".join(

        [

            f"- {k}: {v}"

            for k, v in DEVICE_DEFINITION.items()

        ]

    )



    prompt = f"""

You are an expert in rhetorical sarcasm understanding.

Given a sarcastic dialogue, identify the rhetorical device(s)
responsible for constructing the sarcastic meaning.

Multiple rhetorical mechanisms may co-exist.
Output one or more labels from the predefined taxonomy.



Rhetorical Labels:

{labels}



Example 1:

Context:
"What are you, a little kid? Is she gonna cut your dinner
into little pieces, too?"

Label:
Rhetorical Question, Hyperbole



Example 2:

Context:
"I'm gonna miss her."
(after Raj breaks Emily's drawer)

Label:
Presupposition, Innuendo



Example 3:

Context:
"You could barely survive a tiny turtle bite."

Label:
Intentional Reenactment



Now classify the following dialogue.



Dialogue:

{dialogue}



Sarcastic segment:

{segment}



Output ONLY the rhetorical labels separated by commas.
No explanation.

"""


    return prompt.strip()



# ============================================================
# LLM API
# ============================================================


def call_llm(
        prompt: str,
        timeout: int = 60,
        retries: int = 3
):


    api_url = os.getenv(

        "ARK_API_URL",

        "https://ark.cn-beijing.volces.com/api/v3/chat/completions"

    )


    api_key = os.getenv(
        "ARK_API_KEY"
    )


    model_name = os.getenv(

        "MODEL_NAME",

        "your-model-name"

    )


    if api_key is None:

        raise ValueError(

            "Missing ARK_API_KEY environment variable."

        )



    headers = {


        "Content-Type":

            "application/json",


        "Authorization":

            f"Bearer {api_key}"

    }



    payload = {


        "model":

            model_name,


        "messages":

        [

            {

                "role":

                    "user",


                "content":

                    prompt

            }

        ],


        "temperature":

            0.01,


        "max_tokens":

            32

    }




    for _ in range(retries):


        try:


            response = requests.post(

                api_url,

                headers=headers,

                json=payload,

                timeout=timeout

            )


            response.raise_for_status()



            return (

                response.json()

                ["choices"][0]

                ["message"]

                ["content"]

                .strip()

            )



        except Exception:


            time.sleep(2)




    return ""



# ============================================================
# Label normalization
# ============================================================


def normalize_prediction(
        text: str
) -> List[str]:


    text = text.lower()


    predictions = []



    # longest matching first
    for key, value in sorted(

            DEVICE_LOOKUP.items(),

            key=lambda x: len(x[0]),

            reverse=True

    ):


        if key in text:


            predictions.append(value)



    return predictions



# ============================================================
# Evaluation protocol
# ============================================================


def evaluate_prediction(

        prediction: List[str],

        gold: List[str],

        mode: str

):


    prediction_lower = [

        x.lower()

        for x in prediction

    ]


    gold_lower = [

        x.lower()

        for x in gold

    ]



    gold_set = set(gold_lower)



    # Top-1

    if mode == "top1":


        if len(prediction_lower) == 0:

            return False


        return prediction_lower[0] in gold_set



    # Top-2

    elif mode == "top2":


        return (

            len(

                set(prediction_lower[:2])

                &
                gold_set

            )

            > 0

        )



    # Any-match

    elif mode == "any-match":


        return (

            len(

                set(prediction_lower)

                &
                gold_set

            )

            > 0

        )



    else:

        raise ValueError(
            "Unsupported evaluation mode."
        )



# ============================================================
# Evaluation
# ============================================================


def evaluate(

        dataset_path: str,

        mode: str,

        sample_num=None

):


    with open(

            dataset_path,

            "r",

            encoding="utf-8"

    ) as f:


        data = json.load(f)




    if sample_num:


        data = random.sample(

            data,

            min(sample_num, len(data))

        )




    correct = 0



    category_total = defaultdict(int)

    category_correct = defaultdict(int)



    for idx, item in enumerate(data):


        print(

            f"Processing {idx+1}/{len(data)}"

        )



        prompt = build_prompt(item)



        raw_prediction = call_llm(prompt)



        prediction = normalize_prediction(

            raw_prediction

        )



        gold = item["rhetorical device"]



        if isinstance(gold, list):

            gold_labels = gold

        else:

            gold_labels = [gold]




        if evaluate_prediction(

                prediction,

                gold_labels,

                mode

        ):


            correct += 1




        prediction_set = {


            x.lower()

            for x in prediction

        }



        for label in gold_labels:


            key = label.lower()


            category_total[key] += 1



            if key in prediction_set:


                category_correct[key] += 1




        time.sleep(1)





    print("\n========== Results ==========")



    accuracy = correct / len(data)



    print(

        f"Evaluation Mode: {mode}"

    )


    print(

        f"Accuracy: {accuracy:.4f}"

    )



    print("\nCategory Accuracy:")



    for label in RHETORICAL_DEVICES:


        key = label.lower()


        if category_total[key] > 0:


            score = (

                category_correct[key]

                /

                category_total[key]

            )


            print(

                f"{label}: {score:.4f}"

            )



# ============================================================
# Main
# ============================================================


if __name__ == "__main__":


    parser = argparse.ArgumentParser()



    parser.add_argument(

        "--dataset",

        default="RedSD.json"

    )


    parser.add_argument(

        "--mode",

        default="any-match",

        choices=[

            "top1",

            "top2",

            "any-match"

        ]

    )


    parser.add_argument(

        "--sample_num",

        type=int,

        default=None

    )



    args = parser.parse_args()



    evaluate(

        dataset_path=args.dataset,

        mode=args.mode,

        sample_num=args.sample_num

    )