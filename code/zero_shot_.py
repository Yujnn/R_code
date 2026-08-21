"""
Zero-shot LLM-based rhetorical sarcasm classification evaluation.

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
export MODEL_NAME="your_model_name"

"""


import os
import json
import time
import argparse
import random
import requests

from typing import Dict, List
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
        "Surface meaning opposite to intended meaning.",


    "Echoic Mention":
        "Mocking repetition or reference to a previous statement.",


    "Hyperbole":
        "Extreme exaggeration beyond reality.",


    "Rhetorical Question":
        "A question implying an assertion or attitude.",


    "Self-Deprecation":
        "Self-directed mockery or criticism.",


    "Presupposition":
        "An implicit assumption treated as shared knowledge.",


    "Innuendo":
        "Indirect expression of negative meaning.",


    "Intentional Reenactment":
        "Exaggerated replay of a situation or event.",


    "Unexpected Twist":
        "Sudden contrast or reversal in interpretation."

}



DEVICE_LOOKUP = {

    k.lower(): k

    for k in RHETORICAL_DEVICES

}



# ============================================================
# Zero-shot Prompt
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

Your task is to identify the rhetorical device(s)
responsible for constructing the sarcastic meaning.

A sarcastic utterance may involve multiple rhetorical mechanisms.
Select one or more labels from the predefined taxonomy.


Rhetorical Labels:

{labels}


Dialogue:

{dialogue}


Sarcastic segment:

{segment}


Output ONLY the rhetorical label(s), separated by commas.
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
            "Please set ARK_API_KEY environment variable."
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
# Prediction normalization
# ============================================================


def normalize_prediction(
        text: str
) -> List[str]:


    text = text.lower()


    results = []


    # Match long labels first
    for key, value in sorted(

            DEVICE_LOOKUP.items(),

            key=lambda x: len(x[0]),

            reverse=True

    ):


        if key in text:

            results.append(value)



    return list(dict.fromkeys(results))



# ============================================================
# Evaluation metrics
# ============================================================


def evaluate_prediction(

        prediction: List[str],

        gold: List[str],

        mode: str

):


    prediction = [

        x.lower()

        for x in prediction

    ]


    gold = [

        x.lower()

        for x in gold

    ]



    gold_set = set(gold)



    # -----------------------------
    # Top-1
    # -----------------------------

    if mode == "top1":


        if len(prediction) == 0:

            return False


        return prediction[0] in gold_set



    # -----------------------------
    # Top-2
    # -----------------------------

    elif mode == "top2":


        prediction = prediction[:2]


        return (

            len(

                set(prediction)

                &
                gold_set

            )

            > 0

        )



    # -----------------------------
    # Any-Match
    # -----------------------------

    elif mode == "any-match":


        return (

            len(

                set(prediction)

                &
                gold_set

            )

            > 0

        )


    else:

        raise ValueError(
            "Unknown evaluation mode."
        )



# ============================================================
# Main evaluation
# ============================================================


def evaluate(

        dataset_path: str,

        mode: str = "any-match",

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



        # category-level statistics

        pred_set = {

            x.lower()

            for x in prediction

        }


        for label in gold_labels:


            key = label.lower()


            category_total[key] += 1


            if key in pred_set:

                category_correct[key] += 1



        time.sleep(1)



    print("\n========== Results ==========")


    accuracy = (

        correct /

        len(data)

    )


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
# Entry
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