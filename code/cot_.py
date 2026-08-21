"""
Chain-of-Thought LLM-based rhetorical sarcasm classification evaluation.

This script evaluates CoT prompting on RedSD.

The model is instructed to:
1. interpret literal meaning;
2. analyze contextual assumptions;
3. identify rhetorical mechanisms;
4. output rhetorical labels.

Evaluation protocols:
    - top1
    - top2
    - any-match


Environment variables:

ARK_API_KEY
MODEL_NAME
ARK_API_URL(optional)

"""


import os
import json
import time
import argparse


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
# CoT Prompt Construction
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

You are a rhetorical analysis system.

Your task is to identify the rhetorical device(s)
that construct the sarcastic meaning of an utterance.



Please reason step by step:

1. Interpret the literal meaning of the utterance.

2. Consider contextual assumptions,
speaker intention, and pragmatic expectations.

3. Determine which rhetorical mechanism(s)
explain the discrepancy between surface expression
and intended meaning.

4. Output one or more rhetorical labels from
the predefined taxonomy.



Rhetorical Labels:

{labels}



Dialogue:

{dialogue}



Sarcastic segment:

{segment}



First provide your reasoning process,
then output the final rhetorical labels.

The final output format should be:

Labels:
[label1, label2]

"""


    return prompt.strip()



# ============================================================
# API call
# ============================================================


def call_llm(

        prompt: str,

        timeout=60,

        retries=3

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

            "Missing ARK_API_KEY."

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

            512

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

            )



        except Exception:


            time.sleep(2)



    return ""




# ============================================================
# Extract labels from CoT output
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



    return list(dict.fromkeys(predictions))




# ============================================================
# Evaluation
# ============================================================


def evaluate_prediction(

        prediction,

        gold,

        mode

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



    if mode == "top1":


        return (

            len(prediction)>0

            and

            prediction[0] in gold_set

        )


    elif mode == "top2":


        return bool(

            set(prediction[:2])

            &

            gold_set

        )


    elif mode == "any-match":


        return bool(

            set(prediction)

            &

            gold_set

        )


    else:


        raise ValueError(

            "Unknown evaluation mode"

        )





# ============================================================
# Evaluation Pipeline
# ============================================================


def evaluate(

        dataset_path,

        mode="any-match"

):


    with open(

        dataset_path,

        "r",

        encoding="utf-8"

    ) as f:


        data = json.load(f)




    correct = 0



    category_total = defaultdict(int)

    category_correct = defaultdict(int)




    for idx,item in enumerate(data):


        print(

            f"Processing {idx+1}/{len(data)}"

        )



        prompt = build_prompt(item)



        response = call_llm(prompt)



        prediction = normalize_prediction(

            response

        )



        gold = item["rhetorical device"]



        if isinstance(gold,list):


            gold_labels = gold


        else:


            gold_labels = [gold]




        if evaluate_prediction(

            prediction,

            gold_labels,

            mode

        ):


            correct += 1




        pred_set = {


            x.lower()

            for x in prediction

        }



        for label in gold_labels:


            key = label.lower()


            category_total[key]+=1



            if key in pred_set:


                category_correct[key]+=1




        time.sleep(1)





    print("\n========== Results ==========")



    print(

        f"Mode: {mode}"

    )


    print(

        f"Accuracy: {correct/len(data):.4f}"

    )



    print("\nCategory Accuracy:")



    for label in RHETORICAL_DEVICES:


        key = label.lower()


        if category_total[key]:


            print(

                f"{label}: "

                f"{category_correct[key]/category_total[key]:.4f}"

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



    args = parser.parse_args()



    evaluate(

        args.dataset,

        args.mode

    )