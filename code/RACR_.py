"""
RACR:
Rhetoric-Aware Counterfactual Reasoning

Inference pipeline:

utterance x
      |
      v
counterfactual generation
      |
      v
stage-I conflict ranking
      |
      v
stage-II conflict attribution
      |
      v
rule-guided calibration
      |
      v
structured reasoning signal z
      |
      v
LLM prediction


The implementation follows the structured reasoning
signal format described in the paper.
"""


import os
import json
import time
import argparse
import requests
import numpy as np


from typing import Dict, List





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





# ============================================================
# Linguistic rules
#
# Rule-guided calibration
# S_rule(r)=N_activated/N_total
# ============================================================


LINGUISTIC_RULES = {


"Irony":[

    "literal meaning conflicts with intended meaning",

    "positive wording expressing negative attitude",

    "sarcastic praise or exaggerated approval",

    "polarity reversal between statement and context"

],



"Echoic Mention":[

    "quotation or repetition of previous words",

    "mocking another person's expression",

    "reference to prior statement",

    "echoing unacceptable opinion"

],



"Hyperbole":[

    "extreme quantity or degree expression",

    "exaggerated adjective or adverb",

    "impossible or unrealistic description",

    "overstatement beyond factual reality"

],



"Rhetorical Question":[

    "question form without expectation of answer",

    "question implying criticism",

    "question used to express attitude",

    "obvious answer assumed"

],



"Self-Deprecation":[

    "negative expression targeting speaker",

    "self-directed criticism",

    "self-mocking statement",

    "humorous lowering of self-image"

],



"Presupposition":[

    "implicit assumption",

    "background information taken as true",

    "trigger words such as again, still, already",

    "unstated shared knowledge"

],



"Innuendo":[

    "indirect negative implication",

    "hidden criticism",

    "ambiguous insulting meaning",

    "meaning beyond literal expression"

],



"Intentional Reenactment":[

    "imitation of previous event",

    "exaggerated replay",

    "dramatic reconstruction",

    "performed repetition of behavior"

],



"Unexpected Twist":[

    "sudden contrast",

    "unexpected conclusion",

    "narrative reversal",

    "violation of expectation"

]


}






# ============================================================
# API configuration
# ============================================================


def call_llm(prompt:str):


    api_url = os.getenv(

        "ARK_API_URL"

    )


    api_key = os.getenv(

        "ARK_API_KEY"

    )


    model_name = os.getenv(

        "MODEL_NAME"

    )



    if not api_key:

        raise ValueError(

            "Missing ARK_API_KEY"

        )



    headers={

        "Content-Type":

        "application/json",


        "Authorization":

        f"Bearer {api_key}"

    }




    payload={


        "model":

        model_name,


        "messages":[

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

        2048

    }



    response=requests.post(

        api_url,

        headers=headers,

        json=payload,

        timeout=60

    )


    response.raise_for_status()


    return (

        response.json()

        ["choices"][0]

        ["message"]

        ["content"]

        .strip()

    )






# ============================================================
# Counterfactual Generation
#
# Produce:
# x_r_lit
# x_r_int
# x_r_ctx
# ============================================================



def generate_counterfactual(

        utterance:str,

        rhetoric:str

):


    prompt=f"""

You are a rhetorical reasoning expert.


Original utterance:

{utterance}


Candidate rhetorical mechanism:

{rhetoric}



Generate three controlled counterfactual views.


1. literal_counterfactual:
Remove the rhetorical mechanism while preserving factual content.


2. intent_neutralized_counterfactual:
Remove sarcastic intention while preserving context.


3. context_adjusted_counterfactual:
Modify contextual assumptions related to this rhetorical mechanism.



Return JSON only:


{{
"literal_counterfactual":"",
"intent_neutralized_counterfactual":"",
"context_adjusted_counterfactual":""
}}

"""


    output=call_llm(prompt)



    try:

        return json.loads(output)


    except:


        return {


        "literal_counterfactual":"",

        "intent_neutralized_counterfactual":"",

        "context_adjusted_counterfactual":""


        }
    # ============================================================
# Conflict Estimation
#
# Estimate how much rhetorical meaning disappears
# after counterfactual intervention.
#
# Conflict_r(x)
# ============================================================


def estimate_conflict(

        original:str,

        counterfactual:str

):


    prompt=f"""

You are analyzing rhetorical conflict.


Original utterance:

{original}



Counterfactual:

{counterfactual}



Estimate whether the rhetorical mechanism disappears
after this modification.


Output only one score:

0.0 = no rhetorical change

1.0 = rhetorical mechanism completely removed


Score:

"""


    result=call_llm(prompt)



    try:

        score=float(result)

        score=max(

            0.0,

            min(

                1.0,

                score

            )

        )


    except:


        score=0.0



    return score







# ============================================================
# Stage-I:
# Conflict Ranking
#
# candidate_scores:
#
# rhetorical_type
# conflict_score
# ranking_score=P(r|x)
#
# ============================================================


def stage1_conflict_ranking(

        utterance:str

):


    candidate_scores=[]


    counterfactual_evidence=[]



    for rhetoric in RHETORICAL_DEVICES:



        # -----------------------------------
        # generate three counterfactual views
        # -----------------------------------


        cf = generate_counterfactual(

            utterance,

            rhetoric

        )



        conflict_values=[]



        for key in [

            "literal_counterfactual",

            "intent_neutralized_counterfactual",

            "context_adjusted_counterfactual"

        ]:


            score=estimate_conflict(

                utterance,

                cf[key]

            )


            conflict_values.append(score)



        # average three intervention effects


        conflict_score=float(

            np.mean(

                conflict_values

            )

        )



        candidate_scores.append(

            {


            "rhetorical_type":

                rhetoric,


            "conflict_score":

                conflict_score


            }

        )



        counterfactual_evidence.append(

            {


            "candidate_rhetoric":

                rhetoric,


            "literal_counterfactual":

                cf[

                "literal_counterfactual"

                ],


            "intent_neutralized_counterfactual":

                cf[

                "intent_neutralized_counterfactual"

                ],


            "context_adjusted_counterfactual":

                cf[

                "context_adjusted_counterfactual"

                ]

            }

        )




    # ---------------------------------------
    # Convert conflict score into probability
    #
    # P(r|x)
    # ---------------------------------------


    scores=np.array(

        [

        item["conflict_score"]

        for item in candidate_scores

        ]

    )



    exp_scores=np.exp(scores)



    probs=(

        exp_scores /

        exp_scores.sum()

    )



    for item,p in zip(

        candidate_scores,

        probs

    ):


        item[

            "ranking_score"

        ]=float(p)



    # ranking


    candidate_scores.sort(

        key=lambda x:

        x["ranking_score"],

        reverse=True

    )



    top_candidates=[

        x["rhetorical_type"]

        for x in candidate_scores[:3]

    ]



    return {


        "candidate_scores":

            candidate_scores,


        "top_candidates":

            top_candidates

    }, counterfactual_evidence







# ============================================================
# Stage-II:
# Conflict Attribution
#
# Delta_k^r
# S_attr(r)
#
# ============================================================


def conflict_attribution(

        utterance:str,

        rhetoric:str

):


    prompt=f"""

Analyze why this rhetorical mechanism creates conflict.


Utterance:

{utterance}



Candidate rhetorical mechanism:

{rhetoric}



Decompose the conflict into four dimensions:


1. logic:
logical contradiction


2. emotion:
emotional polarity mismatch


3. entity:
speaker/entity inconsistency


4. context:
contextual/pragmatic inconsistency



Return JSON only:


{{
"logic":0.0,
"emotion":0.0,
"entity":0.0,
"context":0.0
}}


Each value ranges from 0 to 1.

"""


    result=call_llm(prompt)



    try:

        values=json.loads(result)


    except:


        values={

            "logic":0.0,

            "emotion":0.0,

            "entity":0.0,

            "context":0.0

        }




    components=[]


    for k,v in values.items():


        components.append(

            {


            "component":

                k,


            "attribution":

                float(v)

            }

        )



    score=np.mean(

        [

        float(v)

        for v in values.values()

        ]

    )



    return {


        "rhetorical_type":

            rhetoric,


        "important_components":

            components,


        "attribution_score":

            float(score)

    }








# ============================================================
# Rule-guided Calibration
#
# S_rule(r)
#
# S_rule(r)=
# N_activated(r)/N_total(r)
#
# ============================================================


def rule_guided_calibration(

        utterance:str,

        rhetoric:str

):


    rules=LINGUISTIC_RULES[rhetoric]



    rule_text="\n".join(

        [

        "- "+r

        for r in rules

        ]

    )



    prompt=f"""

You are a linguistic rule verifier.


Utterance:

{utterance}



Candidate rhetorical category:

{rhetoric}



Check which linguistic indicators are activated.



Possible indicators:

{rule_text}



Return JSON:


{{
"activated_rules":[
"matched indicator"
]
}}


Do not add explanations.

"""


    result=call_llm(prompt)



    try:


        output=json.loads(result)


        activated=output.get(

            "activated_rules",

            []

        )


    except:


        activated=[]



    # calculate according to paper formula


    score=(

        len(activated)

        /

        len(rules)

    )



    return {


        "rhetorical_type":

            rhetoric,


        "activated_rules":

            activated,


        "rule_score":

            float(score)

    }









# ============================================================
# Evidence Fusion
#
# S_final(r)
#
# S_final =
# alpha Conflict
# + beta Attribution
# + gamma Rule
#
# ============================================================


def evidence_fusion(

        stage1,

        attribution_results,

        rule_results,

        alpha=0.5,

        beta=0.3,

        gamma=0.2

):


    attr_map={

        x["rhetorical_type"]:

        x["attribution_score"]

        for x in attribution_results

    }



    rule_map={

        x["rhetorical_type"]:

        x["rule_score"]

        for x in rule_results

    }




    final_scores=[]



    for item in stage1["candidate_scores"]:


        r=item["rhetorical_type"]



        final_score=(


            alpha *

            item["conflict_score"]


            +

            beta *

            attr_map.get(

                r,

                0

            )


            +

            gamma *

            rule_map.get(

                r,

                0

            )


        )



        final_scores.append(

            {


            "rhetorical_type":

                r,


            "final_score":

                float(final_score)

            }

        )



    final_scores.sort(

        key=lambda x:

        x["final_score"],

        reverse=True

    )



    return {


        "ranking_scores":

            final_scores

    }
# ============================================================
# Build Structured Reasoning Signal z
#
# Strictly follows Appendix format
#
# ============================================================


def build_structured_reasoning_signal(

        utterance:str

):


    # -----------------------------
    # Stage-I
    # -----------------------------


    stage1, counterfactual_evidence = (
        stage1_conflict_ranking(
            utterance
        )
    )



    top_candidates = (

        stage1["top_candidates"]

    )



    # -----------------------------
    # Stage-II
    # Only analyze top candidates
    # -----------------------------


    attribution_results=[]


    rule_results=[]



    for rhetoric in top_candidates:


        attribution_results.append(

            conflict_attribution(

                utterance,

                rhetoric

            )

        )


        rule_results.append(

            rule_guided_calibration(

                utterance,

                rhetoric

            )

        )




    # -----------------------------
    # Evidence Fusion
    # -----------------------------


    fusion=evidence_fusion(

        stage1,

        attribution_results,

        rule_results

    )



    # ========================================================
    # Structured reasoning signal z
    #
    # EXACT FORMAT IN PAPER
    # ========================================================


    z={


        "input":{


            "utterance":

                utterance

        },



        "counterfactual_evidence":

            counterfactual_evidence,



        "stage1_conflict_ranking":{


            "candidate_scores":

                stage1[

                "candidate_scores"

                ],


            "top_candidates":

                stage1[

                "top_candidates"

                ]

        },



        "stage2_conflict_attribution":

            attribution_results,



        "rule_guided_evidence":

            rule_results,



        "evidence_fusion":

            fusion

    }



    return z







# ============================================================
# LLM Prediction from structured signal z
#
# z -> downstream LLM
#
# ============================================================



def llm_prediction_from_signal(

        z,

        protocol="Any-Match"

):



    prompt=f"""

You are a fine-grained rhetorical sarcasm classifier.


You are provided with a structured reasoning signal
generated by RACR.


The signal contains:

- counterfactual evidence
- conflict ranking
- conflict attribution
- linguistic rule evidence
- fused rhetorical scores


Use this evidence for final prediction.



Structured Reasoning Signal:

{json.dumps(

    z,

    indent=2,

    ensure_ascii=False

)}



Task:

Identify the rhetorical mechanism(s)
responsible for the sarcastic meaning.


Evaluation protocol:

{protocol}



Available labels:

{RHETORICAL_DEVICES}



Output ONLY labels separated by commas.

No explanation.

"""


    result=call_llm(prompt)



    return result







# ============================================================
# Prediction normalization
# ============================================================


def normalize_labels(

        prediction:str

):


    labels=[]


    text=prediction.lower()



    for label in RHETORICAL_DEVICES:


        if label.lower() in text:


            labels.append(label)



    return list(set(labels))







# ============================================================
# Top-1 evaluation
# ============================================================


def evaluate_top1(

        prediction_labels,

        gold_labels

):


    if len(prediction_labels)==0:

        return False



    return (

        prediction_labels[0]

        in

        gold_labels

    )








# ============================================================
# Top-2 evaluation
# ============================================================


def evaluate_top2(

        prediction_labels,

        gold_labels

):


    pred=set(

        prediction_labels[:2]

    )


    gold=set(

        gold_labels

    )


    return len(

        pred & gold

    )>0








# ============================================================
# Any-Match evaluation
#
# Multi-label RedSD setting
# ============================================================



def evaluate_any_match(

        prediction_labels,

        gold_labels

):


    pred=set(

        prediction_labels

    )


    gold=set(

        gold_labels

    )


    return bool(

        pred & gold

    )







# ============================================================
# Complete RACR inference
# ============================================================



def racr_inference(

        utterance:str,

        protocol="Any-Match"

):



    # Step 1:
    # Generate structured reasoning signal


    z=build_structured_reasoning_signal(

        utterance

    )



    # Step 2:
    # Feed z into downstream LLM


    prediction=llm_prediction_from_signal(

        z,

        protocol

    )



    labels=normalize_labels(

        prediction

    )



    return {


        "structured_reasoning_signal":

            z,


        "llm_raw_prediction":

            prediction,


        "final_labels":

            labels


    }







# ============================================================
# Save output
# ============================================================



def save_result(

        result,

        filename="racr_output.json"

):


    with open(

        filename,

        "w",

        encoding="utf-8"

    ) as f:


        json.dump(

            result,

            f,

            indent=2,

            ensure_ascii=False

        )



    print(

        f"Saved to {filename}"

    )








# ============================================================
# Main
# ============================================================


if __name__=="__main__":



    parser=argparse.ArgumentParser()



    parser.add_argument(

        "--utterance",

        type=str,

        required=True

    )



    parser.add_argument(

        "--protocol",

        type=str,

        default="Any-Match",

        choices=[

            "Top-1",

            "Top-2",

            "Any-Match"

        ]

    )



    parser.add_argument(

        "--output",

        default="racr_output.json"

    )




    args=parser.parse_args()





    result=racr_inference(

        args.utterance,

        args.protocol

    )




    save_result(

        result,

        args.output

    )




    print(

        "\n========== RACR Prediction =========="

    )


    print(

        result["final_labels"]

    )