import os
from openai import OpenAI
import pandas as pd
from tqdm import tqdm

api_key = os.environ["OPENAI_API_KEY"]
client = OpenAI(api_key=api_key)
tqdm.pandas()

def generate_response(row):
    response = client.chat.completions.create(
        model = "gpt-3.5-turbo",
        messages = [
            {
                "role": "system", 
                "content": "Check-worthiness definition: The process of finding whether a given TEXT contains verifiable factual claims prone to be fact-checked. You are an expert in check-worthiness. Determine if the TEXT contains a verifiable factual claim that is subject to fact-checking. Before responding, systematically consider these auxiliary questions: 'Does it contain a verifiable factual claim?' and 'Is it harmful?' Respond only with Yes if the TEXT contains such a claim, and No if it does not.\nTEXT:"
            },
            {
                "role": "user", 
                "content": "And so I know that this campaign has caused some questioning and worries on the part of many leaders across the globe."
            },
            {
                "role": "assistant", "content": "No"
            },
            {
                "role": "user", 
                "content": "I must remind him the Democrats have controlled the Congress for the last twenty-two years and they wrote all the tax bills."
            },
            {
                "role": "assistant", "content": "Yes"
            },
            {
                "role": "user", 
                "content": row["Text"]
            }
        ]
    )

    return response.choices[0].message.content

test_df = pd.read_csv("data/CT24_checkworthy_test/english/CT24_checkworthy_english_test.tsv", sep="\t")

run_id = "gpt-fs-no-ctx-v2-test"

test_df.rename(columns={"Sentence_id": "id"}, inplace=True)
test_df["class_label"] = test_df.progress_apply(generate_response, axis=1)
test_df["run_id"] = run_id
test_df.drop(columns=["Text"], inplace=True)

test_df.to_csv(
    f"data/CT24_checkworthy_english/{run_id}_CT24_checkworthy_english_test.tsv", 
    sep="\t", 
    index=False
)
