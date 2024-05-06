import os
from openai import OpenAI
import pandas as pd
from tqdm import tqdm
from scorer import evaluate

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
                "content": "I think we've seen a deterioration of values. I think for a while as a nation we condoned those things we should have condemned. For a while, as I recall, it even seems to me that there was talk of legalizing or decriminalizing marijuana and other drugs, and I think that's all wrong. So we've seen a deterioration in values, and one of the things that I think we should do about it in terms of cause is to instill values into the young people in our schools. We got away, we got into this feeling that value- free education was the thing. And I don't believe that at all I do believe there are fundamental rights and wrongs as far as use. And, of course, as far as the how we make it better, yes, we can do better on interdiction."
            },
            {
                "role": "assistant", "content": "No"
            },
            {
                "role": "user", 
                "content": "I'm saying that those of us who are elected to positions of political leadership, Jim, have a special responsibility, not only to come up with programs, and I have outlined in detail the very important, very strong program of enforcement as well as drug education prevention. And Mr. Bush is right - the two go hand in hand. But if our government itself is doing business with people who we know are engaged in drug profiteering and drug trafficking, if we don't understand that that sends out a very, very bad message to our young people, it's a little difficult for me to understand just how we can reach out to that youngster that I talked about and to young people like her all over the country, and say to them we want to help you. Now, I've outlined in great detail a program for being tough on enforcement at home and abroad, doubling the number of drug enforcement agents, having a hemispheric summit soon after the 20th of January when we bring our democratic neighbors and allies together here in this hemisphere and go to work together."
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

test_df = pd.read_csv("data/CT24_checkworthy_english/with_context/CT24_checkworthy_english_test_with_context.tsv", sep="\t")

run_id = "gpt-fs-with-ctx-v2-test"

test_df.rename(columns={"Sentence_id": "id"}, inplace=True)
test_df["class_label"] = test_df.progress_apply(generate_response, axis=1)
test_df["run_id"] = run_id
test_df.drop(columns=["Text"], inplace=True)

test_df.to_csv(
    f"data/CT24_checkworthy_english/{run_id}_CT24_checkworthy_english_test.tsv", 
    sep="\t", 
    index=False
)

# acc, precision, recall, f1 = evaluate(
#     "data/CT24_checkworthy_english/CT24_checkworthy_english_dev-test.tsv", 
#     f"data/CT24_checkworthy_english/{run_id}_CT24_checkworthy_english_dev-test.tsv",
#     "english"
# )

# print(f"Metrics (positive class): Acc: {acc}, P: {precision}, R: {recall}, F1: {f1}")
