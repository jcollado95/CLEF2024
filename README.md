CLEF2024

This repository contains the code implemented for our CLEF2024 participation in task 1 of the [CheckThat! Lab](https://checkthat.gitlab.io/clef2024/task1/).

We have taken two approaches based on GPT3.5:

1.  Base: Fewshot prompting.
2.  With context: Fewshot prompting where we have concatenated texts with consecutive previous texts according to the Sentence_id field.