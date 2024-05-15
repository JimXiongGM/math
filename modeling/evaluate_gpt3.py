import os
import openai
import numpy as np
import operator
import json
from dataset.util import last_boxed_only_string
from math_equivalence import is_equiv

os.environ["https_proxy"] = "http://127.0.0.1:7893"
os.environ["http_proxy"] = "http://127.0.0.1:7893"

DEFAULT_KEY = os.environ.get("OPENAI_API_KEY", None)
assert DEFAULT_KEY is not None, "OPENAI_API_KEY is None"

DEFAULT_CHAT_MODEL = "gpt-4-turbo"

client = openai.OpenAI(api_key=DEFAULT_KEY)


def call_engine(train_prompt, problem):
    '''
    Given a problem, returns the most likely answer determined by the GPT engine 
    '''
    test_question = "\n\nQuestion: " + problem.strip() + "\n" + "Let's think step by step:"
    prompt = train_prompt + test_question
    # print(len(prompt))
    num_tokens = 1024

    system_content="You are a math problem solver."
    messages =  [
            {"role": "system", "content": system_content},
            {"role": "user", "content": prompt},
        ]

    response = client.chat.completions.create(
        model=DEFAULT_CHAT_MODEL,
        messages=messages,
        temperature=0,
        top_p=1,
        n=1,
        stop=["\n\n"],
        max_tokens=num_tokens,
        # presence_penalty=presence_penalty,
        # frequency_penalty=frequency_penalty,
        # logit_bias=logit_bias,
        # **kwargs,
    )
    response = json.loads(response.model_dump_json())

    tokens = response["choices"][0]["message"]["content"]
    # startindex = -1 * num_tokens
    # endindex = -1 * num_tokens + 1
    # for token in tokens[startindex + 1:]:
    #     if token == "$" or token == "###" or token == "\n":
    #         break
    #     else:
    #         endindex += 1
    # final_answer = ""
    # for i in range(startindex, endindex):
    #     all_answers = c["choices"][0]["logprobs"]["top_logprobs"][i]
    #     best_answer = max(all_answers.items(), key=operator.itemgetter(1))[0]
    #     final_answer += best_answer
    # return final_answer

    # parse the output after The answer is
    start = tokens.find("The answer is")
    if start == -1:
        return None
    start += len("The answer is")
    final_answer = tokens[start:].strip()
    if final_answer[-1] == ".":
        final_answer = final_answer[:-1]
    return final_answer

def remove_boxed(s):
    left = "\\boxed{"
    try:
        assert s[:len(left)] == left
        assert s[-1] == "}"
        return s[len(left):-1]
    except:
        return None

# with open("modeling/train_prompt.txt", "r") as f:
with open("modeling/prompt_grad_5.txt", "r") as f:
    train_prompt = f.read()

rootdir = "MATH/test"


def run(max=-1):
    outputs = []
    answers = []
    types = []
    levels = []

    fnames_list = []

    cors = {}
    subject_cors = {}
    level_cors = {}
    correct = 0
    total = 0
    for subdir, dirs, files in os.walk(rootdir):
        for idx,file in enumerate(files):
            
            # debug
            if idx == 0:
                continue

            fnames_list.append(os.path.join(subdir, file))
            with open(os.path.join(subdir, file), 'r') as fp:
                try:
                    problem_data = json.load(fp)
                except Exception as e:
                    print(f"Error loading JSON from {file}", e)
                    raise e
                prob_level = problem_data["level"]
                prob_type = problem_data["type"]
                try:
                    prob_level = int(prob_level.split("Level ")[1])
                except:
                    prob_level = None
                model_output = call_engine(train_prompt, problem_data["problem"])
                answer = remove_boxed(last_boxed_only_string(problem_data["solution"]))

                levels.append(prob_level)
                types.append(prob_type)
                outputs.append(model_output)
                answers.append(answer)

                print("Model output:")
                print(model_output)
                print("Correct answer:")
                print(answer)
                print("--------------------------------------------")

                try:
                    equiv = is_equiv(model_output, answer)
                except:
                    equiv = False
                if (prob_level, prob_type) in cors:
                    cors[(prob_level, prob_type)].append(equiv)
                else:
                    cors[(prob_level, prob_type)] = [equiv]
                if prob_level in level_cors:
                    level_cors[prob_level].append(equiv)
                else:
                    if prob_level is not None:
                        level_cors[prob_level] = [equiv]
                if prob_type in subject_cors:
                    subject_cors[prob_type].append(equiv)
                else:
                    if prob_type is not None:
                        subject_cors[prob_type] = [equiv]
                if equiv:
                    correct += 1
                total += 1

                print(str(correct) + "/" + str(total))

            if max > 0 and total > max:
                break
        if max > 0 and total > max:
            break

    with open("outputs_answers_gpt4_turbo.txt", "w+") as f:
        for k, (output, answer, prob_type, prob_level, fname) in enumerate(zip(outputs, answers, types, levels, fnames_list)):
            f.write("{} TYPE: {} | LEVEL: {} | OUTPUT: {} | ANSWER: {} | FNAME: {}\n".format(k, prob_type, prob_level, output, answer, fname))

        f.write("#####################\n")
        # also get accuracies for each
        for subject in ['Prealgebra', 'Algebra', 'Number Theory', 'Counting & Probability', 'Geometry', 'Intermediate Algebra', 'Precalculus']:
            for level in range(1, 6):
                key = (level, subject)
                if key not in cors.keys():
                    print("Skipping", key)
                    continue
                cors_list = cors[key]
                print("{} Level {} Accuracy = {}/{} = {:.3f}".format(subject, level, np.sum(cors_list), len(cors_list), np.mean(cors_list)))
                f.write("{} Level {} Accuracy = {}/{} = {:.3f}\n".format(subject, level, np.sum(cors_list), len(cors_list), np.mean(cors_list)))
        print("#####################")
        f.write("#####################\n")
        for level in sorted(level_cors):
            if level not in level_cors.keys():
                print("Skipping", level)
                continue
            cors_list = level_cors[level]
            print("Level {} Accuracy = {}/{} = {:.3f}".format(level, np.sum(cors_list), len(cors_list), np.mean(cors_list)))
            f.write("Level {} Accuracy = {}/{} = {:.3f}\n".format(level, np.sum(cors_list), len(cors_list), np.mean(cors_list)))
        print("#####################")
        f.write("#####################\n")
        for subject in ['Prealgebra', 'Algebra', 'Number Theory', 'Counting & Probability', 'Geometry', 'Intermediate Algebra', 'Precalculus']:
            if subject not in subject_cors.keys():
                print("Skipping", subject)
                continue
            cors_list = subject_cors[subject]
            print("{} Accuracy = {}/{} = {:.3f}".format(subject, np.sum(cors_list), len(cors_list), np.mean(cors_list)))
            f.write("{} Accuracy = {}/{} = {:.3f}\n".format(subject, np.sum(cors_list), len(cors_list), np.mean(cors_list)))
        print("#####################")
        f.write("#####################\n")
        print("Overall Accuracy = {}/{} = {:.3f}".format(correct, total, correct/total))
        f.write("Overall Accuracy = {}/{} = {:.3f}\n".format(correct, total, correct/total))

if __name__ == "__main__":
    run(max=10)

