import os
import sys

# Get parent folder path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Append to sys.path
sys.path.append(parent_dir)

from torch.utils.data import Dataset

import pandas as pd

import ast
import re


# Function for role & persona template
def rp_choice(option):
    if option == 0:
        template = ""
    elif option == 1:
        template = "당신은 수능 국어 영역 출제 경력 10년 이상의 전문가로, 수험생들의 선택 경향과 정답률에 대한 깊은 통찰을 가지고 있습니다.\n주어진 지문, 문제, 보기, 선택지 등을 종합적으로 분석하여 실제 수험생들의 정답률을 예측하십시오.\n\n"
    elif option == 2:
        template = "당신은 국어 문제를 분석하여, 해당 문제를 접한 학생들이 얼마나 정답을 맞췄을지를 예측하는 베테랑 국어 강사입니다.\n주어진 지문, 문제, 보기, 선택지 등을 종합적으로 분석하여 실제 수험생들의 정답률을 예측하십시오.\n\n"
    elif option == 3:
        template = "나는 국어 문제의 난이도를 분석하고, 이를 기반으로 학생들의 정답률을 예측하는 교육 평가 인공지능이다.\n주어진 지문, 문제, 보기, 선택지 등을 종합적으로 분석하여 실제 수험생들의 정답률을 예측해보자.\n\n"
    return template


# Function to classify exam's style
def classify_exam(month):
    if month in [3, 4, 5, 7, 10]:
        style = "교육청 학력평가"
    elif month in [6, 9]:
        style = "평가원 모의고사"
    else:
        style = "대학수학능력시험"
    return style


# Custom dataset for fine-tune EXAONE
class CSATPromptDataset(Dataset):
    # Generator
    def __init__(self, df, option):
        self.df = df
        self.option = option

    # Length getter
    def __len__(self):
        return len(self.df)

    # Getter : Wrapped by prompt
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        template = rp_choice(self.option)
        style = classify_exam(row["month"])
        choices = ast.literal_eval(row["choices"])
        text = (
            f"{template}"
            # f"<정답률 추정 기준>\n- 정보가 분산된 긴 지문일수록 난이도가 높음\n- 지문과 선택지의 대응이 명확할수록 정답률이 높음\n- 선택지들이 추상적이거나 유사한 표현일 경우 정답률이 낮음\n- 문제 유형이 추론이나 간접적 해석이면 난이도가 높음\n- 빈출 유형 및 자주 나오는 문제는 정답률이 높음\n- 선지 간 난이도 편차가 클 경우, 정답률은 높음\n- 오답 선지가 명백히 틀린 경우, 정답률은 높음\n\n"
            # f"<출력 형식>\n- 정답률만 소수 둘째 자리까지 출력하십시오. (예: 0.73)\n- 그 외 설명이나 사족은 포함하지 마십시오.\n\n"
            f"[시험 종류] {row['year']}학년도 {row['month']}월 {style}\n\n"
            f"[지문]\n{row['paragraph']}\n\n"
            # f"[문제 번호]\n{row['question_id']}\n\n"
            f"[문제]\n{row['question']}\n\n"
            f"[보기]\n{row['question_plus']}\n\n"
            f"[선택지]\n1. {choices[0]}\n2. {choices[1]}\n3. {choices[2]}\n4. {choices[3]}\n5. {choices[4]}"
            # f"[정답]\n{row['answer']}\n\n"
            # f"---\n\n"
            # f"[정답률]\n"
        )
        text = re.sub(r"\[보기\]\nnan\n\n", "", text)
        label = row["answer_rate"]
        return {"text": text, "label": label}


# Function to raw data into appropriate forms
def preprocess_month(question_id):
    month = question_id[3]
    return int(month[1]) if month[0] == "0" else int(month)


# Function to raw data into appropriate forms
def preprocess_c_rate(choices_rate):
    choices_rate = ast.literal_eval(choices_rate)
    return [round(0.01 * rate, 2) for rate in choices_rate]


# Function to transform label into difficulty
def label_to_diff(label):
    return (
        0
        if label >= 0.90
        else 1 if label >= 0.80 else 2 if label >= 0.60 else 3 if label >= 0.50 else 4
    )


# Function to preprocess given data
def load_data(data_path):

    # Load csv file
    csat_kor_df = pd.read_csv(data_path)
    # Assert
    assert csat_kor_df["type"].eq(0).all()

    # Data transformation & Column selection
    csat_kor_df.drop(columns=["type"], axis=1, inplace=True)
    csat_kor_df["question_id"] = csat_kor_df["question_id"].map(
        lambda x: x.split("_")[1:] if x.split("_")[0] == "Even" else x.split("_")
    )
    csat_kor_df["year"] = csat_kor_df["question_id"].map(lambda x: int(x[2]))
    csat_kor_df["month"] = csat_kor_df["question_id"].map(preprocess_month)
    csat_kor_df["question_id"] = csat_kor_df["question_id"].map(lambda x: int(x[4]))
    csat_kor_df["answer_rate"] = csat_kor_df["answer_rate"].map(
        lambda x: round(0.01 * x, 2)
    )
    csat_kor_df["choices_rate"] = csat_kor_df["choices_rate"].map(preprocess_c_rate)
    csat_kor_df["difficulty"] = csat_kor_df["answer_rate"].map(label_to_diff)

    # Reorder dataframe's columns
    new_order = [
        "year",
        "month",
        "paragraph",
        "question_id",
        "question",
        "question_plus",
        "choices",
        "answer",
        "answer_rate",
        "choices_rate",
        "difficulty",
    ]
    csat_kor_df = csat_kor_df[new_order]

    return csat_kor_df


if __name__ == "__main__":
    dataset = "CSAT_Kor.csv"
    data_path = f"{parent_dir}/data/{dataset}"
    csat_kor_df = load_data(data_path)
    # csat_kor_df.to_csv("CSAT_Kor.csv", index=False, encoding="utf-8")
    csat_kor_dataset = CSATPromptDataset(csat_kor_df, 0)
    print(csat_kor_dataset[0]["text"])
    print()
