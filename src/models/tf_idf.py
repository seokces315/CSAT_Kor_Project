import os
import sys

required_dir1 = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
required_dir2 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(required_dir1)
sys.path.append(required_dir2)

from utils import set_seed
from data import load_data

from torch.utils.data import Dataset

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_squared_error, mean_absolute_error

from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from catboost import CatBoostRegressor

import numpy as np
import ast
import optuna
import warnings

warnings.filterwarnings("ignore")


# Custom dataset for TF-IDF vectorize
class TFIDFDataset(Dataset):
    # Generator
    def __init__(self, df):
        self.df = df

    # Length getter
    def __len__(self):
        return len(self.df)

    # Getter : Wrapped by prompt
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        choices = ast.literal_eval(row["choices"])
        text = (
            f"[지문]\n{row['paragraph']}\n"
            f"[문제]\n{row['question']}\n"
            f"[보기]\n{row['question_plus']}\n"
            f"[선택지]\n1. {choices[0]}\n2. {choices[1]}\n3. {choices[2]}\n4. {choices[3]}\n5. {choices[4]}"
        )
        text = text.replace("[보기]\nnan\n", "")
        label = row["answer_rate"]
        return {"text": text, "label": label}


# Function for evaluation
def eval_with_Regressor(model_id, model_instance, train_set, test_set):

    # Model fitting
    X_train, Y_train = train_set
    model_instance.fit(X_train, Y_train)

    # Evaluate with test data
    X_test, Y_test = test_set
    test_preds = model_instance.predict(X_test)

    # Applying given metrics
    # 1. MSE, MAE
    mse = mean_squared_error(Y_test, test_preds)
    mae = mean_absolute_error(Y_test, test_preds)

    # 2. Accuracy (-10% ~ +10%)
    correct = (np.abs(Y_test - test_preds) <= 0.10).sum()
    total = len(Y_test)
    accuracy = 1.0 * correct / total

    print(f"[ {model_id} ]")
    print(f"MSE : {mse:.4f}")
    print(f"MAE : {mae:.4f}")
    print(f"Accuracy -> {accuracy:.4f}")
    print()


# Baseline flow for TF_IDF ML Method
def baseline_TI():

    # Set seed for reproducibility
    set_seed(42)

    # Load dataset
    data_path = f"{required_dir1}/data/CSAT_Kor.csv"
    csat_kor_df = load_data(data_path)

    # Prepare train/test custom dataset
    csat_kor_train_df, csat_kor_test_df = train_test_split(
        csat_kor_df,
        test_size=0.1,
        random_state=42,
        stratify=csat_kor_df["difficulty"],
    )
    csat_kor_train_dataset = TFIDFDataset(csat_kor_train_df)
    csat_kor_test_dataset = TFIDFDataset(csat_kor_test_df)

    # Dataset -> List
    train_texts = [example["text"] for example in csat_kor_train_dataset]
    train_labels = [float(example["label"]) for example in csat_kor_train_dataset]
    test_texts = [example["text"] for example in csat_kor_test_dataset]
    test_labels = [float(example["label"]) for example in csat_kor_test_dataset]
    # print(train_texts[0])
    # print()
    # print(len(train_texts))
    # print(len(test_texts))
    # print()

    # TF-IDF Vectorize
    vectorizer = TfidfVectorizer(max_features=5000)
    train_embeddings = vectorizer.fit_transform(train_texts)
    test_embeddings = vectorizer.transform(test_texts)

    # Define Regression Models
    model_dict = {
        "LightGBM": LGBMRegressor(random_state=42, verbose=-1),
        "RandomForest": RandomForestRegressor(random_state=42, verbose=0),
        "LinearRegression": LinearRegression(),
        "SVR": SVR(),
        "CatBoost": CatBoostRegressor(random_state=42, verbose=0),
    }

    # Train loop
    train_set = (train_embeddings, train_labels)
    test_set = (test_embeddings, test_labels)

    # Function for hyper parameter searching
    def objective(trial):
        params = {
            "iterations": trial.suggest_int("iterations", 100, 1000),
            "depth": trial.suggest_int("depth", 4, 10),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
            "random_strength": trial.suggest_float("random_strength", 1e-9, 10.0),
            "random_state": 42,
            "verbose": 0,
        }

        model = CatBoostRegressor(**params)
        model.fit(
            train_embeddings,
            train_labels,
            eval_set=(test_embeddings, test_labels),
            early_stopping_rounds=50,
            verbose=0,
        )

        preds = model.predict(test_embeddings)
        mae = mean_absolute_error(test_labels, preds)

        return mae

    # # Optuna 스터디 생성 및 최적화 수행
    # study = optuna.create_study(direction="minimize")
    # study.optimize(objective, n_trials=50)

    # # Optuna로 얻은 최적 하이퍼파라미터
    # best_params = study.best_trial.params

    # # 고정 파라미터 추가 (optuna에서 튜닝 안 한 것들)
    # best_params.update({"random_state": 42, "verbose": 0})

    # 최종 모델 학습
    # model_id = "CatBoost"
    # final_model = CatBoostRegressor(**best_params)

    # eval_with_Regressor(model_id, final_model, train_set, test_set)
    for model_id, model_instance in model_dict.items():
        eval_with_Regressor(
            model_id,
            model_instance,
            train_set,
            test_set,
        )


if __name__ == "__main__":
    baseline_TI()
