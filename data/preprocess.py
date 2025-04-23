import os
import glob
import json
import pandas as pd


# Function to preprocess json data & make csv file
def make_csv(data_folder, debug=False):

    # Main variables
    json_file_list = []

    # Generate json files list
    current_dir = os.getcwd()
    for folder in data_folder:
        folder_path = os.path.join(current_dir, folder, "*.json")
        json_file_list.extend(glob.glob(folder_path))
    if debug is True:
        print(len(json_file_list))
        print(json_file_list[:3])
        print()

    # Load json file
    total_data_dict = []
    for json_file in json_file_list:
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Data transformation & Column selection
        for item in data:
            # Data collection is limited on type "0"
            item_type = item.get("type")
            if item_type != 0:
                continue

            # 1 paragraph corresponds to 1 problem
            problem_length = len(item["problems"])
            for i in range(problem_length):
                new_item = {
                    "type": item_type,
                    "question_id": item["problems"][i]["question_id"],
                    "paragraph": item["paragraph"],
                    "question": item["problems"][i]["question"],
                    "question_plus": item["problems"][i].get("question_plus"),
                    "choices": item["problems"][i]["choices"],
                    "choices_rate": item["problems"][i]["choices_rate"],
                    "answer": item["problems"][i]["answer"],
                    "answer_rate": item["problems"][i]["answer_rate"],
                }
                total_data_dict.append(new_item)
    if debug is True:
        print(len(total_data_dict))
        print(total_data_dict[0])
        print()

    # Make csj file from total_data_dict
    csat_kor_df = pd.DataFrame(total_data_dict)
    csat_kor_df.to_csv("CSAT_Kor.csv", index=False, encoding="utf-8")


if __name__ == "__main__":
    # Data folder path
    data_folder = ["CSAT_Kor", "Even_Kor"]
    make_csv(data_folder, debug=True)
