import pandas as pd 
import numpy as np


# # Code for random preprocess of palisades data from various sources
# csv_data = pd.read_csv('palisades_data.csv')["text"]
# xls_data = pd.read_excel('palisades.xlsx', sheet_name="Sheet1")
# xls_data.columns = ["text"]
# print(xls_data.head())

# tweets = csv_data.to_list() + xls_data["text"].to_list()
# # Remove duplicates
# tweets = list(set(tweets))

# # save new data to csv
# tweets_df = pd.DataFrame(tweets, columns=["text"])
# tweets_df.to_csv('palisades_combined.csv', index=False)

# tweets_df = pd.read_csv('palisades_combined.csv')
# # Remove duplicates
# tweets_df = tweets_df.drop_duplicates(subset=["text"])
# # Remove empty rows
# tweets_df = tweets_df.dropna(subset=["text"])
# # Remove rows with only whitespace
# tweets_df = tweets_df[tweets_df["text"].str.strip() != ""]

# tweets_df.to_csv('palisades_cleaned.csv', index=False)
# tweets_df = pd.read_csv('palisades_cleaned.csv')
# updated_df = pd.DataFrame()
# updated_df["tweet_id"] = np.array(list(range(1, len(tweets_df["text"]) + 1)))
# updated_df["tweet_text"] = tweets_df["text"].to_list()
# updated_df.to_csv('palisades_cleaned.tsv', sep='\t', index=False)

# tests_df = pd.read_csv('palisades_test.csv')
# print(tests_df.head())
# print(tests_df.columns)


# # Code to check the cleaned palisades data
# cleaned_df = pd.read_csv('palisades_cleaned.tsv', sep='\t')
# print(cleaned_df.head())
# print(cleaned_df.columns)
# print(cleaned_df["tweet_text"].isnull().sum())  # Check for null values in the 'tweet_text' column
# print(len(cleaned_df["tweet_text"]))

# # code to convert manually annotated palisades test data to tsv to account for multiple commas

# ids = []
# texts = []
# labels = []

# with open("palisades_test.csv", "r", encoding="utf-8") as f:
#     for line in f:
#         line = line.strip()
#         if not line:
#             continue  # skip empty lines

#         # Find first and last comma
#         first_comma = line.find(",")
#         last_comma = line.rfind(",")

#         if first_comma == -1 or last_comma == -1 or first_comma == last_comma:
#             continue  # skip malformed lines

#         id_part = line[:first_comma]
#         text_part = line[first_comma + 1:last_comma]
#         label_part = line[last_comma + 1:]

#         ids.append(id_part.strip())
#         texts.append(text_part.strip())
#         labels.append(label_part.strip())

# print(len(ids), len(texts), len(labels))
# print(ids[:5], texts[:5], labels[:5])

# test_df = pd.DataFrame({"tweet_id": np.array(ids), "tweet_text": texts, "class_label": labels})
# test_df.to_csv('palisades_test.tsv', sep='\t', index=False)

# # code to extract the remaining unlabeled tweets from palisades_cleaned based on ids in palisades_test.tsv

# test_df = pd.read_csv('palisades_test.tsv', sep='\t')
# print(test_df.head())
# cleaned_df = pd.read_csv('palisades_cleaned.tsv', sep='\t')
# print(len(cleaned_df))

# test_df["tweet_id"] = test_df["tweet_id"].astype(str)
# cleaned_df["tweet_id"] = cleaned_df["tweet_id"].astype(str)

# filtered_df = cleaned_df[~cleaned_df["tweet_id"].isin(test_df["tweet_id"])]
# print(filtered_df.head())
# print(len(filtered_df))

# filtered_df.to_csv('palisades_unlabeled.tsv', sep='\t', index=False)

# code to combine multiple historicsl datasets into one

df1_train = pd.read_csv("california_wildfires_2018_train.tsv", sep="\t")
df1_dev = pd.read_csv("california_wildfires_2018_dev.tsv", sep="\t")
df1_test = pd.read_csv("california_wildfires_2018_test.tsv", sep="\t")

df2_train = pd.read_csv("canada_wildfires_2016_train.tsv", sep="\t")
df2_dev = pd.read_csv("canada_wildfires_2016_dev.tsv", sep="\t")
df2_test = pd.read_csv("canada_wildfires_2016_test.tsv", sep="\t")

historical_train = pd.concat([df1_train, df2_train], ignore_index=True)
historical_dev = pd.concat([df1_dev, df2_dev], ignore_index=True)   
historical_test = pd.concat([df1_test, df2_test], ignore_index=True)

historical_train.to_csv("historical_train.tsv", sep="\t", index=False)
historical_dev.to_csv("historical_dev.tsv", sep="\t", index=False)
historical_test.to_csv("historical_test.tsv", sep="\t", index=False)



