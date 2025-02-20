# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pyre-unsafe

import abc
import logging
import os
import sys
import tarfile
from typing import Dict, Optional, Union
from tqdm import tqdm
from urllib.request import urlretrieve
from zipfile import ZipFile
import numpy as np
import pandas as pd
from transformers import BertTokenizer, BertModel
import torch
import pickle
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logging.basicConfig(stream=sys.stdout, level=logging.INFO)


class DataProcessor:
    """
    This preprocessor does not remap item_ids. This is intended so that we can easily join other
    side-information based on item_ids later.
    """

    def __init__(
        self,
        prefix: str,
        expected_num_unique_items: Optional[int],
        expected_max_item_id: Optional[int],
        save_text_info: bool = False
    ) -> None:
        self._prefix: str = prefix
        self._expected_num_unique_items = expected_num_unique_items
        self._expected_max_item_id = expected_max_item_id
        self.save_text_info = save_text_info

    @abc.abstractmethod
    def expected_num_unique_items(self) -> Optional[int]:
        return self._expected_num_unique_items

    @abc.abstractmethod
    def expected_max_item_id(self) -> Optional[int]:
        return self._expected_max_item_id

    @abc.abstractmethod
    def processed_item_csv(self) -> str:
        pass

    def output_format_csv(self) -> str:
        return f"tmp/{self._prefix}/sasrec_format.csv"

    def to_seq_data(
        self,
        ratings_data: pd.DataFrame,
        user_data: Optional[pd.DataFrame] = None,
        movie_data: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        if movie_data is not None and self.save_text_info:
            print("movie_data is not None")
            token_embedding_map = movie_data.set_index("movie_id")["text_info_embedding"].to_dict()
            with open(f"tmp/{self._prefix}/token_embedding_map.pkl", "wb") as pickle_file:
                pickle.dump(token_embedding_map, pickle_file)
            
        if user_data is not None:
            ratings_data_transformed = ratings_data.join(
                user_data.set_index("user_id"), on="user_id"
            )
        else:
            ratings_data_transformed = ratings_data
        
        ratings_data_transformed['sequence_len'] = ratings_data_transformed['item_ids'].apply(
            lambda x: len(x)
        )
        ratings_data_transformed['sequence_len'].describe()
        ratings_data_transformed.item_ids = ratings_data_transformed.item_ids.apply(
            lambda x: ",".join([str(v) for v in x])
        )
        ratings_data_transformed.ratings = ratings_data_transformed.ratings.apply(
            lambda x: ",".join([str(v) for v in x])
        )
        ratings_data_transformed.timestamps = ratings_data_transformed.timestamps.apply(
            lambda x: ",".join([str(v) for v in x])
        )
        ratings_data_transformed.rename(
            columns={
                "item_ids": "sequence_item_ids",
                "ratings": "sequence_ratings",
                "timestamps": "sequence_timestamps",
            },
            inplace=True,
        )
        print(ratings_data_transformed.columns)
        return ratings_data_transformed

    def file_exists(self, name: str) -> bool:
        return os.path.isfile("%s/%s" % (os.getcwd(), name))


class MovielensSyntheticDataProcessor(DataProcessor):
    def __init__(
        self,
        prefix: str,
        expected_num_unique_items: Optional[int] = None,
        expected_max_item_id: Optional[int] = None,
    ) -> None:
        super().__init__(prefix, expected_num_unique_items, expected_max_item_id)

    def preprocess_rating(self) -> None:
        return


class MovielensDataProcessor(DataProcessor):
    def __init__(
        self,
        download_path: str,
        saved_name: str,
        prefix: str,
        convert_timestamp: bool,
        expected_num_unique_items: Optional[int] = None,
        expected_max_item_id: Optional[int] = None,
        text_embedding_model: str = 'bert-base-uncased',
        cold_sequence_len: int = 6,
    ) -> None:
        super().__init__(prefix, expected_num_unique_items, expected_max_item_id)
        self._download_path = download_path
        self._saved_name = saved_name
        self._convert_timestamp: bool = convert_timestamp
        self._text_embedding_model = text_embedding_model
        self.cold_sequence_len = cold_sequence_len

    def download(self) -> None:
        if not self.file_exists(self._saved_name):
            urlretrieve(self._download_path, self._saved_name)
        if self._saved_name[-4:] == ".zip":
            ZipFile(self._saved_name, "r").extractall(path="tmp/")
        else:
            with tarfile.open(self._saved_name, "r:*") as tar_ref:
                tar_ref.extractall("tmp/")

    def processed_item_csv(self) -> str:
        return f"tmp/processed/{self._prefix}/movies.csv"

    def sasrec_format_csv_by_user_train(self, user_info=False) -> str:
        if user_info:
            return f"tmp/{self._prefix}/sasrec_format_by_user_train_userinfo.csv"
        else:
            return f"tmp/{self._prefix}/sasrec_format_by_user_train.csv"

    def sasrec_format_csv_by_user_test(self) -> str:
        return f"tmp/{self._prefix}/sasrec_format_by_user_test.csv"

    def sasrec_format_csv_by_user_test_for_user_cold_rec(self, user_info=False) -> str:
        if user_info:
            return f"tmp/{self._prefix}/sasrec_format_by_user_test_for_user_cold_rec_len={self.cold_sequence_len}_userinfo.csv"
        else:
            return f"tmp/{self._prefix}/sasrec_format_by_user_test_for_user_cold_rec_len={self.cold_sequence_len}.csv"

    def preprocess_rating(self) -> int:
        self.download()

        if self._prefix == "ml-1m":
            users = pd.read_csv(
                f"tmp/{self._prefix}/users.dat",
                sep="::",
                names=["user_id", "sex", "age_group", "occupation", "zip_code"],
            )
            ratings = pd.read_csv(
                f"tmp/{self._prefix}/ratings.dat",
                sep="::",
                names=["user_id", "movie_id", "rating", "unix_timestamp"],
            )
            movies = pd.read_csv(
                f"tmp/{self._prefix}/movies.dat",
                sep="::",
                names=["movie_id", "title", "genres"],
                encoding="iso-8859-1",
            )
        elif self._prefix == "ml-20m":
            # ml-20m
            # ml-20m doesn't have user data.
            users = None
            # ratings: userId,movieId,rating,timestamp
            ratings = pd.read_csv(
                f"tmp/{self._prefix}/ratings.csv",
                sep=",",
            )
            ratings.rename(
                columns={
                    "userId": "user_id",
                    "movieId": "movie_id",
                    "timestamp": "unix_timestamp",
                },
                inplace=True,
            )
            # movieId,title,genres
            # 1,Toy Story (1995),Adventure|Animation|Children|Comedy|Fantasy
            # 2,Jumanji (1995),Adventure|Children|Fantasy
            movies = pd.read_csv(
                f"tmp/{self._prefix}/movies.csv",
                sep=",",
                encoding="iso-8859-1",
            )
            movies.rename(columns={"movieId": "movie_id"}, inplace=True)
        else:
            assert self._prefix == "ml-20mx16x32"
            # ml-1b
            user_ids = []
            movie_ids = []
            for i in range(16):
                train_file = f"tmp/{self._prefix}/trainx16x32_{i}.npz"
                with np.load(train_file) as data:
                    user_ids.extend([x[0] for x in data["arr_0"]])
                    movie_ids.extend([x[1] for x in data["arr_0"]])
            ratings = pd.DataFrame(
                data={
                    "user_id": user_ids,
                    "movie_id": movie_ids,
                    "rating": user_ids,  # placeholder
                    "unix_timestamp": movie_ids,  # placeholder
                }
            )
            users = None
            movies = None

        if movies is not None:
            # ML-1M and ML-20M only
            movies["year"] = movies["title"].apply(lambda x: x[-5:-1])
            movies["cleaned_title"] = movies["title"].apply(lambda x: x[:-7])
            # movies.year = pd.Categorical(movies.year)
            # movies["year"] = movies.year.cat.codes
            movies["text_info"] = movies.apply(
                lambda row: f"{row['title']}, {row['genres']}",
                axis=1
            )

            def get_bert_embeddings_batch(texts):
                """
                将句子列表转换为 BERT 嵌入
                :param texts: List[str], 输入句子列表
                :return: List[torch.Tensor], 每个句子的 (768,) 嵌入
                """
                # 初始化 BERT 模型和 Tokenizer
                tokenizer = BertTokenizer.from_pretrained(self._text_embedding_model)
                model = BertModel.from_pretrained(self._text_embedding_model)
                model = model.to(device)

                inputs = tokenizer(
                    texts,
                    return_tensors="pt",
                    truncation=True,
                    padding="max_length",
                    max_length=128
                )
                inputs = {key: value.to(device) for key, value in inputs.items()}

                with torch.no_grad():
                    outputs = model(**inputs)

                # 提取每个句子的 [CLS] 嵌入
                embeddings = outputs.last_hidden_state[:, 0, :].cpu().tolist()  # 移动回 CPU
                return embeddings

            batch_size = 32
            all_embeddings = []
            if self.save_text_info:
                for i in tqdm(range(0, len(movies), batch_size)):
                    batch_texts = movies["text_info"].iloc[i:i+batch_size].tolist()
                    batch_embeddings = get_bert_embeddings_batch(batch_texts)
                    all_embeddings.extend(batch_embeddings)

                movies["text_info_embedding"] = all_embeddings

        if users is not None:
            ## Users (ml-1m only)
            users.sex = pd.Categorical(users.sex)
            users["sex"] = users.sex.cat.codes

            users.age_group = pd.Categorical(users.age_group)
            users["age_group"] = users.age_group.cat.codes

            users.occupation = pd.Categorical(users.occupation)
            users["occupation"] = users.occupation.cat.codes

            users.zip_code = pd.Categorical(users.zip_code)
            users["zip_code"] = users.zip_code.cat.codes

        # Normalize movie ids to speed up training
        print(
            f"{self._prefix} #item before normalize: {len(set(ratings['movie_id'].values))}"
        )
        max_id = max(set(ratings['movie_id'].values))
        print(
            f"{self._prefix} max item id before normalize: {max(set(ratings['movie_id'].values))}"
        )
        # print(f"ratings.movie_id.cat.categories={ratings.movie_id.cat.categories}; {type(ratings.movie_id.cat.categories)}")
        # print(f"ratings.movie_id.cat.codes={ratings.movie_id.cat.codes}; {type(ratings.movie_id.cat.codes)}")
        # print(movie_id_to_cat)
        # ratings["movie_id"] = ratings.movie_id.cat.codes
        # print(f"{self._prefix} #item after normalize: {len(set(ratings['movie_id'].values))}")
        # print(f"{self._prefix} max item id after normalize: {max(set(ratings['movie_id'].values))}")
        # movies["remapped_id"] = movies["movie_id"].apply(lambda x: movie_id_to_cat[x])

        if self._convert_timestamp:
            ratings["unix_timestamp"] = pd.to_datetime(
                ratings["unix_timestamp"], unit="s"
            )

        # Save primary csv's
        if not os.path.exists(f"tmp/processed/{self._prefix}"):
            os.makedirs(f"tmp/processed/{self._prefix}")
        if users is not None:
            users.to_csv(f"tmp/processed/{self._prefix}/users.csv", index=False)
        if movies is not None:
            movies.to_csv(f"tmp/processed/{self._prefix}/movies.csv", index=False)
        ratings.to_csv(f"tmp/processed/{self._prefix}/ratings.csv", index=False)

        num_unique_users = len(set(ratings["user_id"].values))
        num_unique_items = len(set(ratings["movie_id"].values))

        # SASRec version
        ratings_group = ratings.sort_values(by=["unix_timestamp"]).groupby("user_id")
        seq_ratings_data = pd.DataFrame(
            data={
                "user_id": list(ratings_group.groups.keys()),
                "item_ids": list(ratings_group.movie_id.apply(list)),
                "ratings": list(ratings_group.rating.apply(list)),
                "timestamps": list(ratings_group.unix_timestamp.apply(list)),
            }
        )

        result = pd.DataFrame([[]])
        for col in ["item_ids"]:
            result[col + "_mean"] = seq_ratings_data[col].apply(len).mean()
            result[col + "_min"] = seq_ratings_data[col].apply(len).min()
            result[col + "_max"] = seq_ratings_data[col].apply(len).max()
            value_counts = seq_ratings_data[col].apply(len).value_counts(normalize=True)

            # 绘制频率分布图
            plt.figure(figsize=(10, 6))
            value_counts.plot(kind='bar')
            plt.title(f"Frequency Distribution of {col}")
            plt.xlabel("Values")
            plt.ylabel("Frequency")
            plt.xticks(rotation=45)
            plt.savefig(f"tmp/{self._prefix}/seq_len_dist.png")
        percentile_20 = np.percentile(seq_ratings_data["item_ids"].apply(len), 20)
        print(self._prefix, "20% 分位数:", percentile_20)
        print(result)
        # print(self._expected_max_item_id, self._expected_num_unique_items)

        if self._prefix == "ml-1m":
            max_id = 3952
        if users is not None:
            for col in ['sex','age_group','occupation']:
                users[col] += int(max_id + 1)
                max_id = users[col].max()
                print(max_id)

        # Cold item
        train_list = [item for sublist in seq_ratings_data['item_ids'] for item in sublist[:-1]]
        test_list = [sublist[-1] for sublist in seq_ratings_data['item_ids']]

        # 统计 train list 中的每个元素出现的次数
        train_item_counts = pd.Series(train_list).value_counts()

        # 筛选出满足条件的 items
        target_items = [item for item in test_list if train_item_counts.get(item, 0) <= 5]

        # 打印结果
        print(target_items)
        print(len(target_items))
        print(len(train_list))
        print(len(set(train_list)))
        print(len(test_list))
        print(len(set(test_list)))

        seq_ratings_data = self.to_seq_data(seq_ratings_data, users, movies)
        seq_ratings_data.sample(frac=1).reset_index().to_csv(
            self.output_format_csv(), index=False, sep=","
        )

        # Split by user ids (not tested yet)
        user_id_split = int(num_unique_users * 0.9)
        seq_ratings_data_train = seq_ratings_data[
            seq_ratings_data["user_id"] <= user_id_split
        ]
        seq_ratings_data_train.sample(frac=1).reset_index().to_csv(
            self.sasrec_format_csv_by_user_train(),
            index=False,
            sep=",",
        )
        seq_ratings_data_test = seq_ratings_data[
            seq_ratings_data["user_id"] > user_id_split
        ]
        seq_ratings_data_test.sample(frac=1).reset_index().to_csv(
            self.sasrec_format_csv_by_user_test(), index=False, sep=","
        )
        print(
            f"{self._prefix}: train num user: {len(set(seq_ratings_data_train['user_id'].values))}"
        )
        print(
            f"{self._prefix}: test num user: {len(set(seq_ratings_data_test['user_id'].values))}"
        )

        # train_items = sum(seq_ratings_data_train['sequence_item_ids'].apply(lambda x: x.split(",")),[])
        # test_items = sum(seq_ratings_data_test['sequence_item_ids'].apply(lambda x: x.split(",")),[])
        # A_set = set(test_items)
        # B_set = set(train_items)
        # A_not_in_B_set = A_set - B_set  # 等价于 A_set.difference(B_set)
        # A_not_in_B = list(A_not_in_B_set)  # 如果还需要转换回列表
        # print(A_not_in_B)
        # print(len(A_not_in_B))

        # cut seq to cold_sequence_len
        seq_ratings_data_test['sequence_item_ids'] = seq_ratings_data_test['sequence_item_ids'].apply(lambda x: ",".join(x.split(",")[:self.cold_sequence_len]))
        seq_ratings_data_test['sequence_ratings'] = seq_ratings_data_test['sequence_ratings'].apply(lambda x: ",".join(x.split(",")[:self.cold_sequence_len]))
        seq_ratings_data_test['sequence_timestamps'] = seq_ratings_data_test['sequence_timestamps'].apply(lambda x: ",".join(x.split(",")[:self.cold_sequence_len]))
        seq_ratings_data_test['sequence_len'] = self.cold_sequence_len 
        
        seq_ratings_data_test.sample(frac=1).reset_index().to_csv(
            self.sasrec_format_csv_by_user_test_for_user_cold_rec(False), index=False, sep=","
        )

        # add user info
        if users is not None:
            for df in [seq_ratings_data_train, seq_ratings_data_test]:
                df["sequence_item_ids"] = df.apply(
                    lambda row: f"{row['sex']},{row['age_group']},{row['occupation']}," + row["sequence_item_ids"],
                    axis=1
                )
                df['sequence_ratings'] = df['sequence_ratings'].apply(lambda x: "0,0,0," + x)
                df['sequence_timestamps'] = df['sequence_timestamps'].apply(lambda x: "0,0,0," + x)
                df['sequence_len'] = self.cold_sequence_len + 3
            seq_ratings_data_test.sample(frac=1).reset_index().to_csv( 
                self.sasrec_format_csv_by_user_test_for_user_cold_rec(True), index=False, sep=","
            )
            seq_ratings_data_train.sample(frac=1).reset_index().to_csv( 
                self.sasrec_format_csv_by_user_train(True), index=False, sep=","
            )

        # if self.expected_num_unique_items() is not None:
        #     assert (
        #         self.expected_num_unique_items() == num_unique_items
        #     ), f"Expected items: {self.expected_num_unique_items()}, got: {num_unique_items}"

        return num_unique_items


class AmazonDataProcessor(DataProcessor):
    def __init__(
        self,
        download_path: str,
        saved_name: str,
        prefix: str,
        expected_num_unique_items: Optional[int],
    ) -> None:
        super().__init__(
            prefix,
            expected_num_unique_items=expected_num_unique_items,
            expected_max_item_id=None,
        )
        self._download_path = download_path
        self._saved_name = saved_name
        self._prefix = prefix

    def download(self) -> None:
        if not self.file_exists(self._saved_name):
            urlretrieve(self._download_path, self._saved_name)

    def preprocess_rating(self) -> int:
        self.download()

        ratings = pd.read_csv(
            self._saved_name,
            sep=",",
            names=["user_id", "item_id", "rating", "timestamp"],
        )
        print(f"{self._prefix} #data points before filter: {ratings.shape[0]}")
        print(
            f"{self._prefix} #user before filter: {len(set(ratings['user_id'].values))}"
        )
        print(
            f"{self._prefix} #item before filter: {len(set(ratings['item_id'].values))}"
        )

        # filter users and items with presence < 5
        item_id_count = (
            ratings["item_id"]
            .value_counts()
            .rename_axis("unique_values")
            .reset_index(name="item_count")
        )
        user_id_count = (
            ratings["user_id"]
            .value_counts()
            .rename_axis("unique_values")
            .reset_index(name="user_count")
        )
        ratings = ratings.join(item_id_count.set_index("unique_values"), on="item_id")
        ratings = ratings.join(user_id_count.set_index("unique_values"), on="user_id")
        ratings = ratings[ratings["item_count"] >= 5]
        ratings = ratings[ratings["user_count"] >= 5]
        print(f"{self._prefix} #data points after filter: {ratings.shape[0]}")

        # categorize user id and item id
        ratings["item_id"] = pd.Categorical(ratings["item_id"])
        ratings["item_id"] = ratings["item_id"].cat.codes
        ratings["user_id"] = pd.Categorical(ratings["user_id"])
        ratings["user_id"] = ratings["user_id"].cat.codes
        print(
            f"{self._prefix} #user after filter: {len(set(ratings['user_id'].values))}"
        )
        print(
            f"{self._prefix} #item ater filter: {len(set(ratings['item_id'].values))}"
        )

        num_unique_items = len(set(ratings["item_id"].values))

        # SASRec version
        ratings_group = ratings.sort_values(by=["timestamp"]).groupby("user_id")

        seq_ratings_data = pd.DataFrame(
            data={
                "user_id": list(ratings_group.groups.keys()),
                "item_ids": list(ratings_group.item_id.apply(list)),
                "ratings": list(ratings_group.rating.apply(list)),
                "timestamps": list(ratings_group.timestamp.apply(list)),
            }
        )

        # seq_ratings_data = seq_ratings_data[
        #     seq_ratings_data["item_ids"].apply(len) >= 5
        # ]

        result = pd.DataFrame([[]])
        for col in ["item_ids"]:
            result[col + "_mean"] = seq_ratings_data[col].apply(len).mean()
            result[col + "_min"] = seq_ratings_data[col].apply(len).min()
            result[col + "_max"] = seq_ratings_data[col].apply(len).max()
            value_counts = seq_ratings_data[col].apply(len).value_counts(normalize=True)

            # 绘制频率分布图
            plt.figure(figsize=(10, 6))
            value_counts.plot(kind='bar')
            plt.title(f"Frequency Distribution of {col}")
            plt.xlabel("Values")
            plt.ylabel("Frequency")
            plt.xticks(rotation=45)
            plt.savefig(f"tmp/{self._prefix}/seq_len_dist.png")
        percentile_20 = np.percentile(seq_ratings_data["item_ids"].apply(len), 20)
        print("20% 分位数:", percentile_20)
        print(self._prefix)
        print(result)

        if not os.path.exists(f"tmp/{self._prefix}"):
            os.makedirs(f"tmp/{self._prefix}")

        seq_ratings_data = self.to_seq_data(seq_ratings_data)
        seq_ratings_data.sample(frac=1).reset_index().to_csv(
            self.output_format_csv(), index=False, sep=","
        )

        if self.expected_num_unique_items() is not None:
            assert (
                self.expected_num_unique_items() == num_unique_items
            ), f"expected: {self.expected_num_unique_items()}, actual: {num_unique_items}"
            logging.info(f"{self.expected_num_unique_items()} unique items.")

        return num_unique_items


def get_common_preprocessors() -> Dict[
    str,
    Union[AmazonDataProcessor, MovielensDataProcessor, MovielensSyntheticDataProcessor],
]:
    ml_1m_dp = MovielensDataProcessor(  # pyre-ignore [45]
        "http://files.grouplens.org/datasets/movielens/ml-1m.zip",
        "tmp/movielens1m.zip",
        prefix="ml-1m",
        convert_timestamp=False,
        expected_num_unique_items=7175,
        expected_max_item_id=7421,
    )
    ml_20m_dp = MovielensDataProcessor(  # pyre-ignore [45]
        "http://files.grouplens.org/datasets/movielens/ml-20m.zip",
        "tmp/movielens20m.zip",
        prefix="ml-20m",
        convert_timestamp=False,
        expected_num_unique_items=26744,
        expected_max_item_id=131262,
    )
    ml_1b_dp = MovielensDataProcessor(  # pyre-ignore [45]
        "https://files.grouplens.org/datasets/movielens/ml-20mx16x32.tar",
        "tmp/movielens1b.tar",
        prefix="ml-20mx16x32",
        convert_timestamp=False,
    )
    ml_3b_dp = MovielensSyntheticDataProcessor(  # pyre-ignore [45]
        prefix="ml-3b",
        expected_num_unique_items=26743 * 32,
        expected_max_item_id=26743 * 32,
    )
    amzn_books_dp = AmazonDataProcessor(  # pyre-ignore [45]
        "http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/ratings_Books.csv",
        "tmp/ratings_Books.csv",
        prefix="amzn_books",
        expected_num_unique_items=695762,
    )
    return {
        "ml-1m": ml_1m_dp,
        "ml-20m": ml_20m_dp,
        "ml-1b": ml_1b_dp,
        "ml-3b": ml_3b_dp,
        "amzn-books": amzn_books_dp,
    }
