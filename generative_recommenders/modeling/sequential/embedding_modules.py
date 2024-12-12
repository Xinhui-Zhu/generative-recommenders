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

import torch

from generative_recommenders.modeling.initialization import truncated_normal


class EmbeddingModule(torch.nn.Module):

    @abc.abstractmethod
    def debug_str(self) -> str:
        pass

    @abc.abstractmethod
    def get_item_embeddings(self, item_ids: torch.Tensor) -> torch.Tensor:
        pass

    @property
    @abc.abstractmethod
    def item_embedding_dim(self) -> int:
        pass


class LocalEmbeddingModule(EmbeddingModule):

    def __init__(
        self,
        num_items: int,
        item_embedding_dim: int,
    ) -> None:
        super().__init__()

        self._item_embedding_dim: int = item_embedding_dim
        self._item_emb = torch.nn.Embedding(
            num_items + 1, item_embedding_dim, padding_idx=0
        )
        self.reset_params()

    def debug_str(self) -> str:
        return f"local_emb_d{self._item_embedding_dim}"

    def reset_params(self) -> None:
        for name, params in self.named_parameters():
            if "_item_emb" in name:
                print(
                    f"Initialize {name} as truncated normal: {params.data.size()} params"
                )
                truncated_normal(params, mean=0.0, std=0.02)
            else:
                print(f"Skipping initializing params {name} - not configured")

    def get_item_embeddings(self, item_ids: torch.Tensor) -> torch.Tensor:
        return self._item_emb(item_ids)

    @property
    def item_embedding_dim(self) -> int:
        return self._item_embedding_dim


class CategoricalEmbeddingModule(EmbeddingModule):

    def __init__(
        self,
        num_items: int,
        item_embedding_dim: int,
        item_id_to_category_id: torch.Tensor,
    ) -> None:
        super().__init__()

        self._item_embedding_dim: int = item_embedding_dim
        self._item_emb: torch.nn.Embedding = torch.nn.Embedding(
            num_items + 1, item_embedding_dim, padding_idx=0
        )
        self.register_buffer("_item_id_to_category_id", item_id_to_category_id)
        self.reset_params()

    def debug_str(self) -> str:
        return f"cat_emb_d{self._item_embedding_dim}"

    def reset_params(self) -> None:
        for name, params in self.named_parameters():
            if "_item_emb" in name:
                print(
                    f"Initialize {name} as truncated normal: {params.data.size()} params"
                )
                truncated_normal(params, mean=0.0, std=0.02)
            else:
                print(f"Skipping initializing params {name} - not configured")

    def get_item_embeddings(self, item_ids: torch.Tensor) -> torch.Tensor:
        item_ids = self._item_id_to_category_id[(item_ids - 1).clamp(min=0)] + 1
        return self._item_emb(item_ids)

    @property
    def item_embedding_dim(self) -> int:
        return self._item_embedding_dim

import pickle

class LocalEmbeddingWithTextModule(EmbeddingModule):

    def __init__(
        self,
        num_items: int,
        item_embedding_dim: int,
        token_embedding_map_path: str,
    ) -> None:
        """
        初始化 LocalEmbeddingWithTextModule
        :param num_items: 项目总数
        :param item_embedding_dim: ID 嵌入的维度
        :param token_embedding_map_path: 保存的 token_embedding_map 的文件路径
        """
        super().__init__()
        self._item_embedding_dim: int = item_embedding_dim

        # 初始化项目 ID 嵌入
        self._item_emb = torch.nn.Embedding(
            num_items + 1, item_embedding_dim, padding_idx=0
        )
        self.reset_params()

        # 加载预生成的文本嵌入
        with open(token_embedding_map_path, "rb") as f:
            token_embedding_map = pickle.load(f)

        # 将 token_embedding_map 转换为张量
        token_embedding_tensor = torch.zeros(
            (num_items + 1, len(next(iter(token_embedding_map.values()))))
        )
        for token_id, embedding in token_embedding_map.items():
            token_embedding_tensor[token_id] = torch.tensor(embedding)

        # 创建一个固定的嵌入层用于文本嵌入
        self.text_embedding = torch.nn.Embedding.from_pretrained(
            token_embedding_tensor, freeze=True, padding_idx=0
        )

    def debug_str(self) -> str:
        return f"local_emb_d{self._item_embedding_dim}_text_preloaded"

    def reset_params(self) -> None:
        for name, params in self.named_parameters():
            if "_item_emb" in name:
                print(
                    f"Initialize {name} as truncated normal: {params.data.size()} params"
                )
                truncated_normal(params, mean=0.0, std=0.02)
            else:
                print(f"Skipping initializing params {name} - not configured")

    def get_item_embeddings(
        self, item_ids: torch.Tensor
    ) -> torch.Tensor:
        """
        获取 item 的嵌入向量（ID 嵌入 + 文本嵌入）
        :param item_ids: 项目 ID 的张量
        :return: 拼接后的嵌入向量
        """
        # ID 嵌入
        item_emb = self._item_emb(item_ids)

        # 文本嵌入
        text_emb = self.text_embedding(item_ids)

        # 拼接 ID 嵌入和文本嵌入
        combined_emb = torch.cat([item_emb, text_emb], dim=-1)  # [batch_size, embedding_dim * 2]
        return combined_emb

    @property
    def item_embedding_dim(self) -> int:
        """
        返回拼接后的总嵌入维度
        """
        return self._item_embedding_dim + self.text_embedding.embedding_dim
