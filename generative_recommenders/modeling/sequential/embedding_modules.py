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

from transformers import AutoTokenizer, AutoModel
import torch
from generative_recommenders.modeling.initialization import truncated_normal


class LocalEmbeddingWithTextModule(EmbeddingModule):

    def __init__(
        self,
        num_items: int,
        item_embedding_dim: int,
        text_embedding_dim: int = 768,
        text_model_name: str = "bert-base-uncased",
    ) -> None:
        """
        初始化 LocalEmbeddingWithTextModule
        :param num_items: 项目总数
        :param item_embedding_dim: ID 嵌入的维度
        :param text_embedding_dim: 文本嵌入的维度
        :param text_model_name: 文本嵌入模型的名称（默认为 "bert-base-uncased"）
        """
        super().__init__()
        self._item_embedding_dim: int = item_embedding_dim
        self._text_embedding_dim: int = text_embedding_dim

        # 初始化项目 ID 嵌入
        self._item_emb = torch.nn.Embedding(
            num_items + 1, item_embedding_dim, padding_idx=0
        )
        self.reset_params()

        # 初始化文本嵌入模型
        self.tokenizer = AutoTokenizer.from_pretrained(text_model_name)
        self.text_encoder = AutoModel.from_pretrained(text_model_name)
        self.text_projection = torch.nn.Linear(self.text_encoder.config.hidden_size, text_embedding_dim)

    def debug_str(self) -> str:
        return f"local_emb_d{self._item_embedding_dim}_text_d{self._text_embedding_dim}"

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
        self, item_ids: torch.Tensor, item_texts: list[str]
    ) -> torch.Tensor:
        """
        获取 item 的嵌入向量（ID 嵌入 + 文本嵌入）
        :param item_ids: 项目 ID 的张量
        :param item_texts: 项目对应的文本列表
        :return: 拼接后的嵌入向量
        """
        # ID 嵌入
        item_emb = self._item_emb(item_ids)

        # 文本嵌入
        inputs = self.tokenizer(
            item_texts, padding=True, truncation=True, return_tensors="pt"
        ).to(item_emb.device)
        text_emb = self.text_encoder(**inputs).last_hidden_state.mean(dim=1)  # 平均池化
        text_emb = self.text_projection(text_emb)

        # 拼接 ID 嵌入和文本嵌入
        combined_emb = torch.cat([item_emb, text_emb], dim=-1)  # [batch_size, embedding_dim * 2]
        return combined_emb

    @property
    def item_embedding_dim(self) -> int:
        """
        返回拼接后的总嵌入维度
        """
        return self._item_embedding_dim + self._text_embedding_dim
