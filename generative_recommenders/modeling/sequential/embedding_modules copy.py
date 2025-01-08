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

import pickle

import torch.nn as nn
import torch.nn.functional as F
from generative_recommenders.modeling.initialization import truncated_normal

class DomainGatingEmbeddingModule(EmbeddingModule):

    def __init__(
        self,
        num_items: int,
        item_embedding_dim: int,
        token_embedding_map_path: str = None,
        text_requires_grad: bool = False
    ) -> None:
        """
        初始化 DomainGatingEmbeddingModule
        :param num_items: 项目总数
        :param item_embedding_dim: ID 嵌入的维度
        :param token_embedding_map_path: 保存的 token_embedding_map 的文件路径（如果为 None，则不加载文本嵌入）
        """
        super().__init__()
        self._item_embedding_dim: int = item_embedding_dim
        self._text_embedding_dim: int = 0  # 默认为 0
        self.token_embedding_map_path = token_embedding_map_path

        if token_embedding_map_path is not None:
            with open(token_embedding_map_path, "rb") as f:
                token_embedding_map = pickle.load(f)

            self._text_embedding_dim = len(next(iter(token_embedding_map.values())))
            self._item_embedding_dim = self._text_embedding_dim
            token_embedding_tensor = torch.zeros(
                (num_items + 1, self._text_embedding_dim)
            )
            for token_id, embedding in token_embedding_map.items():
                token_embedding_tensor[token_id] = torch.tensor(embedding)

            self._text_emb = nn.Embedding.from_pretrained(
                token_embedding_tensor, freeze=not text_requires_grad, padding_idx=0
            )
            self._item_emb = torch.nn.Embedding(
                num_items + 1, self._item_embedding_dim, padding_idx=0
            )
            self.reset_params()
        

        # Domain Gating 网络
        self.gating_network = nn.Sequential(
            nn.Linear(item_embedding_dim + text_embedding_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 2),  # 输出两个权重值
            nn.Softmax(dim=-1)  # 确保权重和为1
        )


    def debug_str(self) -> str:
        """
        返回调试信息
        """
        return f"domain_gating_item_emb_d{self._item_embedding_dim}_text_emb_d{self._text_embedding_dim}"
    def reset_params(self) -> None:
        """
        重置参数
        """
        for name, params in self.named_parameters():
            if "_item_emb" in name:
                truncated_normal(params, mean=0.0, std=0.02)
                print(
                    f"Initialize {name} as truncated normal: {params.data.size()} params"
                )
            else:
                print(f"Skipping initializing params {name} - not configured")

    def get_item_embeddings(self, item_ids: torch.Tensor) -> torch.Tensor:
        """
        获取项目的动态加权嵌入
        :param item_ids: 项目 ID 的张量
        :return: 动态加权的嵌入
        """
        # 获取 item 和 text 的嵌入
        item_embeddings = self._item_emb(item_ids)
        text_embeddings = self._text_emb(item_ids)

        # 拼接两个嵌入，用于计算 gating 权重
        concatenated_embeddings = torch.cat([item_embeddings, text_embeddings], dim=-1)

        # 计算 gating 权重
        gating_weights = self.gating_network(concatenated_embeddings)  # 输出形状为 (batch_size, 2)

        # 对 item 和 text 的嵌入进行加权融合
        weighted_item_emb = gating_weights[:, 0:1] * item_embeddings  # 权重对 item_embedding 的加权
        weighted_text_emb = gating_weights[:, 1:2] * text_embeddings  # 权重对 text_embedding 的加权

        # 返回融合后的嵌入
        return weighted_item_emb + weighted_text_emb

    @property
    def item_embedding_dim(self) -> int:
        """
        返回总嵌入维度
        """
        return self._text_embedding_dim
