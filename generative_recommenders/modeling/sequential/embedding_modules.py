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
import torch.nn as nn

from generative_recommenders.modeling.initialization import truncated_normal


class EmbeddingModule(nn.Module):

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

class LocalEmbeddingModule(EmbeddingModule):

    def __init__(
        self,
        num_items: int,
        item_embedding_dim: int,
        token_embedding_map_path: str = None,
        text_freeze: bool = True,
        type_name: str = "only_item"
    ) -> None:
        """
        初始化 LocalEmbeddingModule
        :param num_items: 项目总数
        :param item_embedding_dim: ID 嵌入的维度
        :param token_embedding_map_path: 保存的 token_embedding_map 的文件路径（如果为 None，则不加载文本嵌入）
        """
        super().__init__()
        self._item_embedding_dim: int = item_embedding_dim
        self.token_embedding_map_path = token_embedding_map_path
        self.type_name = type_name
        self.num_items = num_items

        if self.type_name == "item_concat_text":
            token_embedding_tensor = self.get_pretrained_text_embedding()
            # 创建一个包含 ID 嵌入和文本嵌入的完整嵌入层
            total_embedding_dim = item_embedding_dim + self._text_embedding_dim
            self._item_emb = nn.Embedding(
                num_items + 1, total_embedding_dim, padding_idx=0
            )

            # 初始化随机部分并加载文本嵌入
            self.reset_params()
            with torch.no_grad():
                self._item_emb.weight[:, item_embedding_dim:] = token_embedding_tensor
                self._item_emb.weight[:, item_embedding_dim:].requires_grad = not text_freeze
        
        elif "gating" in self.type_name:
            token_embedding_tensor = self.get_pretrained_text_embedding()
            self._text_emb = nn.Embedding.from_pretrained(
                token_embedding_tensor, freeze=text_freeze, padding_idx=0
            )
            self._item_embedding_dim = self._text_embedding_dim
            self._item_emb = nn.Embedding(
                num_items + 1, self._item_embedding_dim, padding_idx=0
            )
            self.reset_params()
            if self.type_name == "domain_gating":
                self.domain_embedding = torch.nn.Parameter(torch.randn(self._item_embedding_dim) * 0.02)
                # self.text_domain_embedding = torch.nn.Parameter(torch.randn(self._text_embedding_dim) * 0.02)
        
        elif self.type_name == "only_item":
            # 仅创建随机初始化的 ID 嵌入层
            self._item_emb = nn.Embedding(
                num_items + 1, item_embedding_dim, padding_idx=0
            )
            self.reset_params()

    def debug_str(self) -> str:
        """
        返回调试信息
        """
        if self.type_name == "item_concat_text":
            return f"item_emb_d{self._item_embedding_dim}_concat_preloaded_text_emb_d{self._text_embedding_dim}"
        elif "gating" in self.type_name:
            return f"{self.type_name}_item_emb_d{self._item_embedding_dim}_text_emb_d{self._text_embedding_dim}"
        elif self.type_name == "only_item":
            return f"item_emb_d{self._item_embedding_dim}"

    def reset_params(self) -> None:
        """
        重置参数
        """
        for name, params in self.named_parameters():
            if "_item_emb" in name:
                # 仅初始化随机部分
                if self.type_name == "item_concat_text":
                    params.data[:, :self._item_embedding_dim] = truncated_normal(
                        params.data[:, :self._item_embedding_dim], mean=0.0, std=0.02
                    )
                    print(
                        f"Initialize {name} (random part) as truncated normal: {params.data[:, :self._item_embedding_dim].size()}"
                    )
                else:
                    truncated_normal(params, mean=0.0, std=0.02)
                    print(
                        f"Initialize {name} as truncated normal: {params.data.size()} params"
                    )
            else:
                print(f"Skipping initializing params {name} - not configured")

    def get_item_embeddings(self, item_ids: torch.Tensor) -> torch.Tensor:
        """
        获取项目+文本嵌入，为不影响其他调用该方法的代码，故没有改名，其实也可以认为这个方法就是通常的forward()方法
        :param item_ids: 项目 ID 的张量
        :return: 项目嵌入
        """
        
        if self.type_name == "domain_gating":
            # 获取 item 和 text 的嵌入
            item_embeddings = self._item_emb(item_ids)
            text_embeddings = self._text_emb(item_ids)

            # 计算权重
            item_weight = torch.exp(torch.sum(self.domain_embedding * item_embeddings, dim=-1, keepdim=True))
            text_weight = torch.exp(torch.sum(self.domain_embedding * text_embeddings, dim=-1, keepdim=True))
            total_weight = item_weight + text_weight

            item_weight = item_weight / total_weight
            text_weight = text_weight / total_weight

            # 加权融合
            weighted_item_emb = item_weight * item_embeddings
            weighted_text_emb = text_weight * text_embeddings

            return weighted_item_emb + weighted_text_emb
        elif self.type_name == "mean_gating":
            item_embeddings = self._item_emb(item_ids)
            text_embeddings = self._text_emb(item_ids)
            return (item_embeddings + text_embeddings) / 2
        else:
            return self._item_emb(item_ids)

    @property
    def item_embedding_dim(self) -> int:
        """
        返回总嵌入维度，为不影响其他调用该方法的代码，故没有改名
        """
        if self.type_name == "item_concat_text":
            return self._item_embedding_dim + self._text_embedding_dim
        else:
            return self._item_embedding_dim

    def get_pretrained_text_embedding(self) -> torch.Tensor:
        # 加载预生成的文本嵌入
        with open(self.token_embedding_map_path, "rb") as f:
            token_embedding_map = pickle.load(f)

        # 将 token_embedding_map 转换为张量
        self._text_embedding_dim = len(next(iter(token_embedding_map.values())))
        token_embedding_tensor = torch.zeros(
            (self.num_items + 1, self._text_embedding_dim)
        )
        for token_id, embedding in token_embedding_map.items():
            token_embedding_tensor[token_id] = torch.tensor(embedding)
        print("token_embedding_tensor.shape =", token_embedding_tensor.shape)

        # zero_rows 是一个布尔张量，表示每一行是否全为0
        zero_rows = (token_embedding_tensor == 0).all(dim=1)

        # 如果要检查是否存在任何全0的行:
        any_zero_rows = zero_rows.any().item()
        print("存在全0行吗？", any_zero_rows)

        # 如果需要找出具体哪些行是全0行：
        zero_row_indices = torch.where(zero_rows)[0]
        print("全0行的行索引:", zero_row_indices)
        print("全0行的行数量:", len(zero_row_indices))

        return token_embedding_tensor
