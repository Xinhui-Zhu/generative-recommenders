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

class LocalEmbeddingModule(EmbeddingModule):

    def __init__(
        self,
        num_items: int,
        item_embedding_dim: int,
        token_embedding_map_path: str = None,
        text_requires_grad: bool = False
    ) -> None:
        """
        初始化 LocalEmbeddingModule
        :param num_items: 项目总数
        :param item_embedding_dim: ID 嵌入的维度
        :param token_embedding_map_path: 保存的 token_embedding_map 的文件路径（如果为 None，则不加载文本嵌入）
        """
        super().__init__()
        self._item_embedding_dim: int = item_embedding_dim
        self._text_embedding_dim: int = 0  # 默认为 0
        self.token_embedding_map_path = token_embedding_map_path

        # 检查是否需要加载文本嵌入
        if token_embedding_map_path is not None:
            # 加载预生成的文本嵌入
            with open(token_embedding_map_path, "rb") as f:
                token_embedding_map = pickle.load(f)

            # 将 token_embedding_map 转换为张量
            self._text_embedding_dim = len(next(iter(token_embedding_map.values())))
            token_embedding_tensor = torch.zeros(
                (num_items + 1, self._text_embedding_dim)
            )
            for token_id, embedding in token_embedding_map.items():
                token_embedding_tensor[token_id] = torch.tensor(embedding)

            # 创建一个包含 ID 嵌入和文本嵌入的完整嵌入层
            total_embedding_dim = item_embedding_dim + self._text_embedding_dim
            self._item_emb = torch.nn.Embedding(
                num_items + 1, total_embedding_dim, padding_idx=0
            )

            # 初始化随机部分并加载文本嵌入
            self.reset_params()
            with torch.no_grad():
                self._item_emb.weight[:, item_embedding_dim:] = token_embedding_tensor
                if not text_requires_grad:
                    self._item_emb.weight[:, item_embedding_dim:].requires_grad = False
        
        else:
            # 仅创建随机初始化的 ID 嵌入层
            self._item_emb = torch.nn.Embedding(
                num_items + 1, item_embedding_dim, padding_idx=0
            )
            self.reset_params()

    def debug_str(self) -> str:
        """
        返回调试信息
        """
        if self.token_embedding_map_path:
            return f"local_emb_d{self._item_embedding_dim}_text_preloaded"
        else:
            return f"local_emb_d{self._item_embedding_dim}"

    def reset_params(self) -> None:
        """
        重置参数
        """
        for name, params in self.named_parameters():
            if "_item_emb" in name:
                # 仅初始化随机部分
                if self.token_embedding_map_path:
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
        获取项目嵌入
        :param item_ids: 项目 ID 的张量
        :return: 项目嵌入
        """
        return self._item_emb(item_ids)

    @property
    def item_embedding_dim(self) -> int:
        """
        返回总嵌入维度
        """
        return self._item_embedding_dim + self._text_embedding_dim
