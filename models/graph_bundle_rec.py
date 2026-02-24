import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiModalGraphBundleRec(nn.Module):
    """
    多模态图束推荐模型（兼容单/多模态）
    修复点：
    1. 初始化时保存modals参数，兼容单/多模态输入
    2. forward函数中仅处理配置指定的模态，避免空列表stack
    3. 融合逻辑适配单模态（直接使用特征）和多模态（stack/融合）
    """

    def __init__(self, config, device):
        super(MultiModalGraphBundleRec, self).__init__()
        # 核心配置（兼容单/多模态）
        self.device = device
        self.modals = config["modals"]  # 保存模态列表：["text"]/["video"]/["text","video"]
        self.num_modals = len(self.modals)
        self.num_users = config["num_users"]
        self.num_movies = config["num_movies"]

        # 图束模型核心维度
        self.node_stalk_dim = config["node_stalk_dim"]
        self.edge_stalk_dim = config["edge_stalk_dim"]
        self.num_diffusion_layers = config["num_diffusion_layers"]
        self.fusion_type = config["fusion_type"]  # attention/concat/average

        # 模态输入维度映射
        self.modal_in_dims = config["modal_in_dims"]  # {"text":768, "video":2048}

        # 1. 模态特征投影层（每个模态单独投影到茎空间维度）
        self.modal_proj = nn.ModuleDict()
        for modal in self.modals:
            self.modal_proj[modal] = nn.Linear(
                self.modal_in_dims[modal],
                self.node_stalk_dim
            )

        # 2. 用户/物品嵌入层
        self.user_emb = nn.Embedding(self.num_users, self.node_stalk_dim)
        self.movie_emb = nn.Embedding(self.num_movies, self.node_stalk_dim)

        # 3. 图扩散层（简化版，保持核心逻辑）
        self.diffusion_layers = nn.ModuleList([
            nn.Linear(self.node_stalk_dim, self.node_stalk_dim)
            for _ in range(self.num_diffusion_layers)
        ])

        # 4. 多模态融合层（适配不同融合方式）
        if self.fusion_type == "attention" and self.num_modals > 1:
            # 仅多模态时初始化注意力层
            self.attention = nn.MultiheadAttention(
                embed_dim=self.node_stalk_dim,
                num_heads=8,
                batch_first=True
            )
        elif self.fusion_type == "concat" and self.num_modals > 1:
            # 拼接后投影回茎空间维度
            self.concat_proj = nn.Linear(
                self.node_stalk_dim * self.num_modals,
                self.node_stalk_dim
            )

        # 5. 预测层（最终评分预测）
        self.predictor = nn.Sequential(
            nn.Linear(self.node_stalk_dim * 2, self.node_stalk_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.node_stalk_dim, 1)
        )

        # 初始化参数
        self._init_weights()

    def _init_weights(self):
        """参数初始化，避免梯度消失/爆炸"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.xavier_uniform_(m.weight)

    def _fusion_modal_feats(self, modal_feats):
        """
        模态特征融合（核心：适配单/多模态）
        Args:
            modal_feats: list of tensor, 每个元素是[batch_size, node_stalk_dim]
        Returns:
            fused_feat: [batch_size, node_stalk_dim]
        """
        # 单模态：直接返回特征
        if self.num_modals == 1:
            return modal_feats[0]

        # 多模态：根据融合方式处理
        if self.fusion_type == "attention":
            # Attention融合：[batch_size, num_modals, node_stalk_dim]
            feat_tensor = torch.stack(modal_feats, dim=1)
            attn_output, _ = self.attention(feat_tensor, feat_tensor, feat_tensor)
            fused_feat = torch.mean(attn_output, dim=1)  # 平均所有模态的注意力输出
        elif self.fusion_type == "concat":
            # 拼接融合：[batch_size, num_modals*node_stalk_dim] → [batch_size, node_stalk_dim]
            feat_concat = torch.cat(modal_feats, dim=1)
            fused_feat = self.concat_proj(feat_concat)
        elif self.fusion_type == "average":
            # 平均融合：简单平均所有模态特征
            feat_stack = torch.stack(modal_feats, dim=1)
            fused_feat = torch.mean(feat_stack, dim=1)
        else:
            # 默认平均融合
            feat_stack = torch.stack(modal_feats, dim=1)
            fused_feat = torch.mean(feat_stack, dim=1)

        return fused_feat

    def forward(self, user_ids, movie_ids, movie_feats):
        """
        前向传播（核心：兼容单/多模态）
        Args:
            user_ids: [batch_size]  用户ID
            movie_ids: [batch_size]  电影ID
            movie_feats: dict  模态特征字典，key=modal, value=[num_movies, in_dim]
        Returns:
            pred_ratings: [batch_size, 1]  预测评分
        """
        # 1. 获取用户/电影基础嵌入
        user_emb = self.user_emb(user_ids)  # [batch_size, node_stalk_dim]
        movie_base_emb = self.movie_emb(movie_ids)  # [batch_size, node_stalk_dim]

        # 2. 处理模态特征（仅处理配置指定的模态，避免空列表）
        modal_feats = []
        for modal in self.modals:
            # 获取当前批次的模态特征：[batch_size, in_dim]
            raw_feat = movie_feats[modal][movie_ids]
            # 投影到茎空间维度：[batch_size, node_stalk_dim]
            proj_feat = self.modal_proj[modal](raw_feat)
            modal_feats.append(proj_feat)

        # 3. 模态特征融合（兼容单/多模态）
        fused_movie_feat = self._fusion_modal_feats(modal_feats)

        # 4. 图扩散（融合电影基础嵌入和模态特征）
        movie_emb = movie_base_emb + fused_movie_feat
        for layer in self.diffusion_layers:
            movie_emb = F.relu(layer(movie_emb))

        # 5. 评分预测（用户嵌入 + 电影嵌入）
        concat_emb = torch.cat([user_emb, movie_emb], dim=1)
        pred_ratings = self.predictor(concat_emb)

        return pred_ratings


# 测试代码（验证单/多模态是否正常）
if __name__ == "__main__":
    # 测试配置
    test_config = {
        "modals": ["text","video"],  # 可切换为["video"]/["text","video"]测试
        "modal_in_dims": {"text": 768, "video": 2048},
        "num_users": 6040,
        "num_movies": 3706,
        "node_stalk_dim": 64,
        "edge_stalk_dim": 64,
        "num_diffusion_layers": 2,
        "fusion_type": "attention"
    }
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 初始化模型
    model = MultiModalGraphBundleRec(test_config, device).to(device)
    print(f"✅ Model initialization successful (modal:{test_config['modals']}）")

    # 构造测试数据
    batch_size = 32
    user_ids = torch.randint(0, 6040, (batch_size,)).to(device)
    movie_ids = torch.randint(0, 3706, (batch_size,)).to(device)

    # 构造模态特征（模拟真实特征维度）
    movie_feats = {}
    if "text" in test_config["modals"]:
        movie_feats["text"] = torch.randn(3706, 768).to(device)
    if "video" in test_config["modals"]:
        movie_feats["video"] = torch.randn(3706, 2048).to(device)

    # 前向传播测试
    with torch.no_grad():
        pred = model(user_ids, movie_ids, movie_feats)

    print(f"✅ Forward propagation successful! Output dimension:{pred.shape}（expected:[{batch_size}, 1]）")
    print(f"Total number of model parameters:{sum(p.numel() for p in model.parameters()):,}")
