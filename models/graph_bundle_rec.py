import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiModalGraphBundleRec(nn.Module):
    """
    Multimodal graph bundle recommendation model (compatible with single/multimodal inputs)
    1. Save modals parameters during initialization to ensure compatibility with single/multimodal inputs.
    2. Only process the configured modalities in the forward function to avoid empty stack lists.
    3. Adapt the fusion logic to single-modal (direct feature usage) and multimodal (stack/fusion) inputs.
    """

    def __init__(self, config, device):
        super(MultiModalGraphBundleRec, self).__init__()
        # Core configuration (compatible with single/multimodal)
        self.device = device
        self.modals = config["modals"]
        self.num_modals = len(self.modals)
        self.num_users = config["num_users"]
        self.num_movies = config["num_movies"]

        # Core dimensions of the diagram model
        self.node_stalk_dim = config["node_stalk_dim"]
        self.edge_stalk_dim = config["edge_stalk_dim"]
        self.num_diffusion_layers = config["num_diffusion_layers"]
        self.fusion_type = config["fusion_type"]  # attention/concat/average

        # Modal input dimension mapping
        self.modal_in_dims = config["modal_in_dims"]  # {"text":768, "video":2048}

        # 1. Modal feature projection layer (each modality is projected individually to the stem space dimension)
        self.modal_proj = nn.ModuleDict()
        for modal in self.modals:
            self.modal_proj[modal] = nn.Linear(
                self.modal_in_dims[modal],
                self.node_stalk_dim
            )

        # 2. User/Item Embedding Layer
        self.user_emb = nn.Embedding(self.num_users, self.node_stalk_dim)
        self.movie_emb = nn.Embedding(self.num_movies, self.node_stalk_dim)

        # 3. Graph diffusion layer
        self.diffusion_layers = nn.ModuleList([
            nn.Linear(self.node_stalk_dim, self.node_stalk_dim)
            for _ in range(self.num_diffusion_layers)
        ])

        # 4. Multimodal fusion layer (adapts to different fusion methods)
        if self.fusion_type == "attention" and self.num_modals > 1:
            # Initialize attention layer only in multimodal mode
            self.attention = nn.MultiheadAttention(
                embed_dim=self.node_stalk_dim,
                num_heads=8,
                batch_first=True
            )
        elif self.fusion_type == "concat" and self.num_modals > 1:
            # After splicing, the projection back to the stem space dimension
            self.concat_proj = nn.Linear(
                self.node_stalk_dim * self.num_modals,
                self.node_stalk_dim
            )

        # 5. Prediction layer (final score prediction)
        self.predictor = nn.Sequential(
            nn.Linear(self.node_stalk_dim * 2, self.node_stalk_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.node_stalk_dim, 1)
        )

        # Initialization parameters
        self._init_weights()

    def _init_weights(self):
        """Parameter initialization to avoid gradient vanishing/exploding"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.xavier_uniform_(m.weight)

    def _fusion_modal_feats(self, modal_feats):
        """
        Modal feature fusion (core: adaptation to single/multimodal modes)
        Args:
            modal_feats: list of tensor, [batch_size, node_stalk_dim]
        Returns:
            fused_feat: [batch_size, node_stalk_dim]
        """
        # Single-modal: Directly returns features
        if self.num_modals == 1:
            return modal_feats[0]

        # Multimodal: Processed according to the fusion method
        if self.fusion_type == "attention":
            # Attention：[batch_size, num_modals, node_stalk_dim]
            feat_tensor = torch.stack(modal_feats, dim=1)
            attn_output, _ = self.attention(feat_tensor, feat_tensor, feat_tensor)
            fused_feat = torch.mean(attn_output, dim=1)  # Average attention output across all modalities
        elif self.fusion_type == "concat":
            # concat：[batch_size, num_modals*node_stalk_dim] → [batch_size, node_stalk_dim]
            feat_concat = torch.cat(modal_feats, dim=1)
            fused_feat = self.concat_proj(feat_concat)
        elif self.fusion_type == "average":
            # average：Simple averaging of all modal features
            feat_stack = torch.stack(modal_feats, dim=1)
            fused_feat = torch.mean(feat_stack, dim=1)
        else:
            # Default average fusion
            feat_stack = torch.stack(modal_feats, dim=1)
            fused_feat = torch.mean(feat_stack, dim=1)

        return fused_feat

    def forward(self, user_ids, movie_ids, movie_feats):
        """
        Forward propagation (core: compatible with single/multimodal modes)
        Args:
            user_ids: [batch_size]
            movie_ids: [batch_size]
            movie_feats: dict  key=modal, value=[num_movies, in_dim]
        Returns:
            pred_ratings: [batch_size, 1]
        """
        # 1. Get user/movie basic embedding
        user_emb = self.user_emb(user_ids)  # [batch_size, node_stalk_dim]
        movie_base_emb = self.movie_emb(movie_ids)  # [batch_size, node_stalk_dim]

        # 2. Process modal features (only process the modalities specified in the configuration, avoiding empty lists).
        modal_feats = []
        for modal in self.modals:
            # Obtain modal features of the current batch：[batch_size, in_dim]
            raw_feat = movie_feats[modal][movie_ids]
            # Projected onto stem space dimension：[batch_size, node_stalk_dim]
            proj_feat = self.modal_proj[modal](raw_feat)
            modal_feats.append(proj_feat)

        # 3. Modal feature fusion (compatible with single/multimodal)
        fused_movie_feat = self._fusion_modal_feats(modal_feats)

        # 4. Graph diffusion (integrating film-based embeddings and modal features)
        movie_emb = movie_base_emb + fused_movie_feat
        for layer in self.diffusion_layers:
            movie_emb = F.relu(layer(movie_emb))

        # 5. Rating prediction (user embedding + movie embedding)
        concat_emb = torch.cat([user_emb, movie_emb], dim=1)
        pred_ratings = self.predictor(concat_emb)

        return pred_ratings


# Test code (to verify whether single/multimodal operation is normal)
if __name__ == "__main__":
    # Test configuration
    test_config = {
        "modals": ["text","video"],
        "modal_in_dims": {"text": 768, "video": 2048},
        "num_users": 6040,
        "num_movies": 3706,
        "node_stalk_dim": 64,
        "edge_stalk_dim": 64,
        "num_diffusion_layers": 2,
        "fusion_type": "attention"
    }
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Initialize the model
    model = MultiModalGraphBundleRec(test_config, device).to(device)
    print(f"✅ Model initialization successful (modal:{test_config['modals']}）")

    # Construct test data
    batch_size = 32
    user_ids = torch.randint(0, 6040, (batch_size,)).to(device)
    movie_ids = torch.randint(0, 3706, (batch_size,)).to(device)

    # Construct modal features
    movie_feats = {}
    if "text" in test_config["modals"]:
        movie_feats["text"] = torch.randn(3706, 768).to(device)
    if "video" in test_config["modals"]:
        movie_feats["video"] = torch.randn(3706, 2048).to(device)

    # Forward propagation test
    with torch.no_grad():
        pred = model(user_ids, movie_ids, movie_feats)

    print(f"✅ Forward propagation successful! Output dimension:{pred.shape}（expected:[{batch_size}, 1]）")
    print(f"Total number of model parameters:{sum(p.numel() for p in model.parameters()):,}")
