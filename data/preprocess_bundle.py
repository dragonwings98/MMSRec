import os
import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def load_graph_bundle_data(config):
    """
    适配图束理论的数据加载 + 训练/验证/测试集划分（8:1:1）
    核心修复：
    1. 过滤只有1条评分的用户，解决stratify分层采样报错
    2. 支持小数据量采样（调优提速）
    3. 仅加载config中指定的模态特征（兼容单/多模态）
    """
    # 1. 路径解析
    data_path = config["data_path"]
    feat_path = config["feat_path"]
    modals = config["modals"]  # 当前配置指定的模态：["text"]/["video"]/["text","video"]
    modal_in_dims = config["modal_in_dims"]  # 模态输入维度：{"text":768, "video":2048}

    # 2. 加载基础数据
    # 评分数据：user_id, movie_id, rating, timestamp
    ratings = pd.read_csv(
        os.path.join(data_path, "ratings.dat"),
        sep="::", names=["user_id", "movie_id", "rating", "timestamp"],
        engine="python", encoding="latin-1"
    )
    # 电影数据：movie_id, title, genres
    movies = pd.read_csv(
        os.path.join(data_path, "movies.dat"),
        sep="::", names=["movie_id", "title", "genres"],
        engine="python", encoding="latin-1"
    )

    # ========== 核心修复1：小数据量采样（调优提速） ==========
    # 支持采样比例配置（默认全量，调优时用20%）
    sample_frac = config.get("sample_frac", 1.0)  # 从config读取采样比例
    if sample_frac < 1.0:
        print(f"📌 采样{sample_frac * 100}%数据用于调优（提速）")
        # 按用户采样（保证每个用户至少有1条记录）
        user_ids = ratings["user_id"].unique()
        sampled_users = np.random.choice(user_ids, size=int(len(user_ids) * sample_frac), replace=False)
        ratings = ratings[ratings["user_id"].isin(sampled_users)]

    # ========== 核心修复2：过滤只有1条评分的用户 ==========
    # 统计每个用户的评分数量
    user_rating_count = ratings["user_id"].value_counts()
    # 保留至少有2条评分的用户（满足stratify分层要求）
    valid_users = user_rating_count[user_rating_count >= 2].index
    ratings = ratings[ratings["user_id"].isin(valid_users)]
    print(f"📌 过滤后保留用户数：{len(valid_users)}（原用户数：{len(user_rating_count)}）")
    print(f"📌 过滤后评分样本数：{len(ratings)}")

    # 3. 构建ID映射（将原始ID转为连续索引）
    user_ids = ratings["user_id"].unique()
    movie_ids = ratings["movie_id"].unique()
    user2idx = {uid: idx for idx, uid in enumerate(user_ids)}
    movie2idx = {mid: idx for idx, mid in enumerate(movie_ids)}
    num_users = len(user2idx)
    num_movies = len(movie2idx)

    # 4. 转换为连续ID（方便模型嵌入层处理）
    ratings["user_idx"] = ratings["user_id"].map(user2idx)
    ratings["movie_idx"] = ratings["movie_id"].map(movie2idx)

    # 5. 分层划分数据集（核心：按用户分层，保证每个用户的评分分布）
    # 第一步：划分训练集（80%）和临时集（20%）
    train_ratings, temp_ratings = train_test_split(
        ratings, test_size=0.2, random_state=42, stratify=ratings["user_idx"]
    )
    # 第二步：划分验证集（10%）和测试集（10%）
    val_ratings, test_ratings = train_test_split(
        temp_ratings, test_size=0.5, random_state=42, stratify=temp_ratings["user_idx"]
    )

    # 6. 构建训练/验证/测试样本（user-movie对 + 真实评分）
    def build_samples(ratings_df):
        user_movie_pairs = np.array(list(zip(ratings_df["user_idx"], ratings_df["movie_idx"])))
        true_ratings = np.array(ratings_df["rating"], dtype=np.float32)
        return torch.tensor(user_movie_pairs, dtype=torch.long), torch.tensor(true_ratings, dtype=torch.float32)

    train_pairs, train_ratings_arr = build_samples(train_ratings)
    val_pairs, val_ratings_arr = build_samples(val_ratings)
    test_pairs, test_ratings_arr = build_samples(test_ratings)

    # 7. 加载指定的模态特征（核心修复：仅加载config中指定的模态）
    movie_feats = {}  # 最终返回的特征字典，仅包含当前需要的模态
    raw_movie_ids = movies["movie_id"].tolist()
    raw_movie2feat_idx = {mid: idx for idx, mid in enumerate(raw_movie_ids)}  # 原始电影ID到特征索引的映射

    for modal in modals:  # 仅遍历当前配置指定的模态
        # 拼接特征文件路径：feat_path/movie_{modal}_emb.pt
        feat_file = os.path.join(feat_path, f"movie_{modal}_emb.pt")

        # 检查特征文件是否存在（避免文件缺失报错）
        if not os.path.exists(feat_file):
            raise FileNotFoundError(f"图束模型需要{modal}特征文件，但未找到：{feat_file}")

        # 加载模态特征
        modal_feat = torch.load(feat_file, map_location="cpu")
        print(f"✅ 加载{modal}特征（图束适配）：原始维度 {modal_feat.shape}")

        # 特征对齐：将原始特征对齐到当前数据集的电影ID（处理缺失电影）
        aligned_feat = []
        for mid in movie_ids:  # 遍历当前数据集的所有电影ID
            if mid in raw_movie2feat_idx:
                # 存在该电影的特征，直接取
                aligned_feat.append(modal_feat[raw_movie2feat_idx[mid]])
            else:
                # 无特征的电影，用全0填充（避免维度不匹配）
                aligned_feat.append(torch.zeros(modal_in_dims[modal]))

        # 转换为tensor并保存到字典
        movie_feats[modal] = torch.stack(aligned_feat)
        print(f"✅ 对齐{modal}特征：最终维度 {movie_feats[modal].shape}（电影数：{num_movies}，维度：{modal_in_dims[modal]}）")

    # 8. 封装返回数据（仅包含当前需要的模态特征，无冗余）
    data_dict = {
        # 基础信息
        "num_users": num_users,
        "num_movies": num_movies,
        "user2idx": user2idx,
        "movie2idx": movie2idx,
        # 模态相关（核心：仅返回当前配置的模态）
        "movie_feats": movie_feats,
        "modals": modals,
        "modal_in_dims": modal_in_dims,
        # 训练集
        "train_pairs": train_pairs,
        "train_ratings": train_ratings_arr,
        # 验证集
        "val_pairs": val_pairs,
        "val_ratings": val_ratings_arr,
        # 测试集
        "test_pairs": test_pairs,
        "test_ratings": test_ratings_arr
    }

    # 打印数据集划分信息
    print(f"\n===== 数据集划分 =====")
    print(f"训练集样本数：{len(train_ratings_arr)} ({len(train_ratings_arr) / len(ratings) * 100:.1f}%)")
    print(f"验证集样本数：{len(val_ratings_arr)} ({len(val_ratings_arr) / len(ratings) * 100:.1f}%)")
    print(f"测试集样本数：{len(test_ratings_arr)} ({len(test_ratings_arr) / len(ratings) * 100:.1f}%)")
    print(f"当前加载的模态：{modals}")

    return data_dict


# 测试代码（验证单/多模态加载是否正常）
if __name__ == "__main__":
    # 测试配置（可切换modals验证）
    test_config = {
        "data_path": "data/ml-1m",  # 替换为你的数据路径
        "feat_path": "data/features",  # 替换为你的特征路径
        "modals": ["text"],  # 可切换为["video"]/["text","video"]
        "modal_in_dims": {"text": 768, "video": 2048},
        "sample_frac": 0.2  # 测试20%采样
    }

    try:
        data_dict = load_graph_bundle_data(test_config)
        print(f"\n✅ 数据加载成功！")
        print(f"加载的模态：{data_dict['modals']}")
        print(f"特征字典keys：{list(data_dict['movie_feats'].keys())}")
        print(
            f"Text特征维度：{data_dict['movie_feats'].get('text', '无').shape if 'text' in data_dict['movie_feats'] else '无'}")
        print(
            f"Video特征维度：{data_dict['movie_feats'].get('video', '无').shape if 'video' in data_dict['movie_feats'] else '无'}")
    except Exception as e:
        print(f"❌ 数据加载失败：{e}")
        import traceback

        traceback.print_exc()
