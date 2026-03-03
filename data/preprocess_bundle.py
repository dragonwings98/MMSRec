import os
import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def load_graph_bundle_data(config):
    """
    Data loading adapted to graph bundle theory + training/validation/test set partitioning (8:1:1)
    1. Filter users with only one rating, resolving stratified sampling errors.
    2. Support for small data volume sampling (optimized for speed).
    3. Load only modal features specified in the config (compatible with single/multimodal features).
    """
    # 1. Path resolution
    data_path = config["data_path"]
    feat_path = config["feat_path"]
    modals = config["modals"]  # The current configuration specifies the modality：["text"]/["video"]/["text","video"]
    modal_in_dims = config["modal_in_dims"]  # Modal input dimension：{"text":768, "video":2048}

    # 2. Loading basic data
    # Rating data：user_id, movie_id, rating, timestamp
    ratings = pd.read_csv(
        os.path.join(data_path, "ratings.dat"),
        sep="::", names=["user_id", "movie_id", "rating", "timestamp"],
        engine="python", encoding="latin-1"
    )
    # Movie Data：movie_id, title, genres
    movies = pd.read_csv(
        os.path.join(data_path, "movies.dat"),
        sep="::", names=["movie_id", "title", "genres"],
        engine="python", encoding="latin-1"
    )

    # ========== 1：Small data volume sampling (optimization and speed improvement) ==========
    # Supports sampling ratio configuration (default is full, use 20% for optimization).
    sample_frac = config.get("sample_frac", 1.0)  # Read the sampling ratio from the configuration.
    if sample_frac < 1.0:
        print(f"sample {sample_frac * 100}%")
        # Sampling by user (ensuring at least one record for each user)
        user_ids = ratings["user_id"].unique()
        sampled_users = np.random.choice(user_ids, size=int(len(user_ids) * sample_frac), replace=False)
        ratings = ratings[ratings["user_id"].isin(sampled_users)]

    # ========== 2：Filter users with only one rating ==========
    # Count the number of ratings for each user
    user_rating_count = ratings["user_id"].value_counts()
    # Users with at least two ratings are retained (meeting stratify tiering requirements).
    valid_users = user_rating_count[user_rating_count >= 2].index
    ratings = ratings[ratings["user_id"].isin(valid_users)]
    print(f"Number of users retained after filtering:{len(valid_users)}（Original user count：{len(user_rating_count)}）")
    print(f"Number of filtered rating samples：{len(ratings)}")

    # 3. Construct an ID mapping (converting raw IDs into consecutive indices)
    user_ids = ratings["user_id"].unique()
    movie_ids = ratings["movie_id"].unique()
    user2idx = {uid: idx for idx, uid in enumerate(user_ids)}
    movie2idx = {mid: idx for idx, mid in enumerate(movie_ids)}
    num_users = len(user2idx)
    num_movies = len(movie2idx)

    # 4. Convert to continuous IDs (for easier processing in the model embedding layer).
    ratings["user_idx"] = ratings["user_id"].map(user2idx)
    ratings["movie_idx"] = ratings["movie_id"].map(movie2idx)

    # 5. Stratified dataset partitioning (core: stratify by user to ensure the rating distribution for each user)
    # Step 1: Divide the dataset into a training set (80%) and a temporary set (20%).
    train_ratings, temp_ratings = train_test_split(
        ratings, test_size=0.2, random_state=42, stratify=ratings["user_idx"]
    )
    # Step 2: Divide the dataset into a validation set (10%) and a test set (10%).
    val_ratings, test_ratings = train_test_split(
        temp_ratings, test_size=0.5, random_state=42, stratify=temp_ratings["user_idx"]
    )

    # 6. Construct training/validation/test samples (user-movie pairs + real ratings)
    def build_samples(ratings_df):
        user_movie_pairs = np.array(list(zip(ratings_df["user_idx"], ratings_df["movie_idx"])))
        true_ratings = np.array(ratings_df["rating"], dtype=np.float32)
        return torch.tensor(user_movie_pairs, dtype=torch.long), torch.tensor(true_ratings, dtype=torch.float32)

    train_pairs, train_ratings_arr = build_samples(train_ratings)
    val_pairs, val_ratings_arr = build_samples(val_ratings)
    test_pairs, test_ratings_arr = build_samples(test_ratings)

    # 7. Load specified modal features
    movie_feats = {}  # The final returned feature dictionary contains only the modalities needed at the moment.
    raw_movie_ids = movies["movie_id"].tolist()
    raw_movie2feat_idx = {mid: idx for idx, mid in enumerate(raw_movie_ids)}  # Mapping from original movie ID to feature index

    for modal in modals:  # Only traverse the modal specified in the current configuration.
        # The path to the spliced feature file is: feat_path/movie_{modal}_emb.pt
        feat_file = os.path.join(feat_path, f"movie_{modal}_emb.pt")

        # Check if the feature file exists (to avoid errors due to missing files).
        if not os.path.exists(feat_file):
            raise FileNotFoundError(f"The bundle model requires a {modal} feature file, but it was not found: {feat_file}")

        # Loading modal features
        modal_feat = torch.load(feat_file, map_location="cpu")
        print(f"Load {modal} features (bundle adaptation): Original dimensions {modal_feat.shape}")

        # Feature alignment: Aligns the original features to the movie IDs in the current dataset (handles missing movies).
        aligned_feat = []
        for mid in movie_ids:
            if mid in raw_movie2feat_idx:
                aligned_feat.append(modal_feat[raw_movie2feat_idx[mid]])
            else:
                aligned_feat.append(torch.zeros(modal_in_dims[modal]))

        # Convert to tensor and save to dictionary
        movie_feats[modal] = torch.stack(aligned_feat)
        print(f"Alignment {modal} features：final dim: {movie_feats[modal].shape}（movies：{num_movies}，dimension：{modal_in_dims[modal]}）")

    # 8. Encapsulate the returned data
    data_dict = {
        # basic
        "num_users": num_users,
        "num_movies": num_movies,
        "user2idx": user2idx,
        "movie2idx": movie2idx,
        # Modal correlation
        "movie_feats": movie_feats,
        "modals": modals,
        "modal_in_dims": modal_in_dims,
        # training set
        "train_pairs": train_pairs,
        "train_ratings": train_ratings_arr,
        # Validation set
        "val_pairs": val_pairs,
        "val_ratings": val_ratings_arr,
        # test set
        "test_pairs": test_pairs,
        "test_ratings": test_ratings_arr
    }

    # Print dataset partitioning information
    print(f"\n===== Dataset partitioning =====")
    print(f"Number of training set samples：{len(train_ratings_arr)} ({len(train_ratings_arr) / len(ratings) * 100:.1f}%)")
    print(f"Number of validation set samples：{len(val_ratings_arr)} ({len(val_ratings_arr) / len(ratings) * 100:.1f}%)")
    print(f"Number of test set samples：{len(test_ratings_arr)} ({len(test_ratings_arr) / len(ratings) * 100:.1f}%)")
    print(f"Currently loaded modality：{modals}")

    return data_dict


# Test code (to verify whether single/multimodal loading is working correctly)
if __name__ == "__main__":
    # Test configuration (modals verification can be switched)
    test_config = {
        "data_path": "data/ml-1m",
        "feat_path": "data/features",
        "modals": ["text"],  # You can switch to ["video"]/["text","video"]
        "modal_in_dims": {"text": 768, "video": 2048},
        "sample_frac": 0.2  # Test 20% sampling
    }

    try:
        data_dict = load_graph_bundle_data(test_config)
        print(f"\n✅ Data loading successful！")
        print(f"Loading modality：{data_dict['modals']}")
        print(f"Feature dictionary keys：{list(data_dict['movie_feats'].keys())}")
        print(
            f"Text Feature Dimensions：{data_dict['movie_feats'].get('text', 'none').shape if 'text' in data_dict['movie_feats'] else 'none'}")
        print(
            f"Video feature dimensions：{data_dict['movie_feats'].get('video', 'none').shape if 'video' in data_dict['movie_feats'] else 'none'}")
    except Exception as e:
        print(f"❌ Data loading failed：{e}")
        import traceback

        traceback.print_exc()
