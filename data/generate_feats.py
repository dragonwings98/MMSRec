import os
import torch
import pandas as pd
from PIL import Image
from tqdm import tqdm
from transformers import BertTokenizer, BertModel
import torchvision.models as models
import torchvision.transforms as transforms

# ===================== config =====================
DATA_DIR = "/home/li/datasets/multimodal_rec/ml-1m"  # The directory where movies.dat/ratings.dat is stored
POSTER_DIR = "/home/li/datasets/multimodal_rec/ml-1m/posters"  # Poster folder path
SAVE_DIR = "/home/li/datasets/multimodal_rec/ml-1m-multimodal/features"  # Feature storage directory
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# 创建保存目录
os.makedirs(SAVE_DIR, exist_ok=True)

# ===================== 1. Extracting text features (movie_text_emb.pt)=====================
print("===== Extracting text features =====")
# Read movies.dat (solving encoding issues)
movies = pd.read_csv(
    os.path.join(DATA_DIR, "movies.dat"),
    sep="::",
    engine="python",
    names=["movie_id", "title", "genres"],
    encoding="ISO-8859-1"
)
# Constructing text: Title + Type (Enhancing feature richness)
movies["text"] = movies["title"] + " | " + movies["genres"]
movie_ids = movies["movie_id"].tolist()
texts = movies["text"].tolist()

# Loading BERT model
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert_model = BertModel.from_pretrained("bert-base-uncased").to(DEVICE)
bert_model.eval()

# Batch extraction of text features
text_embs = []
batch_size = 32
with torch.no_grad():
    for i in tqdm(range(0, len(texts), batch_size), desc="Text feature extraction"):
        batch_texts = texts[i:i + batch_size]
        inputs = tokenizer(
            batch_texts,
            max_length=64,
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).to(DEVICE)
        outputs = bert_model(**inputs)
        # Extract the features (768 dimensions) of the [CLS] token.
        batch_emb = outputs.last_hidden_state[:, 0, :].cpu()
        text_embs.append(batch_emb)

# 合并并保存
text_emb = torch.cat(text_embs, dim=0)  # [3706, 768]
torch.save(text_emb, os.path.join(SAVE_DIR, "movie_text_emb.pt"))
print(f"文本特征保存完成：{text_emb.shape}")

# ===================== 2. Extracting poster (visual) features (movie_video_emb.pt)=====================
print("\n===== Extracting poster features =====")
# Load ResNet50 (pre-trained, remove classification header)
resnet = models.resnet50(pretrained=True).to(DEVICE)
resnet.eval()
feature_extractor = torch.nn.Sequential(*list(resnet.children())[:-1])

# Image preprocessing (adapted to ResNet)
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Match posters by movie_id
poster_embs = []
for movie_id in tqdm(movie_ids, desc="Poster Feature Extraction"):
    # Create the poster file name (matching your format: 1_Toy Story.jpg).
    # First find the corresponding movie title
    title = movies[movies["movie_id"] == movie_id]["title"].values[0]
    # Handle special characters in the title (to prevent filename matching failures).
    title_clean = title.replace(":", "").replace("/", "").replace("\\", "").replace("?", "")
    poster_name = f"{movie_id}_{title_clean}.jpg"
    poster_path = os.path.join(POSTER_DIR, poster_name)

    # Handling missing posters (filling with zero vectors)
    if not os.path.exists(poster_path):
        poster_embs.append(torch.zeros(2048))
        continue

    # Read and preprocess images
    img = Image.open(poster_path).convert("RGB")
    img_tensor = preprocess(img).unsqueeze(0).to(DEVICE)  # [1, 3, 224, 224]

    # Feature extraction
    with torch.no_grad():
        feat = feature_extractor(img_tensor).squeeze()  # [2048]
    poster_embs.append(feat.cpu())

# Merge and save (reuse fields from movie_video_emb.pt, adapt to MMSRec code).
poster_emb = torch.stack(poster_embs)  # [3706, 2048]
torch.save(poster_emb, os.path.join(SAVE_DIR, "movie_video_emb.pt"))
print(f"Poster features saved：{poster_emb.shape}")



# ===================== 验证所有特征 =====================
print("\n===== Feature generation complete =====")
print(f"Feature storage path：{SAVE_DIR}")
print(f"1. movie_text_emb.pt: {torch.load(os.path.join(SAVE_DIR, 'movie_text_emb.pt')).shape}")
print(f"2. movie_video_emb.pt: {torch.load(os.path.join(SAVE_DIR, 'movie_video_emb.pt')).shape}")
