import os
import torch
import pandas as pd
from PIL import Image
from tqdm import tqdm
from transformers import BertTokenizer, BertModel
import torchvision.models as models
import torchvision.transforms as transforms

# ===================== 配置（修改为你的实际路径）=====================
DATA_DIR = "/home/li/datasets/multimodal_rec/ml-1m"  # 存放movies.dat/ratings.dat的目录
POSTER_DIR = "/home/li/datasets/multimodal_rec/ml-1m/posters"  # 海报文件夹路径
SAVE_DIR = "/home/li/datasets/multimodal_rec/ml-1m-multimodal/features"  # 特征保存目录
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# 创建保存目录
os.makedirs(SAVE_DIR, exist_ok=True)

# ===================== 1. 提取文本特征（movie_text_emb.pt）=====================
print("===== 提取文本特征 =====")
# 读取movies.dat（解决编码问题）
movies = pd.read_csv(
    os.path.join(DATA_DIR, "movies.dat"),
    sep="::",
    engine="python",
    names=["movie_id", "title", "genres"],
    encoding="ISO-8859-1"
)
# 构建文本：标题 + 类型（提升特征丰富度）
movies["text"] = movies["title"] + " | " + movies["genres"]
movie_ids = movies["movie_id"].tolist()
texts = movies["text"].tolist()

# 加载BERT模型
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert_model = BertModel.from_pretrained("bert-base-uncased").to(DEVICE)
bert_model.eval()

# 批量提取文本特征
text_embs = []
batch_size = 32
with torch.no_grad():
    for i in tqdm(range(0, len(texts), batch_size), desc="文本特征提取"):
        batch_texts = texts[i:i + batch_size]
        inputs = tokenizer(
            batch_texts,
            max_length=64,
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).to(DEVICE)
        outputs = bert_model(**inputs)
        # 取[CLS] token的特征（768维）
        batch_emb = outputs.last_hidden_state[:, 0, :].cpu()
        text_embs.append(batch_emb)

# 合并并保存
text_emb = torch.cat(text_embs, dim=0)  # [3706, 768]
torch.save(text_emb, os.path.join(SAVE_DIR, "movie_text_emb.pt"))
print(f"文本特征保存完成：{text_emb.shape}")

# ===================== 2. 提取海报（视觉）特征（movie_video_emb.pt）=====================
print("\n===== 提取海报特征 =====")
# 加载ResNet50（预训练，移除分类头）
resnet = models.resnet50(pretrained=True).to(DEVICE)
resnet.eval()
feature_extractor = torch.nn.Sequential(*list(resnet.children())[:-1])

# 图像预处理（适配ResNet）
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 按movie_id匹配海报
poster_embs = []
for movie_id in tqdm(movie_ids, desc="海报特征提取"):
    # 构建海报文件名（匹配你的格式：1_Toy Story.jpg）
    # 先找到对应电影标题
    title = movies[movies["movie_id"] == movie_id]["title"].values[0]
    # 处理标题特殊字符（避免文件名匹配失败）
    title_clean = title.replace(":", "").replace("/", "").replace("\\", "").replace("?", "")
    poster_name = f"{movie_id}_{title_clean}.jpg"
    poster_path = os.path.join(POSTER_DIR, poster_name)

    # 处理海报缺失的情况（用零向量填充）
    if not os.path.exists(poster_path):
        poster_embs.append(torch.zeros(2048))
        continue

    # 读取并预处理图片
    img = Image.open(poster_path).convert("RGB")
    img_tensor = preprocess(img).unsqueeze(0).to(DEVICE)  # [1, 3, 224, 224]

    # 提取特征
    with torch.no_grad():
        feat = feature_extractor(img_tensor).squeeze()  # [2048]
    poster_embs.append(feat.cpu())

# 合并并保存（复用movie_video_emb.pt字段，适配MMSRec代码）
poster_emb = torch.stack(poster_embs)  # [3706, 2048]
torch.save(poster_emb, os.path.join(SAVE_DIR, "movie_video_emb.pt"))
print(f"海报特征保存完成：{poster_emb.shape}")



# ===================== 验证所有特征 =====================
print("\n===== 特征生成完成 =====")
print(f"特征保存路径：{SAVE_DIR}")
print(f"1. movie_text_emb.pt: {torch.load(os.path.join(SAVE_DIR, 'movie_text_emb.pt')).shape}")
print(f"2. movie_video_emb.pt: {torch.load(os.path.join(SAVE_DIR, 'movie_video_emb.pt')).shape}")
