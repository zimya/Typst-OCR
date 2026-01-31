from torchvision import transforms
from PIL import Image, ImageOps
import torch
import numpy as np
from modelscope import AutoImageProcessor

# 添加允许的全局对象
torch.serialization.add_safe_globals([np.core.multiarray._reconstruct])

TARGET_HEIGHT = 416
TARGET_WIDTH = 512

class MixTexImageProcessor:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
        self.to_tensor = transforms.ToTensor()

    def __call__(self, image: Image.Image):
        # 1. 转换 RGB
        image = image.convert("RGB")

        # 2. 计算保持比例的缩放尺寸
        iw, ih = image.size  # Input Width, Input Height
        w, h = TARGET_WIDTH, TARGET_HEIGHT

        scale = min(w / iw, h / ih)
        nw = int(iw * scale)
        nh = int(ih * scale)

        # 3. 缩放图片
        image = image.resize((nw, nh), Image.BICUBIC)

        # 4. 创建背景并填充 (Padding)
        # 255 表示白色背景，对于 OCR 很重要
        new_image = Image.new('RGB', (w, h), (255, 255, 255))

        # 5. 粘贴图片 (居中粘贴)
        paste_x = (w - nw) // 2
        paste_y = (h - nh) // 2
        new_image.paste(image, (paste_x, paste_y))

        # 6. 转 Tensor 并标准化 (Swin 需要 ImageNet 的 mean/std)
        # ToTensor 会把 [0, 255] 转为 [0.0, 1.0]
        tensor = self.to_tensor(new_image)

        # 手动标准化
        tensor = transforms.functional.normalize(
            tensor, mean=self.mean, std=self.std
        )

        return tensor

# 获取 Swin 的默认均值和方差
processor_cfg = AutoImageProcessor.from_pretrained("microsoft/swin-tiny-patch4-window7-224")
custom_processor = MixTexImageProcessor(
    mean=processor_cfg.image_mean,
    std=processor_cfg.image_std
)

import torch
from modelscope import (
    SwinConfig, SwinModel,
    RobertaConfig, RobertaForCausalLM, # 改用 RoBERTa
    VisionEncoderDecoderModel,
    AutoTokenizer, AutoImageProcessor
)

TARGET_HEIGHT = 416
TARGET_WIDTH = 512

# ============================
# Encoder: Swin Transformer
# ============================
encoder_config = SwinConfig.from_pretrained("microsoft/swin-tiny-patch4-window7-224")
# 强制修改尺寸
encoder_config.image_size = (TARGET_HEIGHT, TARGET_WIDTH)

encoder = SwinModel.from_pretrained(
    "microsoft/swin-tiny-patch4-window7-224",
    config=encoder_config,
    ignore_mismatched_sizes=True # 允许尺寸变化导致的权重插值
)

# ============================
# Tokenizer & Decoder: RoBERTa
# ============================
tokenizer = AutoTokenizer.from_pretrained("MixTeX/MixTex-ZhEn-Latex-OCR", use_fast=True)

# 使用 RoBERTa 配置 (对应论文参数)
decoder_config = RobertaConfig(
    vocab_size=len(tokenizer),
    hidden_size=768,
    num_hidden_layers=4,
    num_attention_heads=12,
    intermediate_size=3072,
    # 关键配置
    is_decoder=True,
    add_cross_attention=True,
    bos_token_id=tokenizer.cls_token_id, # RoBERTa 的开始符通常是 <s> (id 0)
    eos_token_id=tokenizer.sep_token_id,
    pad_token_id=tokenizer.pad_token_id,
    max_position_embeddings=512,
    type_vocab_size=1, # RoBERTa 不需要 token_type_ids
)

# 使用 CausalLM 头部，这样才有 lm_head 输出 logits
decoder = RobertaForCausalLM(decoder_config)

# ============================
# 拼装 VisionEncoderDecoder
# ============================
model = VisionEncoderDecoderModel(encoder=encoder, decoder=decoder)

# 确保特殊 Token 对齐
model.config.decoder_start_token_id = tokenizer.cls_token_id
model.config.pad_token_id = tokenizer.pad_token_id
model.config.eos_token_id = tokenizer.sep_token_id
model.config.vocab_size = len(tokenizer)

# 生成配置
model.generation_config.decoder_start_token_id = tokenizer.cls_token_id
model.generation_config.pad_token_id = tokenizer.pad_token_id
model.generation_config.eos_token_id = tokenizer.sep_token_id
model.generation_config.max_length = 296

# ============================
# 再次检查数据类型 (常见坑)
# ============================
# 确保你的 mixtex dataset 返回的 pixel_values 是 Float32
# 有些 image processor 会返回 float64，导致训练极慢或出错
print("Model initialized correctly with pure Transformers lib.")


from datasets import load_dataset
from PIL import Image
import torch
from torch.utils.data import Dataset

# raw_dataset = load_dataset("MixTex/Pseudo-Latex-ZhEn-1")

raw_dataset = load_dataset(
    "parquet",
    data_files="pseudo_latex_train.parquet",
    # split="train",
)

class MixTexDataset(Dataset):
    def __init__(
        self,
        hf_dataset_split,
        tokenizer,
        image_processor, # 传入上面定义的 custom_processor
        max_length=296,
    ):
        self.data = hf_dataset_split
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]

        # 处理图片
        image = sample["image"]
        # 直接得到 tensor: [3, 400, 500]
        pixel_values = self.image_processor(image)

        # 处理文本
        text = sample["text"]
        tokens = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        input_ids = tokens.input_ids.squeeze(0)

        # 2. 【关键修正】如果第一位是 CLS/BOS，必须去掉！
        # 这样 labels 变成 [Token1, Token2, ..., EOS, PAD...]
        # 模型会自动在 decoder_input_ids 前面加上 decoder_start_token_id (CLS)
        # 从而形成：输入 [CLS] -> 预测 [Token1] 的正确逻辑
        if input_ids[0] == self.tokenizer.cls_token_id:
            input_ids = torch.cat([input_ids[1:], torch.tensor([self.tokenizer.pad_token_id])])

        labels = input_ids
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "pixel_values": pixel_values,
            "labels": labels,
        }

# 实例化 Dataset
split = raw_dataset["train"].train_test_split(test_size=0.01, seed=42)
train_dataset = MixTexDataset(split["train"], tokenizer, custom_processor)
val_dataset = MixTexDataset(split["test"], tokenizer, custom_processor)

import numpy as np
# 定义工具函数
def normalize(s: str) -> str:
    # 与论文一致：仅去除空白，不做结构等价
    return s.replace(" ", "").replace("\n", "")


def compute_metrics(eval_preds):
    pred_ids, label_ids = eval_preds

    # Token Accuracy
    correct = 0
    total = 0
    for p, l in zip(pred_ids, label_ids):
        for pi, li in zip(p, l):
            if li == -100:
                continue
            total += 1
            if pi == li:
                correct += 1
    token_acc = correct / max(total, 1)

    # Exact Match
    em_correct = 0
    for p, l in zip(pred_ids, label_ids):
        pred = tokenizer.decode(
            p,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )
        gt = tokenizer.decode(
            np.where(l != -100, l, tokenizer.pad_token_id),
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )
        if normalize(pred) == normalize(gt):
            em_correct += 1

    em = em_correct / len(pred_ids)

    return {
        "exact_match": em,
        "token_accuracy": token_acc,
    }

from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, default_data_collator
from transformers import AutoModelForSeq2SeqLM

# checkpoint_path = "./mixtex_train3/checkpoint-72850"
# print(f"Loading weights from {checkpoint_path}...")
# model = VisionEncoderDecoderModel.from_pretrained(checkpoint_path)

# model.config.decoder_start_token_id = tokenizer.cls_token_id
# model.config.pad_token_id = tokenizer.pad_token_id
# model.config.vocab_size = len(tokenizer)
# model.generation_config.max_length = 296

data_collator = default_data_collator

# 使用 AdamW，但给 Encoder 和 Decoder 不同的学习率（这也是论文常见的做法）
# Encoder 已经是预训练的，LR 小一点；Decoder 是随机的，LR 大一点。
optimizer = torch.optim.AdamW([
    {'params': model.encoder.parameters(), 'lr': 2e-5},  # Encoder 稍微解冻微调
    {'params': model.decoder.parameters(), 'lr': 1e-4}   # Decoder 随机初始化，需要大学习率
], weight_decay=0.01)

training_args = Seq2SeqTrainingArguments(
    output_dir="./mixtex_train4",
    per_device_train_batch_size=16,
    gradient_accumulation_steps=1,
    per_device_eval_batch_size=8,
    fp16=True,

    num_train_epochs=20,

    logging_steps=100,
    save_steps=1000,
    report_to=["tensorboard"],
    eval_steps=2000,
    eval_strategy="steps",
    save_total_limit=3,
    learning_rate=1e-4,
    warmup_steps=0, # no warmups

    predict_with_generate=True,
    generation_max_length=296,
)


# 必须解冻 Encoder
for p in model.encoder.parameters():
    p.requires_grad = True

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    optimizers=(optimizer, None) # 传入自定义 optimizer，scheduler 默认为空会让 Trainer 自动创建
)

with torch.serialization.safe_globals([np.core.multiarray._reconstruct]):
    trainer.train(resume_from_checkpoint="./mixtex_train3/checkpoint-72850")
