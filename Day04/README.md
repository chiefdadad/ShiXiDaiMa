# Transformer学习笔记

1. ### Transformer核心特点：

   - **并行计算：**提升训练效率；

   - **长距离依赖：**解决序列建模问题；

   - **模块化设计：**编码器-解码器结构；

   - **纯注意力机制：**完全取代RNN/CNN。

     

2. ### 核心组件：

   - 自注意力机制：

     ```python
     scores = Q @ K.T / sqrt(d_k)
     attn = softmax(scores)
     output = attn @ V
     ```

     

   - 多头注意力：

     ```python
     class MultiHeadAttention(nn.Module):
         def __init__(self, d_model, num_heads):
             # 线性投影层初始化
         def forward(self, Q, K, V):
             # 分割多头→计算注意力→拼接结果
     ```

     

   - 前馈网络：

     ```python
     class PositionwiseFeedForward(nn.Module):
         def __init__(self, d_model, d_ff):
             self.linear1 = Linear(d_model, d_ff)
             self.linear2 = Linear(d_ff, d_model)
     ```

     

   - 位置编码：

     ```python
     PE(pos,2i) = sin(pos/10000^(2i/d_model))
     PE(pos,2i+1) = cos(pos/10000^(2i/d_model))
     ```

     

3. ### 完整结构（使用PyTorch + einops来实现）：

   ```python
   class ViT(nn.Module):
       def __init__(self, *, image_size=224, patch_size=16, num_classes=100, dim=1024,
                    depth=6, heads=8, mlp_dim=2048, pool='cls', channels=3,
                    dim_head=64, dropout=0., emb_dropout=0.):
           super().__init__()
           image_height, image_width = pair(image_size)
           patch_height, patch_width = pair(patch_size)
   
           assert image_height % patch_height == 0 and image_width % patch_width == 0, \
               'Image dimensions must be divisible by the patch size.'
   
           num_patches = (image_height // patch_height) * (image_width // patch_width)
           patch_dim = channels * patch_height * patch_width
           assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
   
           self.to_patch_embedding = nn.Sequential(
               Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width),
               nn.LayerNorm(patch_dim),
               nn.Linear(patch_dim, dim),
               nn.LayerNorm(dim),
           )
   
           self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
           self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
           self.dropout = nn.Dropout(emb_dropout)
   
           self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
   
           self.pool = pool
           self.to_latent = nn.Identity()
           self.mlp_head = nn.Linear(dim, num_classes)
   
       def forward(self, img):
           x = self.to_patch_embedding(img)
           b, n, _ = x.shape
   
           cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
           x = torch.cat((cls_tokens, x), dim=1)
           x += self.pos_embedding[:, :(n + 1)]
           x = self.dropout(x)
   
           x = self.transformer(x)
           x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]
           x = self.to_latent(x)
           return self.mlp_head(x)
   ```

   

3. ### ViT和ViT_1D的区别：

   - ViT用于处理2D图像输入（形状：[批次, 通道, 高度, 宽度]），而ViT_1D用于处理1D序列 / 时序数据（形状：[批次, 通道, 序列长度]）；

   - 在分块处理上，ViT将2D图像切割为2D块，使用Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)')，而ViT_1D将1D序列分割为1D块，使用Rearrange('b c (n p) -> b n (p c)')；

   - 二者共享的核心组件：分块嵌入，位置编码，多头注意力机制，前馈网络。

     

   **总结：**ViT是用于图像分类的传统视觉变换器，而ViT_1D则在前者基础上进行改进，用于处理时间序列、音频等1D序列数据。



# Day04作业

- 完成ViT.py文件的运行（运行结果为 **Model output shape: torch.Size([1, 100])** ）；
- 学习网上Yolov5的相关知识。