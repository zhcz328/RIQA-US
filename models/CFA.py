import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossAttentionBlock(nn.Module):
    def __init__(self, text_dim=768, img_dim=448, embed_dim=256):
        super(CrossAttentionBlock, self).__init__()
        self.query_proj = nn.Linear(text_dim, embed_dim) # Project text features to Query
        self.key_proj = nn.Conv2d(img_dim, embed_dim, kernel_size=1) # Project image features to Key
        self.value_proj = nn.Conv2d(img_dim, embed_dim, kernel_size=1) # Project image features to Value
        self.output_proj = nn.Conv2d(embed_dim, img_dim, kernel_size=1) # Restore the channel dimension of image features

    def forward(self, text_feature, image_feature):
        """
        Args:
        text_feature: [B, 512] - text encoding feature
        image_feature: [B, C, H, W] - image feature extracted by ResNet
        Returns:
        updated_feature: [B, C, H, W] - image feature output by Cross-Attention
        """
        B, C, H, W = image_feature.shape

        # 1. Project text features to Query (B, 512 -> B, embed_dim -> B, 1, 1, embed_dim)
        query = self.query_proj(text_feature).unsqueeze(1).unsqueeze(1) # [B, 1, 1, embed_dim]

        # 2. Project image features to Key and Value (B, C, H, W -> B, embed_dim, H, W)
        key = self.key_proj(image_feature) # [B, embed_dim, H, W]
        value = self.value_proj(image_feature) # [B, embed_dim, H, W]

        # 3. Adjust the shape of Key and Value (B, embed_dim, H*W)
        key = key.view(B, key.shape[1], -1) # [B, embed_dim, H*W]
        value = value.view(B, value.shape[1], -1) # [B, embed_dim, H*W]

        # 4. Calculate attention score (B, 1, 1, embed_dim) @ (B, embed_dim, H*W) -> [B, 1, H*W]
        attn = torch.matmul(query.flatten(2), key) # [B, 1, H*W]
        attn = F.softmax(attn / (key.shape[1] ** 0.5), dim=-1) # Softmax weights

        # 5. Weighted Value: (B, 1, H*W) @ (B, embed_dim, H*W)^T -> [B, 1, embed_dim]
        attn_output = torch.matmul(attn, value.permute(0, 2, 1)) # [B, 1, embed_dim]
        attn_output = attn_output.view(B, -1, 1, 1) # [B, embed_dim, 1, 1]

        # 6. Expand the attention output and combine it with the original image features
        attn_output = attn_output.expand(-1, -1, H, W) # [B, embed_dim, H, W]
        attn_output = self.output_proj(attn_output) # [B, C, H, W]

        updated_feature = image_feature + attn_output # residual connection
        return updated_feature

# # Assume text encoding and image features
# batch_size = 4
# text_feature = torch.randn(batch_size, 768) # Input text features (B, 768)
# image_feature = torch.randn(batch_size, 448, 56, 56) # ResNet extracted features (B, C=448, H=56, W=56)
#
# # Initialize CrossAttentionBlock
# cross_attention = CrossAttentionBlock(text_dim=768, img_dim=448, embed_dim=512)
#
# # Forward propagation
# output_feature = cross_attention(text_feature, image_feature)
#
# print("Output Feature Shape:", output_feature.shape)