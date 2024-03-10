import torch 
import torch.nn as nn 

class PatchEmbedding(nn.Module):
    def __init__(self, in_channel, patch_size, embed_dim, num_patches, dropout):
        super(PatchEmbedding, self).__init__()
        self.patcher = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=embed_dim, kernel_size=patch_size, stride=patch_size),
            # flatten 
            nn.Flatten(2)
        )

        # concate the cls_token
        self.cls_token = nn.Parameter(torch.randn(size=(1, 1, embed_dim)), 
                                        requires_grad=True)
        # adding the position embedding
        self.position_embedding = nn.Parameter(torch.randn(size=(1, num_patches+1, embed_dim)), requires_grad=True)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        # initiation cls token, and match the batch size of x in the 0th dimension
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        # after flatten we need to permute the dimension 
        x = self.patcher(x).permute(0, 2, 1)
        # concat cls token
        x = torch.cat([x, cls_token], dim=1)
        # add position embedding
        x = x + self.position_embedding
        x = self.dropout(x)
        return x 


class Vit(nn.Module):
    def __init__(self, in_channel, patch_size, embed_dim, num_patches, dropout, # patch parameters
                 num_heads, activation, num_encoders, num_classes) # transformers parameters
        super(Vit, self).__init__()
        self.patch_embedding = PatchEmbedding(in_channel, patch_size, embed_dim, num_patches, dropout)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dropout=dropout,
                                                   activation=activation, batch_first=True, norm_first=True)
        self.encoder_layers = nn.TransformerEncoder(encoder_layer, num_layers=num_encoders)
        self.MLP = nn.Sequential(
            nn.LayerNorm(normalized_shape=embed_dim),
            nn.Linear(in_features=embed_dim, out_features=num_classes)
        )

    def forward(self, x):
        x = self.patch_embedding(x)
        x = self.encoder_layers(x)
        x = self.MLP(x.squeeze(1))
        return x 