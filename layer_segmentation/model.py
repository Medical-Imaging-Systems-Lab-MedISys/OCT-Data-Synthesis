import torch
import torch.nn as nn
import timm

class RETFoundSegmenter(nn.Module):
    def __init__(self, num_classes=9, img_size=224, pretrained_path=None):
        """
        num_classes: number of output classes
        img_size: input image size (e.g. 224 or 256)
        pretrained_path: Path to the downloaded RETFound weights (.pth)
        """
        super().__init__()
        self.img_size = img_size
        
        # 1. Initialize ViT-Large (RETFound backbone)
        self.encoder = timm.create_model(
            'vit_large_patch16_224', 
            img_size=img_size,
            pretrained=False, 
            num_classes=0
        )
        
        # Load RETFound pre-trained weights if provided
        if pretrained_path:
            checkpoint = torch.load(pretrained_path, map_location='cpu', weights_only=False)
            state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
            # Remove head weights as we don't need them for segmentation
            state_dict = {k: v for k, v in state_dict.items() if 'head' not in k}
            
            # Interpolate position embeddings if there is a size mismatch
            if 'pos_embed' in state_dict:
                pos_embed_checkpoint = state_dict['pos_embed']
                pos_embed_model = self.encoder.pos_embed
                if pos_embed_checkpoint.shape != pos_embed_model.shape:
                    print(f"Interpolating pos_embed from {pos_embed_checkpoint.shape} to {pos_embed_model.shape}...")
                    # ViT has 1 class token
                    num_extra_tokens = 1
                    embedding_dim = pos_embed_checkpoint.shape[-1]
                    orig_size = int((pos_embed_checkpoint.shape[1] - num_extra_tokens) ** 0.5)
                    new_size = int((pos_embed_model.shape[1] - num_extra_tokens) ** 0.5)
                    
                    extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
                    pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
                    pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_dim).permute(0, 3, 1, 2)
                    pos_tokens = torch.nn.functional.interpolate(
                        pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False
                    )
                    pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
                    new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
                    state_dict['pos_embed'] = new_pos_embed
            
            self.encoder.load_state_dict(state_dict, strict=False)
            print(f"Loaded RETFound weights from {pretrained_path}")

        # 2. Convolutional Decoder Head
        self.decoder = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            
            nn.Conv2d(64, num_classes, kernel_size=1)
        )

    def forward(self, x):
        B = x.shape[0]
        
        # Forward pass through RETFound encoder
        features = self.encoder.forward_features(x) # (B, num_patches + 1, 1024)
        
        # Discard the CLS token (index 0)
        features = features[:, 1:, :] # (B, num_patches, 1024)
        
        # Reshape into 2D spatial grid
        grid_size = int(features.shape[1] ** 0.5)
        features = features.permute(0, 2, 1).reshape(B, 1024, grid_size, grid_size)
        
        # Forward pass through Decoder
        logits = self.decoder(features) # (B, num_classes, img_size, img_size)
        return logits

if __name__ == "__main__":
    model = RETFoundSegmenter(num_classes=9, img_size=256)
    dummy_input = torch.randn(2, 3, 256, 256)
    output = model(dummy_input)
    print("Output shape:", output.shape) # Expected: [2, 9, 256, 256]