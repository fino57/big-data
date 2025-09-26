
import torch
from PIL import Image
from conch.open_clip_custom.factory import create_model_from_pretrained

Image.MAX_IMAGE_PIXELS = None

class FeatureExtractor:
    def __init__(self, ckpt_path="ckpts/conch.pth", device=None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        model_cfg = 'conch_ViT-B-16'
        self.model, self.preprocess = create_model_from_pretrained(
            model_cfg, ckpt_path, device=self.device
        )
        self.model.eval()

    def extract_features(self, image_path):
        try:
            image = Image.open(image_path).convert("RGB")
            tensor = self.preprocess(image).unsqueeze(0).to(self.device)
            with torch.no_grad():
                feat = self.model.encode_image(tensor, return_tokens=True)
                
                return feat.squeeze(0).cpu().numpy().tolist()
        except Exception as e:
            print(f"⚠️ Error {image_path}: {str(e)}")
            return None
    
    # def extract_features(self, image_path):
    #     try:
    #         image = Image.open(image_path).convert("RGB")
    #         tensor = self.preprocess(image).unsqueeze(0).to(self.device)

    #     # Token-level embedding (N patches + cls token)
    #         with torch.no_grad():
    #            x = self.model.visual.conv1(tensor)                       # (B, C, H, W)
    #            x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)  # (B, N, C)
    #            cls_token = self.model.visual.cls_token.expand(x.shape[0], -1, -1)  # (B, 1, C)
    #            feats = torch.cat([cls_token, x], dim=1)                 # (B, N+1, C)
    #            feats = feats.squeeze(0).cpu().numpy()                   # (N+1, D)

    #         return feats

    #     except Exception as e:
    #         print(f"⚠️ Error {image_path}: {str(e)}")
    #         return None
