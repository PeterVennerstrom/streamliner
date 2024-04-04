import open_clip
import torch
from PIL import Image

from streamliner.registry import register as REGISTER

torch.set_num_threads(1)


@REGISTER
class CLIPClassifier:
    def __init__(self, texts, arch, pretrain_source, device=0, **kwargs):
        self.device = device
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            arch, pretrain_source, device=f"cuda:{device}"
        )
        self.texts = texts
        self.text_classifiers = self._create_text_classifiers(texts, arch)

    def _create_text_classifiers(self, texts, arch):
        tokenizer = open_clip.get_tokenizer(arch)

        text_classifiers = []
        for text in texts:
            tokenized_text = tokenizer(text).cuda(self.device)
            text_features = self.model.encode_text(tokenized_text)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            text_classifiers.append(text_features)

        return text_classifiers

    def __call__(self, image):
        if isinstance(image, str):
            image = Image.open(image)
        else:
            image = Image.fromarray(image)

        image = self.preprocess(image).unsqueeze(0).cuda(self.device)

        with torch.no_grad(), torch.cuda.amp.autocast():
            image_features = self.model.encode_image(image)
            image_features /= image_features.norm(dim=-1, keepdim=True)

            results = []
            for text_features, text in zip(self.text_classifiers, self.texts):
                text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)
                text_probs = text_probs[0].cpu().numpy().tolist()
                result = dict(text=text, scores=text_probs)
                results.append(result)

            return results
