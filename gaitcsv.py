import torch
import torch.nn.functional as F
from ..base_model import BaseModel
from ..modules import (
    SetBlockWrapper,
    HorizontalPoolingPyramid,
    PackSequenceWrapper,
    SeparateFCs,
    SeparateBNNecks,
)

class GaitCSV(BaseModel):
    def build_network(self, model_cfg):
        self.Backbone = self.get_backbone(model_cfg["backbone_cfg"])
        self.Backbone = SetBlockWrapper(self.Backbone)
        self.FCs = SeparateFCs(**model_cfg["SeparateFCs"])
        self.TP = PackSequenceWrapper(torch.max)
        self.HPP = HorizontalPoolingPyramid(bin_num=model_cfg["bin_num"])
        self.BNNeck = SeparateBNNecks(**model_cfg['SeparateBNNecks'])

    def forward(self, inputs):  #  NM: 68.35% # 65.25 baseline fc

        ipts, labs, typs, vies, seqL = inputs
        sils = ipts[0]
        if len(sils.size()) == 4:
            sils = sils.unsqueeze(1)
        else:
            sils = sils.squeeze(2).unsqueeze(1)
        del ipts
        outs = self.Backbone(sils)  # [n, c, s, h, w]
        outs = self.TP(outs, seqL, options={"dim": 2})[0]  # [n, c, h, w]
        feat = self.HPP(outs)  # [n, c, p]
        embed = self.FCs(feat)  # [n, c, p]
        embed = F.normalize(embed, dim=1)  # norm+tri -> bn+ce
        embed2, logits = self.BNNeck(embed)

        n, _, s, h, w = sils.size()
        retval = {
            "training_feat": {
                "CIML": {"embeddings": embed, "labels": labs, "vies": vies},
                "softmax": {"logits": logits, "labels": labs},
            },
            "visual_summary": {"image/sils": sils.view(n * s, 1, h, w)},
            "inference_feat": {"embeddings": embed},
        }
        return retval