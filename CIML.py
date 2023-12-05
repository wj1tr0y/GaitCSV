import torch
import torch.nn.functional as F

from .base import BaseLoss, gather_and_scale_wrapper


class CIML(BaseLoss):
    def __init__(self, margin, view_num, loss_term_weight=1.0):
        super(CIML, self).__init__(loss_term_weight)
        self.margin = margin
        self.view_num = view_num

    @gather_and_scale_wrapper
    def forward(self, embeddings, labels, vies):
        # embeddings: [n, c, p], label: [n]
        embeddings = (
            embeddings.permute(2, 0, 1).contiguous().float()
        )  # [n, c, p] -> [p, n, c]

        ref_embed, ref_label = embeddings, labels
        dist = self.ComputeDistance(embeddings, ref_embed)  # [p, n1, n2]

        mean_dist = dist.mean((1, 2))  # [p]
        loss_so, loss_ss, loss_os, loss_oo, loss_cons = self.Convert2Triplets(
            labels, vies, dist
        )
        loss_so = F.relu(loss_so + self.margin)
        loss_ss = F.relu(loss_ss + self.margin)
        loss_os = F.relu(loss_os + self.margin)
        loss_oo = F.relu(loss_oo + self.margin)
        loss_cons = F.relu(loss_cons + self.margin)

        loss_avg_0, loss_num_0 = self.AvgNonZeroReducer(loss_so)
        loss_avg_1, loss_num_1 = self.AvgNonZeroReducer(loss_ss)
        loss_avg_2, loss_num_2 = self.AvgNonZeroReducer(loss_os)
        loss_avg_3, loss_num_3 = self.AvgNonZeroReducer(loss_oo)
        loss_avg_4, loss_num_4 = self.AvgNonZeroReducer(loss_cons)

        w0, w1, w2, w3 = loss_num_0/loss_so.size(-1), loss_num_1/loss_ss.size(-1), loss_num_2/loss_os.size(-1), loss_num_3/loss_oo.size(-1)
        
        self.info.update(
            {
                "loss_so": loss_avg_0.detach().clone(),
                "loss_ss": loss_avg_1.detach().clone(),
                "loss_os": loss_avg_2.detach().clone(),
                "loss_oo": loss_avg_3.detach().clone(),
                "loss_cons": loss_avg_4.detach().clone(),
                "loss_num_so": loss_num_0.detach().clone(),
                "loss_num_ss": loss_num_1.detach().clone(),
                "loss_num_os": loss_num_2.detach().clone(),
                "loss_num_oo": loss_num_3.detach().clone(),
                "loss_num_cons": loss_num_4.detach().clone(),
                "mean_dist": mean_dist.detach().clone(),
            }
        )

        return loss_avg_0*(w0+1) + loss_avg_1*(w1+1) + loss_avg_2*(w2+1) + loss_avg_3*(w3+1) + loss_avg_4, self.info

    def AvgNonZeroReducer(self, loss):
        eps = 1.0e-9
        loss_sum = loss.sum(-1)
        loss_num = (loss != 0).sum(-1).float()

        loss_avg = loss_sum / (loss_num + eps)
        loss_avg[loss_num == 0] = 0
        return loss_avg, loss_num

    def ComputeDistance(self, x, y):
        """
            x: [p, n_x, c]
            y: [p, n_y, c]
        """
        # dist = torch.sqrt(F.relu(2 - 2*x.matmul(y.transpose(1, 2))))
        x2 = torch.sum(x ** 2, -1).unsqueeze(2)  # [p, n_x, 1]
        y2 = torch.sum(y ** 2, -1).unsqueeze(1)  # [p, 1, n_y]
        inner = x.matmul(y.transpose(1, 2))  # [p, n_x, n_y]
        dist = x2 + y2 - 2 * inner
        dist = torch.sqrt(F.relu(dist))  # [p, n_x, n_y]
        return dist

    def get_mask(self, labels, n, p):

        matches = labels.unsqueeze(1) == labels.unsqueeze(0)  # [n_r, n_c]
        diffenc = torch.logical_not(matches)  # [n_r, n_c]
        matches = matches.view(1, n, n).repeat(p, 1, 1)
        diffenc = diffenc.view(1, n, n).repeat(p, 1, 1)
        return matches, diffenc

    def Convert2Triplets(self, labels, view_label, dist):
        """
                labels: tensor with size [n]
                view_label : tensor with size [n]
                dist: tensor with size [n,n]
                variables explanation:
                    an_diff: anchor,negative from different views
                    an_same: anchor,negative from same views
                    ap_diff: ...
                    ap_same: ...
                    an_diff_but_with_pos: anchor,negative from different views, but negative from (D_p)
                    an_diff_filter_out_pos: anchor,negative from different views, but negative from (D_n - D_p)
            """
        p, n, _ = dist.size()

        id_matches, id_diffenc = self.get_mask(labels, n, p)
        view_matches, view_diffenc = self.get_mask(view_label, n, p)

        an_diff_mask = (id_diffenc & view_diffenc).byte()
        an_same_mask = (id_diffenc & view_matches).byte()

        ap_diff_mask = (id_matches & view_diffenc).byte()
        ap_same_mask = (id_matches & view_matches).byte()

        v_one_hot = F.one_hot(view_label, num_classes=self.view_num).unsqueeze(1).repeat(1, n, 1) # n n v # 每个样本自己的view
        v_expand = view_label.unsqueeze(0).repeat(n, 1) # view label start from 0...V-1
        matches = id_matches[0, :, :].byte()
        v_expand = (v_expand + 1) * matches - 1 # make sure v==0 is correctly chosen
        v_expand[v_expand < 0] = self.view_num  # the negative sample is assigned to V
        v_scatter = torch.zeros(n, self.view_num + 1).cuda() # 0...V
        v_scatter = v_scatter.scatter(1, v_expand, 1).bool().unsqueeze(0).repeat(n, 1, 1) # [n, v, v+1] == 1 (D_p)
        v_scatter = v_scatter[:, :, :self.view_num] # filter out negative sample V (for each sample's one_hot D_p)

        an_diff_filter_out_pos = (v_one_hot & v_scatter).sum(-1) == 0
        
        an_diff_filter_out_pos = (an_diff_filter_out_pos.unsqueeze(0).repeat(p, 1, 1) & id_diffenc).byte()

        an_diff_but_with_pos = (v_one_hot & v_scatter).sum(-1) != 0
        an_diff_but_with_pos = (an_diff_but_with_pos.unsqueeze(0).repeat(p, 1, 1) & id_diffenc).byte()

        
        # hard_an_diff_filter_out_pos = (an_diff_mask ^ 1) * 0 + an_diff_mask
        # hard_an_diff_filter_out_pos = F.one_hot(torch.max(dist*hard_an_diff_filter_out_pos.float(), dim=-1)[1], num_classes=n)
        hard_an_diff_filter_out_pos = (an_diff_filter_out_pos ^ 1) * 0 + an_diff_filter_out_pos
        hard_an_diff_filter_out_pos = F.one_hot(torch.max(dist*hard_an_diff_filter_out_pos.float(), dim=-1)[1], num_classes=n)

        # hard_an_same = (an_same_mask ^ 1) * 1e12 + an_same_mask
        # hard_an_same = F.one_hot(torch.min(dist*hard_an_same.float(), dim=-1)[1], num_classes=n)

        full_dist = dist.unsqueeze(-1) - dist.unsqueeze(1)

        loss_1_mask = ap_same_mask[0,:,:].unsqueeze(-1) * an_diff_but_with_pos[0,:,:].unsqueeze(-2)
        a_idx, p_idx, n_idx = torch.where(loss_1_mask)
        loss_so = full_dist[:, a_idx, p_idx, n_idx]

        loss_2_mask = ap_same_mask[0,:,:].unsqueeze(-1) * an_same_mask[0,:,:].unsqueeze(-2)
        a_idx, p_idx, n_idx = torch.where(loss_2_mask)
        loss_ss = full_dist[:, a_idx, p_idx, n_idx]

        loss_3_mask = ap_diff_mask[0,:,:].unsqueeze(-1) * an_same_mask[0,:,:].unsqueeze(-2)
        a_idx, p_idx, n_idx = torch.where(loss_3_mask)
        loss_os = full_dist[:, a_idx, p_idx, n_idx]

        loss_4_mask = ap_diff_mask[0,:,:].unsqueeze(-1) * an_diff_but_with_pos[0,:,:].unsqueeze(-2)
        a_idx, p_idx, n_idx = torch.where(loss_4_mask)
        loss_oo = full_dist[:, a_idx, p_idx, n_idx]

        loss_5_mask = hard_an_diff_filter_out_pos[0, :, :].unsqueeze(-1) * an_same_mask[0, :, :].unsqueeze(-2)
        a_idx, p_idx, n_idx = torch.where(loss_5_mask)
        loss_cons = full_dist[:, a_idx, p_idx, n_idx]

        return loss_so, loss_ss, loss_os, loss_oo, loss_cons

