import torch
import torch.nn.functional as F


class SupConLoss(torch.nn.Module):

    def __init__(self, temperature=0.1):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.base_temperature = 0.07

    def forward(self, features, labels):
        # Normalize the feature vectors
        device = features.device
        features = F.normalize(features, dim=1)
        
        # Create masks for positive and negative samples
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float()
        logits_mask = torch.scatter(
            torch.ones_like(mask), 1,
            torch.arange(labels.shape[0]).view(-1, 1).to(device), 0)
        mask = mask * logits_mask

        # Compute the similarity matrix
        similarity_matrix = torch.div(
            torch.matmul(features, features.T) ,  self.temperature)
        
        # For numerical stability
        logits_max, _ = torch.max(similarity_matrix * logits_mask, dim=1, keepdim=True)
        logits = similarity_matrix - logits_max.detach()

        # Logits for positive samples
        exp_logits = torch.exp(logits) * logits_mask

        # Logits for all samples
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mask_pos_pairs = mask.sum(1)
        mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, 1, mask_pos_pairs)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs

        # loss
        loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.mean()

        return loss
