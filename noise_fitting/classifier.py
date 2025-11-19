import torch
import torch.nn as nn
import torch.nn.functional as F

class SupConLoss(nn.Module):
    """
    Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the self-supervised SimCLR case (if labels are None).
    """
    def __init__(self, temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature

    def forward(self, features, labels=None):
        """
        Args:
            features: hidden vector of shape [batch_size, n_views, ...].
            labels: ground truth of shape [batch_size].
        Returns:
            A loss scalar.
        """
        device = features.device

        # 1. Setup Dimensions
        # We assume input shape: [batch_size, n_views, feature_dim]
        # Usually n_views is 2 (original image + augmented version)
        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [batch_size, n_views, ...]')
        
        batch_size = features.shape[0]
        n_views = features.shape[1]

        # Flatten the views to combine them into one large batch
        # New shape: [batch_size * n_views, feature_dim]
        features = torch.cat(torch.unbind(features, dim=1), dim=0)

        # 2. Normalize features (Crucial for Contrastive Learning)
        features = F.normalize(features, dim=1)

        # 3. Compute Similarity Matrix
        # similarity[i, j] = cosine similarity between feat_i and feat_j
        similarity_matrix = torch.matmul(features, features.T)

        # 4. Create Labels Mask
        if labels is not None:
            # Supervised Contrastive:
            # Create a matrix where mask[i, j] is 1 if label[i] == label[j]
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            
            # Extend labels to match the flattened features view
            mask = torch.eq(labels, labels.T).float().to(device)
            
            # Since we have n_views per image, we need to repeat the mask
            mask = mask.repeat(n_views, n_views)
        else:
            # Self-Supervised (SimCLR):
            # Mask is identity (only the image itself is its own positive pair initially)
            mask = torch.eye(batch_size * n_views, dtype=torch.float32).to(device)

        # 5. Remove Self-Contrast
        # We don't want the model to learn that an image is similar to itself (trivial)
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * n_views).view(-1, 1).to(device),
            0
        )
        
        # Mask out the diagonal in the label mask as well
        mask = mask * logits_mask

        # 6. Compute Logits
        # Scale by temperature
        logits = similarity_matrix / self.temperature
        
        # Numerical stability: subtract max from logits
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - logits_max.detach()

        # 7. Compute Loss
        # Denominator: sum of exp(logits) for all negatives AND positives (excluding self)
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # Mean log-likelihood for positive pairs
        # Compute mean of log-likelihood over positive pairs
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # Loss is negative log likelihood
        loss = - mean_log_prob_pos
        
        # Take mean over the batch
        loss = loss.view(n_views, batch_size).mean()

        return loss

# --- Simple Model Definition ---

class SimpleEncoder(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=64, output_dim=32):
        super(SimpleEncoder, self).__init__()
        
        # The Encoder (Backbone)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # The Projection Head
        # (Used only during training, discarded for inference)
        self.projector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        feat = self.encoder(x)
        proj = self.projector(feat)
        # Return projection for loss calculation
        return F.normalize(proj, dim=1)