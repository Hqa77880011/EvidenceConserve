"""Trainable visual, transport, set, PPO and radiance-field baseline models."""

import torch
from torch import nn
from torch.nn import functional as F

from .third_party.set_transformer import ISAB, PMA
from .third_party.gennbv_encoder import HybridEncoder


class EvidenceEncoder(nn.Module):
    """Shared ResNet-18 image encoder with nonnegative Dirichlet evidence."""

    def __init__(self, classes, feature_dim=64, dropout=.2):
        super().__init__()
        from torchvision.models import resnet18
        self.backbone = resnet18(weights=None)
        self.backbone.fc = nn.Identity()
        self.embedding = nn.Sequential(nn.Linear(512, feature_dim), nn.GELU())
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(feature_dim, classes), nn.Softplus())

    def forward(self, image):
        feature = self.embedding(self.backbone(image))
        return self.head(feature), feature


def evidential_loss(evidence, label, annealing=.001):
    """Expected categorical NLL and wrong-class Dirichlet KL to the unit prior."""
    alpha = evidence + 1
    target = F.one_hot(label.long(), alpha.shape[-1]).to(alpha)
    nll = ((torch.digamma(alpha.sum(-1, keepdim=True)) - torch.digamma(alpha)) * target).sum(-1)
    adjusted = target + (1 - target) * alpha
    total = adjusted.sum(-1, keepdim=True)
    kl = (torch.lgamma(total).squeeze(-1) - torch.lgamma(adjusted).sum(-1)
          - torch.lgamma(alpha.new_tensor(float(alpha.shape[-1])))
          + ((adjusted - 1) * (torch.digamma(adjusted) - torch.digamma(total))).sum(-1))
    return (nll + annealing * kl).mean()


def relative_pose(target, source):
    """Translation and periodic Euler differences; identity pose maps to zeros."""
    delta = target - source
    return torch.cat([delta[..., :3], delta[..., 3:].sin(), delta[..., 3:].cos() - 1], -1)


class PoseTransport(nn.Module):
    """Nonnegative pose-conditioned KxK map, exactly identity at zero motion."""

    def __init__(self, classes, width=64):
        super().__init__()
        self.classes = classes
        self.network = nn.Sequential(nn.Linear(9, width), nn.GELU(), nn.Linear(width, classes * classes))

    def forward(self, target_pose, source_pose):
        feature = relative_pose(target_pose, source_pose)
        matrix = F.softplus(self.network(feature)).reshape(-1, self.classes, self.classes)
        strength = torch.linalg.vector_norm(feature, dim=-1).tanh()[:, None, None]
        identity = torch.eye(self.classes, device=matrix.device, dtype=matrix.dtype)[None]
        return (1 - strength) * identity + strength * matrix


class SetEvidence(nn.Module):
    """Official ISAB/ISAB/PMA Set Transformer, followed by an evidence head."""

    def __init__(self, feature_dim, classes, width=64, inducing=8, heads=4):
        super().__init__()
        self.encoder = nn.Sequential(ISAB(feature_dim, width, heads, inducing, ln=True),
                                     ISAB(width, width, heads, inducing, ln=True))
        self.pool = PMA(width, heads, 1, ln=True)
        self.head = nn.Sequential(nn.Linear(width, classes), nn.Softplus())

    def forward(self, features):
        return self.head(self.pool(self.encoder(features))).squeeze(1)


class ConcatEvidence(nn.Module):
    """Ordered, zero-padded feature concatenation ablation with an explicit count."""

    def __init__(self, feature_dim, classes, max_views, width=64):
        super().__init__()
        self.max_views = max_views
        self.network = nn.Sequential(nn.Linear(feature_dim * max_views + 1, width), nn.GELU(),
                                     nn.Linear(width, classes), nn.Softplus())

    def forward(self, features):
        count = features.shape[1]
        if not 1 <= count <= self.max_views:
            raise ValueError("Concatenation prefix exceeds the registered view budget")
        padded = F.pad(features, (0, 0, 0, self.max_views - count)).flatten(1)
        return self.network(torch.cat([padded, padded.new_full((len(padded), 1), count / self.max_views)], -1))


class CandidateDynamics(nn.Module):
    """Pre-capture pose/history model; dropout permits genuine MC predictive samples."""

    def __init__(self, classes, feature_dim, width=64, dropout=.2):
        super().__init__()
        self.classes = classes
        self.network = nn.Sequential(nn.Linear(6 + 9 + classes + feature_dim + 1, width), nn.GELU(),
                                     nn.Dropout(dropout), nn.Linear(width, width), nn.GELU(),
                                     nn.Dropout(dropout), nn.Linear(width, classes + feature_dim))

    def forward(self, values):
        values = self.network(values)
        return F.softplus(values[..., :self.classes]), values[..., self.classes:]


class GenNBVGraphPolicy(nn.Module):
    """Released GenNBV occupancy/pose encoder with a legal-graph PPO actor.

    This discrete reachable-action adaptation uses learned coverage rewards;
    it is independent of the upstream Isaac Gym simulator and drone controller.
    """

    def __init__(self, width=256):
        super().__init__()
        self.features = HybridEncoder(width=width)
        self.action = nn.Sequential(nn.Linear(6, width), nn.Tanh())
        self.actor = nn.Sequential(nn.Linear(2 * width, width), nn.Tanh(), nn.Linear(width, 1))
        self.critic = nn.Sequential(nn.Linear(width, width), nn.Tanh(), nn.Linear(width, 1))

    def forward(self, voxel, pose_history, candidate_poses):
        state = self.features(voxel, pose_history)
        action = self.action(candidate_poses)
        expanded = state[:, None].expand(-1, action.shape[1], -1)
        return self.actor(torch.cat([expanded, action], -1)).squeeze(-1), self.critic(state).squeeze(-1)


def ppo_loss(logits, values, actions, old_log_probability, advantages, returns, clip=.2):
    """Clipped PPO surrogate, value regression and categorical entropy."""
    policy = torch.distributions.Categorical(logits=logits)
    ratio = (policy.log_prob(actions) - old_log_probability).exp()
    surrogate = torch.minimum(ratio * advantages, ratio.clamp(1 - clip, 1 + clip) * advantages)
    return -surrogate.mean() + .5 * F.mse_loss(values, returns) - .01 * policy.entropy().mean()


class TinyRadianceField(nn.Module):
    """Small differentiable emission/absorption RF for the FisherRF criterion."""

    def __init__(self, width=24):
        super().__init__()
        self.geometry = nn.Sequential(nn.Linear(9, width), nn.Softplus(), nn.Linear(width, width), nn.Softplus())
        self.density = nn.Linear(width, 1)
        self.color = nn.Sequential(nn.Linear(width + 3, width), nn.Softplus(), nn.Linear(width, 3))

    def forward(self, points, directions):
        encoded = torch.cat([points, points.sin(), points.cos()], -1)
        feature = self.geometry(encoded)
        return F.softplus(self.density(feature)), self.color(torch.cat([feature, directions], -1)).sigmoid()


def render_rays(field, origins, directions, near=.1, far=4., samples=8):
    """Differentiable volume rendering; no candidate RGB target is needed."""
    distance = torch.linspace(near, far, samples + 1, device=origins.device, dtype=origins.dtype)
    midpoint = (distance[:-1] + distance[1:]) / 2
    points = origins[:, None] + directions[:, None] * midpoint[None, :, None]
    sigma, rgb = field(points, directions[:, None].expand_as(points))
    alpha = 1 - torch.exp(-sigma[..., 0] * (distance[1:] - distance[:-1]))
    transmittance = torch.cumprod(torch.cat([torch.ones_like(alpha[:, :1]), 1 - alpha + 1e-8], 1), 1)[:, :-1]
    weight = alpha * transmittance
    return (weight[..., None] * rgb).sum(1) + (1 - weight.sum(1, keepdim=True))


def fisher_diagonal(field, origins, directions, **render_options):
    """Gaussian-RGB Fisher diagonal from per-output radiance Jacobian squares."""
    parameters = tuple(field.parameters())
    prediction = render_rays(field, origins, directions, **render_options).flatten()
    diagonal = prediction.new_zeros(sum(parameter.numel() for parameter in parameters))
    for index in range(len(prediction)):
        gradients = torch.autograd.grad(prediction[index], parameters, retain_graph=index + 1 < len(prediction))
        diagonal += torch.cat([gradient.flatten() for gradient in gradients]).detach().square()
    return diagonal


def fisher_score(candidate_information, acquired_information, regularization=1e-3):
    """FisherRF diagonal trace H_candidate (H_acquired + lambda I)^-1."""
    return (candidate_information / (acquired_information + regularization)).sum()
