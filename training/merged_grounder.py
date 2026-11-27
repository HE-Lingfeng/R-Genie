"""Learn where an edit starts and ends from a frozen implicit-instruction readout."""
import torch
from torch import nn
from torch.nn import functional as F

from experiments.reasoning.grounding import Grounder, grounding_loss
from training.prepare_merged_regions import ACTIONS, GEOMETRIC


def warp_region(probability, transform):
    """Translate by fractions of the canvas and resize about the foreground centroid."""
    batch = len(probability)
    axis = (torch.arange(16, device=probability.device, dtype=probability.dtype) + 0.5) / 16
    y, x = torch.meshgrid(axis, axis, indexing='ij')
    coordinates = torch.stack((x, y), dim=-1)
    mass = probability.reshape(batch, 16, 16, 1)
    center = (mass * coordinates).sum((1, 2)) / mass.sum((1, 2)).clamp_min(1e-6)
    offset, scale = transform[:, :2], transform[:, 2:].clamp(-2, 2).exp()
    sampling = (coordinates - (center + offset)[:, None, None]) / scale[:, None, None]
    sampling = (sampling + center[:, None, None]) * 2 - 1
    return F.grid_sample(mass.permute(0, 3, 1, 2), sampling, align_corners=False).flatten(1)


class MergedGrounder(nn.Module):
    def __init__(self, width=64):
        super().__init__()
        self.source_head = Grounder(width=width)
        self.visual = nn.Sequential(nn.LayerNorm(2560), nn.Linear(2560, width))
        self.readout = nn.Sequential(nn.LayerNorm(2560), nn.Linear(2560, width))
        self.fuse = nn.Sequential(nn.Linear(3 * width, width), nn.LayerNorm(width), nn.GELU())
        self.action = nn.Linear(width, len(ACTIONS))
        self.transform = nn.Linear(width, 3)
        self.regions = nn.Sequential(
            nn.Conv2d(width + 4, width, 3, padding=1), nn.GroupNorm(8, width), nn.GELU(),
            nn.Conv2d(width, width, 3, padding=2, dilation=2), nn.GELU(), nn.Conv2d(width, 2, 1))
        nn.init.zeros_(self.transform.weight)
        nn.init.zeros_(self.transform.bias)
        nn.init.zeros_(self.regions[-1].weight)
        nn.init.zeros_(self.regions[-1].bias)
        axis = (torch.arange(16) + 0.5) / 16
        y, x = torch.meshgrid(axis, axis, indexing='ij')
        self.register_buffer('coordinates', torch.stack((x, y)).flatten(1))
        self.geometric_actions = tuple(i for i, name in enumerate(ACTIONS) if name in GEOMETRIC)

    def forward(self, visual, words, pad, prior):
        empty = torch.zeros_like(prior)
        source_logits = self.source_head(visual, words, pad, prior, empty, empty)
        source = source_logits.sigmoid()
        patches = self.visual(visual.float())
        attention = source_logits.softmax(1)
        pooled = (patches * attention.unsqueeze(-1)).sum(1)
        query = self.readout(words[:, -1].float())
        hidden = self.fuse(torch.cat((pooled, query, pooled * query), dim=1))
        action = self.action(hidden)
        transform = self.transform(hidden)
        destination = warp_region(source, transform)
        center = attention @ self.coordinates.T
        relative = self.coordinates.unsqueeze(0) - center.unsqueeze(-1)
        context = (patches * hidden[:, None]).transpose(1, 2)
        spatial = torch.cat((context, source[:, None], destination[:, None], relative), dim=1)
        anchor, correction = self.regions(spatial.reshape(len(visual), -1, 16, 16)).flatten(2).unbind(1)
        probabilities = action.softmax(1)
        p_geometry = probabilities[:, self.geometric_actions].sum(1, keepdim=True)
        p_anchor = probabilities[:, ACTIONS.index('anchored_add'), None]
        # An addition uses the source object as an anchor. Its changed pixels may lie outside it.
        edit = (1 - p_geometry - p_anchor) * source + p_geometry * torch.maximum(source, destination)
        edit = edit + p_anchor * (anchor - 3).sigmoid()
        return dict(source=source_logits, edit=torch.logit(edit.clamp(1e-5, 1-1e-5)) + correction,
                    destination=destination, action=action, transform=transform)


def objective(output, labels):
    instance = labels['has_instance']
    geometric = labels['has_geometry']
    loss = grounding_loss(output['edit'], labels['edit'])
    if instance.any():
        loss = loss + grounding_loss(output['source'][instance], labels['source'][instance])
        loss = loss + 0.5 * F.cross_entropy(output['action'][instance], labels['action'][instance])
    if geometric.any():
        probability = output['destination'][geometric]
        target = labels['destination'][geometric]
        # Keep the fractional occupancy at cell boundaries, including thin objects.
        dice = (2 * (probability * target).sum(1) + 1) / (probability.sum(1) + target.sum(1) + 1)
        alignment = F.binary_cross_entropy(probability.clamp(1e-5, 1-1e-5), target) + 1 - dice.mean()
        transform = F.smooth_l1_loss(output['transform'][geometric],
                                     labels['requested_dx_dy_log_scale'][geometric], beta=0.05)
        loss = loss + alignment + transform
    return loss
