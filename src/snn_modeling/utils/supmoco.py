import copy
import torch
import torch.nn as nn
import torch.nn.functional as F


@torch.no_grad()
def build_momentum_encoder(model):
    momentum_model = copy.deepcopy(model)
    momentum_model.eval()
    for param in momentum_model.parameters():
        param.requires_grad = False
    return momentum_model


@torch.no_grad()
def momentum_update(online_model, momentum_model, momentum):
    online_dict = online_model.state_dict()
    momentum_dict = momentum_model.state_dict()
    for key in momentum_dict.keys():
        if torch.is_floating_point(momentum_dict[key]):
            momentum_dict[key].data.mul_(momentum).add_(online_dict[key].data, alpha=1.0 - momentum)
        else:
            # For integer buffers (e.g. num_batches_tracked), just copy them
            momentum_dict[key].data.copy_(online_dict[key].data)

class SupMoCoState(nn.Module):
    def __init__(self, queue_size=4096, feature_dim=128):
        super().__init__()
        self.queue_size = int(queue_size)
        self.feature_dim = int(feature_dim)

        self.register_buffer("queue", torch.zeros(self.queue_size, self.feature_dim))
        self.register_buffer("queue_labels", torch.full((self.queue_size,), -1, dtype=torch.long))
        self.register_buffer("queue_subject_labels", torch.full((self.queue_size,), -1, dtype=torch.long))
        self.register_buffer("queue_video_ids", torch.full((self.queue_size,), -1, dtype=torch.long))
        self.register_buffer("queue_timestamps", torch.full((self.queue_size,), -1, dtype=torch.long))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        self.register_buffer("queue_filled", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def enqueue(self, features, labels, subject_labels=None, video_ids=None, timestamps=None):
        if features.numel() == 0:
            return

        features = F.normalize(features.detach(), dim=1, eps=1e-6)
        labels = labels.detach().long()
        if subject_labels is None:
            subject_labels = torch.full_like(labels, -1)
        else:
            subject_labels = subject_labels.detach().long()
        if video_ids is None:
            video_ids = torch.full_like(labels, -1)
        else:
            video_ids = video_ids.detach().long()
        if timestamps is None:
            timestamps = torch.full_like(labels, -1)
        else:
            timestamps = timestamps.detach().long()

        n = features.shape[0]
        if n >= self.queue_size:
            self.queue.copy_(features[-self.queue_size:])
            self.queue_labels.copy_(labels[-self.queue_size:])
            self.queue_subject_labels.copy_(subject_labels[-self.queue_size:])
            self.queue_video_ids.copy_(video_ids[-self.queue_size:])
            self.queue_timestamps.copy_(timestamps[-self.queue_size:])
            self.queue_ptr.zero_()
            self.queue_filled.fill_(self.queue_size)
            return

        ptr = int(self.queue_ptr.item())
        end = ptr + n

        if end <= self.queue_size:
            self.queue[ptr:end] = features
            self.queue_labels[ptr:end] = labels
            self.queue_subject_labels[ptr:end] = subject_labels
            self.queue_video_ids[ptr:end] = video_ids
            self.queue_timestamps[ptr:end] = timestamps
        else:
            first = self.queue_size - ptr
            second = n - first
            self.queue[ptr:] = features[:first]
            self.queue_labels[ptr:] = labels[:first]
            self.queue_subject_labels[ptr:] = subject_labels[:first]
            self.queue_video_ids[ptr:] = video_ids[:first]
            self.queue_timestamps[ptr:] = timestamps[:first]
            self.queue[:second] = features[first:]
            self.queue_labels[:second] = labels[first:]
            self.queue_subject_labels[:second] = subject_labels[first:]
            self.queue_video_ids[:second] = video_ids[first:]
            self.queue_timestamps[:second] = timestamps[first:]

        self.queue_ptr[0] = (ptr + n) % self.queue_size
        self.queue_filled[0] = min(self.queue_size, int(self.queue_filled.item()) + n)

    @torch.no_grad()
    def get_queue(self):
        filled = int(self.queue_filled.item())
        if filled <= 0:
            empty_long = torch.zeros(0, dtype=torch.long, device=self.queue.device)
            return self.queue[:0], self.queue_labels[:0], self.queue_subject_labels[:0], torch.zeros(0, dtype=torch.long, device=self.queue.device), empty_long, empty_long
        
        ptr = int(self.queue_ptr.item())
        indices = torch.arange(filled, device=self.queue.device)
        # The newest element is at (ptr - 1) % filled.
        # Its age should be 0. We can compute age generally as:
        queue_ages = (ptr - 1 - indices) % filled
        
        return (self.queue[:filled], self.queue_labels[:filled], self.queue_subject_labels[:filled],
                queue_ages, self.queue_video_ids[:filled], self.queue_timestamps[:filled])
