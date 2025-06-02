import torch
import torch.nn.functional as F
import numpy as np
import quest.utils.tensor_utils as TensorUtils
import itertools

from lerobot.common.policies.pretrained import PreTrainedPolicy
from quest.algos.base import ChunkPolicy

from lerobot.custom_scripts.common.policies.quest.configuring_quest import QuestConfig


class QuestPolicy(PreTrainedPolicy):
    config_class = QuestConfig
    name = "Quest"

    def __init__(self,
                 config: QuestConfig,
                 dataset_stats: dict[str, dict[str, Tensor]] | None = None,
                 ):
        super().__init__(config)
        config.validate_features()
        self.config = config

        self.normalize_inputs = Normalize(config.input_features, config.normalization_mapping, dataset_stats)
        self.normalize_targets = Normalize(
            config.output_features, config.normalization_mapping, dataset_stats
        )
        self.unnormalize_outputs = Unnormalize(
            config.output_features, config.normalization_mapping, dataset_stats
        )

        self.model = QueST(config)
        self.reset()

    @abc.abstractmethod
    def get_optim_params(self) -> dict:
        return self.parameters()

    @abc.abstractmethod
    def reset(self):
        self._action_queue = deque([], maxlen=self.config.n_action_steps)

    @abc.abstractmethod
    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict | None]:
        if self.config.is_training:
            if self.config.stage == 0:
                self.model.autoencoder()
            elif self.config.stage == 1:

            elif self.config.stage == 2:

            else:
                raise ValueError("Unsupported stage %d" % self.config.stage)


    def get_optimizers(self):
        if self.stage == 0:
            decay, no_decay = TensorUtils.separate_no_decay(self.autoencoder)
            optimizers = [
                self.optimizer_factory(params=decay),
                self.optimizer_factory(params=no_decay, weight_decay=0.)
            ]
            return optimizers
        elif self.stage == 1:
            decay, no_decay = TensorUtils.separate_no_decay(self,
                                                            name_blacklist=('autoencoder',))
            optimizers = [
                self.optimizer_factory(params=decay),
                self.optimizer_factory(params=no_decay, weight_decay=0.)
            ]
            return optimizers
        elif self.stage == 2:
            decay, no_decay = TensorUtils.separate_no_decay(self,
                                                            name_blacklist=('autoencoder',))
            decoder_decay, decoder_no_decay = TensorUtils.separate_no_decay(self.autoencoder.decoder)
            optimizers = [
                self.optimizer_factory(params=itertools.chain(decay, decoder_decay)),
                self.optimizer_factory(params=itertools.chain(no_decay, decoder_no_decay), weight_decay=0.)
            ]
            return optimizers

    def get_context(self, data):
        obs_emb = self.obs_encode(data)
        task_emb = self.get_task_emb(data).unsqueeze(1)
        context = torch.cat([task_emb, obs_emb], dim=1)
        return context

    def compute_loss(self, data):
        if self.stage == 0:
            return self.compute_autoencoder_loss(data)
        elif self.stage == 1:
            return self.compute_prior_loss(data)
        elif self.stage == 2:
            return self.compute_prior_loss(data)

    def compute_autoencoder_loss(self, data):
        pred, pp, pp_sample, aux_loss, _ = self.autoencoder(data["actions"])
        recon_loss = self.loss(pred, data["actions"])
        if self.autoencoder.vq_type == 'vq':
            loss = recon_loss + aux_loss
        else:
            loss = recon_loss

        info = {
            'loss': loss.item(),
            'recon_loss': recon_loss.item(),
            'aux_loss': aux_loss.sum().item(),
            'pp': pp.item(),
            'pp_sample': pp_sample.item(),
        }
        return loss, info

    def compute_prior_loss(self, data):
        data = self.preprocess_input(data, train_mode=True)
        with torch.no_grad():
            indices = self.autoencoder.get_indices(data["actions"]).long()
        context = self.get_context(data)
        start_tokens = (torch.ones((context.shape[0], 1), device=self.device, dtype=torch.long) * self.start_token)
        x = torch.cat([start_tokens, indices[:, :-1]], dim=1)
        targets = indices.clone()
        logits = self.policy_prior(x, context)
        prior_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        with torch.no_grad():
            logits = logits[:, :, :self.codebook_size]
            probs = torch.softmax(logits, dim=-1)
            sampled_indices = torch.multinomial(probs.view(-1, logits.shape[-1]), 1)
            sampled_indices = sampled_indices.view(-1, logits.shape[1])

        pred_actions = self.autoencoder.decode_actions(sampled_indices)
        l1_loss = self.loss(pred_actions, data["actions"])
        total_loss = prior_loss + self.l1_loss_scale * l1_loss
        info = {
            'loss': total_loss.item(),
            'nll_loss': prior_loss.item(),
            'l1_loss': l1_loss.item()
        }
        return total_loss, info

    def sample_actions(self, data):
        data = self.preprocess_input(data, train_mode=False)
        context = self.get_context(data)
        sampled_indices = self.policy_prior.get_indices_top_k(context, self.codebook_size)
        pred_actions = self.autoencoder.decode_actions(sampled_indices)
        pred_actions = pred_actions.permute(1, 0, 2)
        return pred_actions.detach().cpu().numpy()