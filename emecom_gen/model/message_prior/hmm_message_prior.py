import torch
from torch import Tensor
from torch.nn import Parameter
from torch.distributions import Categorical

from .message_prior_output import MessagePriorOutput
from .message_prior_base import MessagePriorBase


class HiddenMarkovMessagePrior(MessagePriorBase):
    def __init__(
        self,
        n_hidden_states: int,
        n_observable_states: int,
    ) -> None:
        super().__init__()
        self.n_observable_states = n_observable_states
        self.n_hidden_states = n_hidden_states

        self.init_hidden_state_weight = Parameter(torch.zeros(n_hidden_states))
        self.transition_weight = Parameter(torch.zeros(n_hidden_states, n_hidden_states))
        self.emission_weight = Parameter(torch.zeros(n_hidden_states, n_observable_states))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.normal_(self.init_hidden_state_weight)
        torch.nn.init.normal_(self.transition_weight)
        torch.nn.init.normal_(self.emission_weight)

    def forward(
        self,
        message: Tensor,
        message_length: Tensor,
    ):
        batch_size, total_length = message.shape
        device = message.device

        init_state_log_distr = self.init_hidden_state_weight.log_softmax(dim=-1)
        transition_log_distr = self.transition_weight.log_softmax(dim=-1)
        emission_log_distr = self.emission_weight.log_softmax(dim=-1)

        emission_log_probs = torch.gather(
            input=emission_log_distr.reshape(1, 1, self.n_hidden_states, self.n_observable_states).expand(
                batch_size, total_length, self.n_hidden_states, self.n_observable_states
            ),
            dim=-1,
            index=message.reshape(batch_size, total_length, 1, 1).expand(
                batch_size, total_length, self.n_hidden_states, self.n_observable_states
            ),
        ).select(dim=-1, index=0)

        step_to_log_alpha: dict[int, Tensor] = {}
        step_to_log_alpha[0] = emission_log_probs[:, 0] + init_state_log_distr

        for step in range(1, total_length):
            step_to_log_alpha[step] = emission_log_probs[:, step] + (
                step_to_log_alpha[step - 1].reshape(batch_size, self.n_hidden_states, 1)
                + transition_log_distr.reshape(1, self.n_hidden_states, self.n_hidden_states)
            ).logsumexp(dim=-2)

        message_log_probs = (
            torch.stack(list(step_to_log_alpha.values()), dim=1)
            .logsumexp(dim=-1)
            .diff(prepend=torch.zeros(size=(batch_size, 1), device=device))
        )

        return MessagePriorOutput(message_log_probs=message_log_probs)

    def sample_sequence(self):
        state = int(Categorical(logits=self.init_hidden_state_weight).sample().item())

        sequence: list[int] = []
        while len(sequence) == 0 or sequence[-1] != 0:
            sequence.append(int(Categorical(logits=self.emission_weight[state]).sample().item()))
            state = int(Categorical(logits=self.transition_weight[state]).sample().item())

        return sequence
