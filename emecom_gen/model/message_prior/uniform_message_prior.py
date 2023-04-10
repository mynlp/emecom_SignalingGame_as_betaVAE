from .length_exponential_message_prior import LengthExponentialMessagePrior


class UniformMessagePrior(LengthExponentialMessagePrior):
    def __init__(
        self,
        vocab_size: int,
        max_len: int,
    ) -> None:
        super().__init__(
            vocab_size=vocab_size,
            max_len=max_len,
            base=1,
        )
