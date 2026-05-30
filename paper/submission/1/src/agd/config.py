from dataclasses import dataclass


@dataclass
class AGDConfig:
    """Configuration for AGD pretraining experiments."""

    vocab_size: int = 6400
    hidden_size: int = 768
    num_hidden_layers: int = 8
    num_attention_heads: int = 8
    intermediate_size: int = 2048
    max_position_embeddings: int = 512
    tie_word_embeddings: bool = True
    lm_head_bias: bool = False

    # AGD variant: s1 (tied baseline), s2 (full untied), s3 (bias-only),
    # s6 (hidden low-rank only), s12 (bias + hidden low-rank, main AGD input-side)
    variant: str = "s1"
    variant_rank: int = 32

    def __post_init__(self) -> None:
        self.variant = self.variant.lower()
