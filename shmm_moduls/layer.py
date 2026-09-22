import tensorflow as tf
from hidten import HMMMode, HMMConfig
from hidten.config import ModelConfig, with_config
from hidten.tf import TFHMM, TFTransitioner

import numpy as np
from itertools import product


class MLPConfig(ModelConfig):

    units: int
    d_out: int | None = None
    activation_hidden: str = "relu"
    activation_out: str | None = None
    bias: bool = True


@with_config(MLPConfig)
class MLP(tf.keras.Layer):

    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.config = MLPConfig(**kwargs)
        self.norm = tf.keras.layers.LayerNormalization()
        self.f1 = tf.keras.layers.Dense(
            self.config.units,
            activation=self.config.activation_hidden,
            use_bias=self.config.bias,
        )

    def build(self, input_shape: tuple[int | None, ...]) -> None:
        self.norm.build(input_shape)
        self.f1.build(input_shape)
        self.f2 = tf.keras.layers.Dense(
            input_shape[-1],
            activation=self.config.activation_out,
            use_bias=self.config.bias,
        )
        self.f2.build(input_shape[:-1] + (self.config.units, ))
        super().build(input_shape)

    def call(self, x: tf.Tensor, training: bool = False) -> tf.Tensor:
        x = self.f2(self.f1(self.norm(x)))
        return x


class HMMBlockConfig(ModelConfig):

    embed: int
    states: int
    order: int = 1
    heads: int = 1
    parallel: int = 1
    embedIgnoreOrder: bool = True
    latent: int | None = None


@with_config(HMMBlockConfig)
class HMMBlock(tf.keras.Layer):

    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.config = HMMBlockConfig(**kwargs)
        self.states = self.config.states ** self.config.order
        self.hmm = TFHMM(states=self.states, heads=self.config.heads)
        self.norm = tf.keras.layers.LayerNormalization()
        self.embed = tf.keras.layers.Dense(
            self.config.embed if self.config.embedIgnoreOrder else (self.config.embed ** self.config.order),
            activation="softmax",
            use_bias=True,
        )
        self.history = None

    def build(self, input_shape: tuple[int | None, ...]) -> None:
        if input_shape[2] is None:
            input_shape = input_shape[:-1] + (self.config.latent, ) 
        self.norm.build(input_shape)
        self.embed.build(input_shape)

        if self.config.order > 1:
            new_states = list(product(list(range(self.config.states)), repeat=self.config.order))
            transitions = TFTransitioner()
            index = {s: i for i, s in enumerate(new_states)}
            transitions.hmm_config = HMMConfig(states=[len(new_states)] * self.config.heads)
            allowed = []
            for a in new_states:
                for b in new_states:
                    if a[1:] == b[:-1]:
                        allowed.append((index[a], index[b]))
            transitions.allow = allowed
            self.hmm.transitioner = transitions
        self.hmm.build(input_shape[:-1] + (self.config.embed if self.config.embedIgnoreOrder else (self.config.embed ** self.config.order), ))
        self.unembed = self.add_weight(
            shape=(self.config.heads, self.states, input_shape[-1]),
            dtype=tf.float32,
        )
        super().build(input_shape)

    def call(self, x, *args):

        embedded = self.embed(self.norm(x))

        y = self.hmm(
            embedded,
            mode=HMMMode.POSTERIOR,
            parallel=self.config.parallel,
        )

        y = tf.einsum("bthd,hdo->bto", y, self.unembed)

        return y
