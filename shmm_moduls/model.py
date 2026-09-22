import tensorflow as tf
from hidten.config import ModelConfig, with_config

from .layer import MLP, HMMBlock, HMMBlockConfig, MLPConfig


class RNNLayerConfig(ModelConfig):
    """Konfiguration für optionalen RNN-Layer (LSTM/GRU/SimpleRNN)."""
    
    type: str  # "lstm", "gru", "rnn"
    units: int
    return_sequences: bool = True
    bidirectional: bool = False
    # Bei bidirectional=True kombiniert Keras standardmäßig via "concat" und
    # verdoppelt damit die letzte Dimension (2*units). Damit die Ausgabe
    # weiterhin zu `latent` passt und direkt an die nachfolgenden HMM-/MLP-
    # Blöcke übergeben werden kann, wird stattdessen gemittelt ("ave"),
    # sodass die Ausgabedimension in jedem Fall `units` beträgt.
    merge_mode: str = "ave"
 
 
@with_config(RNNLayerConfig)
class RNNLayer(tf.keras.Layer):
    """Wrapper um verschiedene RNN-Typen, als optionales erstes Layer."""
    
    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.config = RNNLayerConfig(**kwargs)
        
        rnn_type = self.config.type.lower()
        if rnn_type == "lstm":
            rnn_cell = tf.keras.layers.LSTM(
                self.config.units,
                return_sequences=self.config.return_sequences,
            )
        elif rnn_type == "gru":
            rnn_cell = tf.keras.layers.GRU(
                self.config.units,
                return_sequences=self.config.return_sequences,
            )
        elif rnn_type == "rnn":
            rnn_cell = tf.keras.layers.SimpleRNN(
                self.config.units,
                return_sequences=self.config.return_sequences,
            )
        else:
            raise ValueError(f"Unbekannter RNN-Typ: {rnn_type}. "
                             f"Verwende 'lstm', 'gru' oder 'rnn'.")
        
        if self.config.bidirectional:
            self.rnn = tf.keras.layers.Bidirectional(rnn_cell, merge_mode=self.config.merge_mode)
        else:
            self.rnn = rnn_cell

    def build(self, input_shape: tuple[int | None, ...]) -> None:
        # Ohne diese explizite build()-Methode markiert Kerass Standard-
        # Layer.build() den Layer als "built", ohne das intern gewrappte
        # self.rnn (LSTM/GRU/SimpleRNN bzw. Bidirectional) tatsächlich zu
        # bauen ("UserWarning: ... does not have a build() method
        # implemented"). Das self.rnn-Sublayer wird deshalb hier explizit
        # mit der bekannten Input-Shape gebaut.
        self.rnn.build(input_shape)
        super().build(input_shape)

    def call(self, x: tf.Tensor, training: bool = False) -> tf.Tensor:
        return self.rnn(x, training=training)


class TransformerLayerConfig(ModelConfig):
    """Konfiguration für optionalen Transformer-Layer."""
    
    d_model: int = 64          # Embedding-Dimension
    num_heads: int = 4         # Anzahl der Attention-Heads
    num_layers: int = 2        # Anzahl der Transformer-Blöcke
    dff: int = 256             # Feedforward-Dimension
    dropout: float = 0.1       # Dropout-Rate
    max_seq_length: int = 512  # Max Sequenz-Länge für Positional Encoding


@with_config(TransformerLayerConfig)
class TransformerBlock(tf.keras.Layer):
    """Ein einzelner Transformer-Block (Attention + FFN)."""
    
    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.config = TransformerLayerConfig(**kwargs)
        
        self.mha = tf.keras.layers.MultiHeadAttention(
            num_heads=self.config.num_heads,
            key_dim=self.config.d_model // self.config.num_heads,
            dropout=self.config.dropout,
        )
        
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(self.config.dff, activation='relu'),
            tf.keras.layers.Dense(self.config.d_model),
        ])
        
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(self.config.dropout)
        self.dropout2 = tf.keras.layers.Dropout(self.config.dropout)
    
    def call(self, x: tf.Tensor, training: bool = False) -> tf.Tensor:
        # Multi-Head Attention
        attn_output = self.mha(x, x, training=training)
        attn_output = self.dropout1(attn_output, training=training)
        x = self.layernorm1(x + attn_output)
        
        # Feedforward
        ffn_output = self.ffn(x)
        ffn_output = self.dropout2(ffn_output, training=training)
        x = self.layernorm2(x + ffn_output)
        
        return x


class TransformerConfig(ModelConfig):
    """Konfiguration für Stand-Alone Transformer-Modell."""
    
    d_model: int = 64
    num_heads: int = 4
    num_layers: int = 2
    dff: int = 256
    dropout: float = 0.1
    output: int = 9
    max_seq_length: int = 512


@with_config(TransformerConfig)
class TransformerModel(tf.keras.Model):
    """Stand-Alone Transformer für Token-Level-Klassifizierung."""
    
    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.config = TransformerConfig(**kwargs)
        
        # Input: [B, T, 2] → embedding zu d_model
        self.embed = tf.keras.layers.Dense(self.config.d_model)
        
        # Positional Encoding
        self.pos_encoding = self._get_positional_encoding(
            self.config.max_seq_length,
            self.config.d_model
        )
        
        self.dropout = tf.keras.layers.Dropout(self.config.dropout)
        
        # Transformer Blöcke
        self.transformer_blocks = [
            TransformerBlock(
                d_model=self.config.d_model,
                num_heads=self.config.num_heads,
                dff=self.config.dff,
                dropout=self.config.dropout,
                max_seq_length=self.config.max_seq_length,
            )
            for _ in range(self.config.num_layers)
        ]
        
        # Output: d_model → output_classes
        self.unembed = tf.keras.layers.Dense(self.config.output)
    
    def _get_positional_encoding(self, max_len: int, d_model: int) -> tf.Tensor:
        """Generiere Positional Encoding (static) via sin/cos interleaving."""
        position = tf.range(max_len, dtype=tf.float32)[:, tf.newaxis]      # [max_len, 1]
        div_term = tf.exp(
            tf.range(0, d_model, 2, dtype=tf.float32) *
            -(tf.math.log(10000.0) / d_model)
        )  # [d_model/2]

        pos_sin = tf.sin(position * div_term)  # [max_len, d_model/2]
        pos_cos = tf.cos(position * div_term)  # [max_len, d_model/2]

        # Interleave sin/cos entlang der letzten Achse:
        # [max_len, d_model/2, 2] -> [max_len, d_model]
        pe = tf.stack([pos_sin, pos_cos], axis=-1)
        pe = tf.reshape(pe, (max_len, d_model))

        return tf.cast(pe[tf.newaxis, ...], tf.float32)
    
    def build(self, input_shape: tuple[int | None, ...]) -> None:
        self.embed.build(input_shape)
        # transformer_blocks bauen sich selbst
        _, T, _ = input_shape
        self.unembed.build((None, T, self.config.d_model))
    
    def call(self, x: tf.Tensor, training: bool = False, return_hidden: bool = False) -> tf.Tensor:
        # Embed
        x = self.embed(x)

        # Add positional encoding
        seq_len = tf.shape(x)[1]
        x = x + self.pos_encoding[:, :seq_len, :]
        x = self.dropout(x, training=training)

        # Pass durch Transformer Blöcke
        for block in self.transformer_blocks:
            x = block(x, training=training)

        # x ist hier die versteckte Repräsentation vor dem Unembed,
        # Shape [B, T, d_model] - wird z.B. für Pooling-Köpfe der
        # Sequenz-Klassifizierung benötigt (siehe training.py).
        if return_hidden:
            return x

        # Unembed
        x = self.unembed(x)
        return x


class MLPOnlyConfig(ModelConfig):
    """Konfiguration für ein reines MLP-Modell (Baseline ohne HMM/RNN/Transformer)."""

    layers: int = 1
    latent: int = 16
    mlp: MLPConfig
    output: int = 9


@with_config(MLPOnlyConfig)
class MLPOnlyModel(tf.keras.Model):
    """
    Baseline-Modell: nur Embed -> N residuale MLP-Blöcke -> Unembed.
    Kein HMM, kein RNN, kein Transformer - dient als einfacher Vergleichs-
    maßstab, um zu prüfen, wie viel die anderen Architekturen tatsächlich
    beitragen.

    Arbeitet Position-für-Position (wie ein Dense-Layer, das auf jeden
    Zeitschritt einzeln angewendet wird) - es gibt also KEINEN Informations-
    austausch zwischen verschiedenen Zeitschritten der Sequenz.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.config = MLPOnlyConfig(**kwargs)

        self.embed = tf.keras.layers.Dense(
            self.config.latent,
            use_bias=False,
        )
        self.unembed = tf.keras.layers.Dense(
            self.config.output,
            use_bias=False,
        )

        self.mlps = [MLP(**self.config.mlp.model_dump()) for _ in range(self.config.layers)]

    def build(self, input_shape: tuple[int | None, ...]) -> None:
        self.embed.build(input_shape)
        BT = input_shape[:-1]
        hidden_shape = BT + (self.config.latent,)

        for mlp in self.mlps:
            mlp.build(hidden_shape)

    def call(self, x: tf.Tensor, training: bool = False, return_hidden: bool = False) -> tf.Tensor:
        x = self.embed(x)
        for mlp in self.mlps:
            x = x + mlp(x, training=training)
        # x ist hier die versteckte Repräsentation vor dem Unembed,
        # Shape [B, T, latent] - wird z.B. für Pooling-Köpfe der
        # Sequenz-Klassifizierung benötigt (siehe training.py).
        if return_hidden:
            return x
        x = self.unembed(x)
        return x


class ResidualCRFConfig(ModelConfig):

    layers: int
    latent: int
    mlp: MLPConfig | None = None
    hmm: HMMBlockConfig | list[HMMBlockConfig]
    rnn: RNNLayerConfig | None = None

    output: int


@with_config(ResidualCRFConfig)
class ResidualCRF(tf.keras.Model):

    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.config = ResidualCRFConfig(**kwargs)

        self.embed = tf.keras.layers.Dense(
            self.config.latent,
            use_bias=False,
        )
        self.unembed = tf.keras.layers.Dense(
            self.config.output,
            use_bias=False,
        )

        # Optionales RNN (LSTM/GRU/SimpleRNN) als erstes Layer nach dem
        # Embedding, z.B. für "lstm_hmm_order1" oder "bilstm_hmm_order1".
        # Ist self.config.rnn nicht gesetzt (Standardfall, z.B. reine
        # HMM-Modelle), bleibt self.rnn_layer None und wird in call()/
        # build() übersprungen.
        self.rnn_layer = RNNLayer(**self.config.rnn.model_dump()) if self.config.rnn is not None else None

        self.hmms = []
        self.mlps = []
        for i in range(self.config.layers):
            if isinstance(self.config.hmm, list):
                self.hmms.append(HMMBlock(**self.config.hmm[i].model_dump()))
            elif isinstance(self.config.hmm, HMMBlockConfig):
                self.hmms.append(HMMBlock(**self.config.hmm.model_dump()))
            else:
                raise ValueError("no supported hmm config in the given configurations")
            if self.config.mlp is not None:
                self.mlps.append(MLP(**self.config.mlp.model_dump()))

    def build(self, input_shape: tuple[int | None, ...]) -> None:
        self.embed.build(input_shape)
        BT = input_shape[:-1]

        # Nach dem RNN (falls vorhanden) beträgt die letzte Dimension
        # `rnn.units` (bei bidirectional=True dank merge_mode="ave"
        # weiterhin `units`, nicht 2*units) statt `latent`. Die
        # nachfolgenden HMM-/MLP-Blöcke müssen also mit dieser tatsächlichen
        # Dimension gebaut werden, nicht pauschal mit `latent`.
        if self.rnn_layer is not None:
            self.rnn_layer.build(BT + (self.config.latent, ))
            D = (self.config.rnn.units, )
        else:
            D = (self.config.latent, )
        hidden_shape = BT + D

        for i in range(self.config.layers):
            self.hmms[i].build(hidden_shape)
            if self.mlps:
                self.mlps[i].build(hidden_shape)


    def call(self, x: tf.Tensor, training: bool = False, return_hidden: bool = False):
        x = self.embed(x)
        if self.rnn_layer is not None:
            x = self.rnn_layer(x, training=training)
        for i in range(self.config.layers):
            hmm_out = self.hmms[i](x)
            x = x + hmm_out
            if self.mlps:
                x = x + self.mlps[i](x)

        # x ist hier die versteckte Repräsentation vor dem Unembed,
        # Shape [B, T, latent] - wird z.B. für Pooling-Köpfe der
        # Sequenz-Klassifizierung benötigt (siehe training.py).
        if return_hidden:
            return x

        x = self.unembed(x)
        return x


def get_model(
    T: int,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    transformer: dict | None = None,
    mlp_only: dict | None = None,
    num_symbols: int = 2,
    **kwargs,
) -> tf.keras.Model:
    """
    Baue ein Modell (ResidualCRF, Transformer oder reines MLP-Baseline-Modell).

    Args:
        T: Sequenz-Länge
        learning_rate: Learning Rate
        weight_decay: Weight Decay
        transformer: Dict mit Transformer-Config, wenn Transformer gewünscht
        mlp_only: Dict mit MLP-Baseline-Config, wenn reines MLP gewünscht
            (keys: "layers", "latent", "units", "activation_hidden")
        num_symbols: Größe des Emissionsalphabets, d.h. die Breite der
            One-Hot-kodierten Eingabe (Standard: 2, für Rückwärts-
            kompatibilität mit rein binären Emissionen). MUSS mit der
            tatsächlichen One-Hot-Tiefe der Eingabedaten übereinstimmen,
            da hiermit die Eingabe-Shape des Embed-Layers festgelegt wird.
        **kwargs: Alle anderen Config-Parameter

    Returns:
        Ein kompiliertes Keras-Modell
    """
    
    if transformer is not None:
        # Baue Transformer-Modell
        model = TransformerModel(
            d_model=transformer.get("d_model", 64),
            num_heads=transformer.get("num_heads", 4),
            num_layers=transformer.get("num_layers", 2),
            dff=transformer.get("dff", 256),
            dropout=transformer.get("dropout", 0.1),
            output=kwargs.get("output", 9),
            max_seq_length=T,
        )
    elif mlp_only is not None:
        # Baue reines MLP-Baseline-Modell (kein HMM/RNN/Transformer)
        model = MLPOnlyModel(
            layers=mlp_only.get("layers", 1),
            latent=mlp_only.get("latent", 16),
            mlp={
                "units": mlp_only.get("units", 32),
                "activation_hidden": mlp_only.get("activation_hidden", "relu"),
            },
            output=kwargs.get("output", 9),
        )
    else:
        # Baue ResidualCRF (Standard)
        model = ResidualCRF(**kwargs)

    model.build((None, T, num_symbols))
    model.compile(
        optimizer=tf.optimizers.AdamW(
            learning_rate=learning_rate,
            weight_decay=weight_decay,
        ),
        loss=tf.losses.SparseCategoricalCrossentropy(
            from_logits=True,
        ),
        metrics=["accuracy"],
    )

    return model