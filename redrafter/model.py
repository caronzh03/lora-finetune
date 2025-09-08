import torch
import torch.nn as nn

from redrafter.modeling_drafter import Drafter


class ReDrafter(nn.Module):
    def __init__(
        self,
        llm: nn.Module,
        drafter: Drafter,
    ):
        super().__init__()
        self.llm = llm
        self.drafter = drafter

    def forward(
        self,
        in_idx=None,
        next_n=1,
    ):
        """Forward pass of the ReDrafter.

        Args:
            in_idx (torch.Tensor, optional): Input token IDs.
            next_n (int, optional): Number of future tokens to predict using the drafter. Defaults to 1.

        Returns:
            torch.Tensor: A tensor containing predictions from ReDrafter.
        """
        with torch.inference_mode():
            outputs = self.llm.hidden_states(in_idx) # [batch_size, seq_len, hidden_dim] = [8, 62, 1024]

        # Clone the output hidden states
        hidden_states = outputs.clone()  # [batch_size, seq_len, hidden_dim] = [8, 62, 1024]
        drafter_logits = []
        input_embs = self.llm.tok_emb(in_idx)  # [batch_size, seq_len, emb_dim] = [8, 62, 1024]
        h, cumsum_input_embs = hidden_states, torch.zeros_like(
            input_embs, dtype=input_embs.dtype, device=input_embs.device
        )
        # Drafter forward function
        # ht: hidden_states at position t
        # et: input_embs at position t
        # [lt]_i: drafter_logits at iteration i predicting token t

        # Iteration 1:
        # [h0,e1] [h1,e2] ... [h_{n-2},e_{n-1}] [h_{n-1},e0]
        #                 V
        #            drafter head
        #                 V
        # [l2]_1  [l3]_1  ... [ln]_1            [l_{n+1}]_1  --> drafter_logits[0]
        #
        # Iteration 2:
        # [h0,e1+e2] [h1,e2+e3] ... [h_{n-2},e_{n-1}+e0] [h_{n-1},e0+e1]
        #                 V
        #            drafter head
        #                 V
        # [l3]_2     [l4]_2     ... [l_{n+1}]_2          [l_{n+3}]_2  --> drafter_logits[1]
        # ...

        # Stack drafter_logits as the return value.
        # E.g. when next_n = 5,
        # [[[l2]_1, [l3]_1, ..., [l_{n+1}]_1],
        #  [[l3]_2, [l4]_2, ..., [l_{n+2}]_2],
        #  ...
        #  [[l6]_5, [l7]_5, ..., [l_{n+5}]_5]],
        for _ in range(next_n):
            input_embs = torch.roll(input_embs, -1, dims=1)
            if self.drafter.config.rnn:
                # s = f(U * s + W * w + b).
                o = self.drafter.rnn_u(cumsum_input_embs)
                cumsum_input_embs = nn.SiLU()(o + self.drafter.rnn_w(input_embs))
            else:
                cumsum_input_embs += input_embs
            h = torch.cat((hidden_states, cumsum_input_embs), -1) # [batch_size, seq_len, hidden_dim + emb_dim] = [8, 62, 2048]
            drafter_logits.append(self.drafter.lm_head(h)) # each drafter_logits = [batch_size, seq_len, vocab_size] = [8, 62, 151936]
        return torch.stack(drafter_logits, dim=0) # [next_n, batch_size, seq_len, vocab_size]
