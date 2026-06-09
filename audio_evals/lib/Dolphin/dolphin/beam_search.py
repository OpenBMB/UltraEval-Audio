import torch
from typing import Any, Dict, List, NamedTuple, Tuple, Union
from espnet.nets.beam_search import BeamSearch, Hypothesis
from espnet.nets.batch_beam_search import BatchBeamSearch
from espnet.nets.e2e_asr_common import end_detect

import logging

logger = logging.getLogger("dolphin")


class SimpleBeamSearch(BatchBeamSearch):
    """Simple beam search decoder.
    Just when the hyps equals to beam size, it will stop decoding."""

    def forward(
        self,
        x: torch.Tensor,
        maxlenratio: float = 0.0,
        minlenratio: float = 0.0,
        pre_x: torch.Tensor = None,
    ) -> List[Hypothesis]:
        """Perform beam search.

        Args:
            x (torch.Tensor): Encoded speech feature (T, D)
            maxlenratio (float): Input length ratio to obtain max output length.
                If maxlenratio=0.0 (default), it uses a end-detect function
                to automatically find maximum hypothesis lengths
                If maxlenratio<0.0, its absolute value is interpreted
                as a constant max output length.
            minlenratio (float): Input length ratio to obtain min output length.
                If minlenratio<0.0, its absolute value is interpreted
                as a constant min output length.
            pre_x (torch.Tensor): Encoded speech feature for sequential attn (T, D)
                Sequential attn computes attn first on pre_x then on x,
                thereby attending to two sources in sequence.

        Returns:
            list[Hypothesis]: N-best decoding results

        """
        logger.debug("Simple beam search decoding...")
        # set length bounds
        if pre_x is not None:
            inp = pre_x
        else:
            inp = x
        if maxlenratio == 0:
            maxlen = inp.shape[0]
        elif maxlenratio < 0:
            maxlen = -1 * int(maxlenratio)
        else:
            maxlen = max(1, int(maxlenratio * inp.size(0)))

        if minlenratio < 0:
            minlen = -1 * int(minlenratio)
        else:
            minlen = int(minlenratio * inp.size(0))
        logger.debug("decoder input length: " + str(inp.shape[0]))
        logger.debug("max output length: " + str(maxlen))
        logger.debug("min output length: " + str(minlen))

        # main loop of prefix search
        running_hyps = self.init_hyp(x if pre_x is None else pre_x)
        ended_hyps = []
        for i in range(maxlen):
            logger.debug("position " + str(i))
            best = self.search(running_hyps, x, pre_x=pre_x)
            # post process of one iteration
            running_hyps = self.post_process(
                i, maxlen, minlen, maxlenratio, best, ended_hyps
            )
            # simple beam search stop condition
            if len(ended_hyps) >= self.beam_size:
                logger.debug(f"end detected for beam sizs: {self.beam_size}")
                break
            # end detection
            if maxlenratio == 0.0 and end_detect([h.asdict() for h in ended_hyps], i):
                logger.debug(f"end detected at {i}")
                break
            if len(running_hyps) == 0:
                logger.debug("no hypothesis. Finish decoding.")
                break
            else:
                logger.debug(f"remained hypotheses: {len(running_hyps)}")

        if self.normalize_length:
            # Note (Jinchuan): -1 since hyp starts with <sos> and
            # initially has score of 0.0
            nbest_hyps = sorted(
                ended_hyps, key=lambda x: x.score / (len(x.yseq) - 1), reverse=True
            )
        else:
            nbest_hyps = sorted(ended_hyps, key=lambda x: x.score, reverse=True)

        # check the number of hypotheses reaching to eos
        if len(nbest_hyps) == 0:
            logger.warning(
                "there is no N-best results, perform recognition "
                "again with smaller minlenratio."
            )
            return (
                []
                if minlenratio < 0.1
                else self.forward(x, maxlenratio, max(0.0, minlenratio - 0.1))
            )

        # report the best result
        best = nbest_hyps[0]
        for k, v in best.scores.items():
            logger.debug(
                f"{v:6.2f} * {self.weights[k]:3} = {v * self.weights[k]:6.2f} for {k}"
            )
        logger.debug(f"total log probability: {best.score:.2f}")
        logger.debug(f"normalized log probability: {best.score / len(best.yseq):.2f}")
        logger.debug(f"total number of ended hypotheses: {len(nbest_hyps)}")
        if self.token_list is not None:
            logger.debug(
                "best hypo: "
                + "".join([self.token_list[x] for x in best.yseq[1:-1]])
                + "\n"
            )
        if best.yseq[1:-1].shape[0] == maxlen:
            logger.warning(
                "best hypo length: {} == max output length: {}".format(
                    best.yseq[1:-1].shape[0], maxlen
                )
            )
            logger.warning(
                "decoding may be stopped by the max output length limitation, "
                + "please consider to increase the maxlenratio."
            )
        return nbest_hyps
