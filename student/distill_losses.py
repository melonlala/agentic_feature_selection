"""Loss functions for behavioral cloning and soft distillation.

Two training objectives are provided:

1. cross_entropy_loss: Standard multi-class cross entropy against the
   teacher's greedy action labels. This is the default hard-label BC loss.

2. kl_distill_loss: KL divergence between student logits (as distribution)
   and teacher softmax probabilities (soft labels). Direction:
       KL(teacher_probs || student_probs)
   i.e., we minimise the extra bits needed to encode teacher's distribution
   using the student's distribution. This penalises the student for giving
   low probability to actions the teacher assigns high probability to.

3. combined_loss: Weighted combination:
       L = (1 - alpha) * CE + alpha * KL

The soft distillation loss is inspired by knowledge distillation (Hinton et al.
2015). Using alpha > 0 can improve generalisation when the teacher's soft
probabilities carry richer supervision signal than the hard argmax.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def cross_entropy_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Standard cross-entropy loss against hard action labels.

    Args:
        logits: Action logits from student, shape [N, n_actions].
        labels: Teacher greedy action labels, shape [N] (int64).

    Returns:
        Scalar mean cross-entropy loss.
    """
    return F.cross_entropy(logits, labels)


def kl_distill_loss(
    logits: torch.Tensor,
    teacher_probs: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    """KL divergence distillation loss.

    Computes KL(teacher_probs || student_probs) where student_probs are
    derived from logits via softmax at the given temperature.

    KL direction: KL(P || Q) = sum P * log(P / Q), where P = teacher and
    Q = student. This penalises the student for mass in regions where the
    teacher has low probability (and vice versa).

    Args:
        logits: Student action logits, shape [N, n_actions].
        teacher_probs: Teacher softmax probabilities, shape [N, n_actions].
        temperature: Temperature for student softmax (default 1.0).

    Returns:
        Scalar mean KL divergence.
    """
    # Student log-probabilities
    student_log_probs = F.log_softmax(logits / temperature, dim=-1)
    # Teacher probabilities (clamp for numerical stability)
    teacher_probs = teacher_probs.clamp(min=1e-8)
    # KL divergence: E[P * (log P - log Q)] = E[P * log P] - E[P * log Q]
    # F.kl_div expects (input=log Q, target=P) and computes sum P*(log P - log Q)
    kl = F.kl_div(student_log_probs, teacher_probs, reduction="batchmean", log_target=False)
    return kl


def combined_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    teacher_probs: torch.Tensor,
    alpha: float = 0.5,
    temperature: float = 1.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Combined hard + soft distillation loss.

    L = (1 - alpha) * CE(logits, labels) + alpha * KL(teacher || student)

    Args:
        logits: Student logits, shape [N, n_actions].
        labels: Hard teacher action labels, shape [N].
        teacher_probs: Teacher softmax probabilities, shape [N, n_actions].
        alpha: Weight for KL loss (0 = pure CE, 1 = pure KL).
        temperature: Softmax temperature for KL term.

    Returns:
        Tuple of (total_loss, {ce_loss, kl_loss}).
    """
    ce = cross_entropy_loss(logits, labels)
    kl = kl_distill_loss(logits, teacher_probs, temperature=temperature)
    total = (1.0 - alpha) * ce + alpha * kl
    return total, {"ce_loss": ce.item(), "kl_loss": kl.item()}
