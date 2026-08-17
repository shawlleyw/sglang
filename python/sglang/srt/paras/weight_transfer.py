import os
from enum import Enum
from typing import Optional, Union


class IntraNodeWeightTransferMethod(str, Enum):
    PEER_ACCESS = "peer_access"
    NCCL = "nccl"


def resolve_intra_node_weight_transfer_method(
    method: Optional[Union[str, IntraNodeWeightTransferMethod]] = None,
) -> IntraNodeWeightTransferMethod:
    value = method or os.environ.get(
        "PARAS_INTRA_NODE_WEIGHT_TRANSFER_METHOD",
        IntraNodeWeightTransferMethod.PEER_ACCESS.value,
    )
    try:
        return IntraNodeWeightTransferMethod(value)
    except ValueError as exc:
        supported = ", ".join(item.value for item in IntraNodeWeightTransferMethod)
        raise ValueError(
            f"Unsupported ParaS intra-node weight transfer method {value!r}; "
            f"expected one of: {supported}"
        ) from exc
