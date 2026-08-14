import os
from enum import Enum
from typing import Optional, Union


class WeightTransferMethod(str, Enum):
    DIRECT = "direct"
    NCCL = "nccl"


def resolve_weight_transfer_method(
    method: Optional[Union[str, WeightTransferMethod]] = None,
) -> WeightTransferMethod:
    value = method or os.environ.get(
        "PARAS_CONFIGURE_METHOD", WeightTransferMethod.DIRECT.value
    )
    try:
        return WeightTransferMethod(value)
    except ValueError as exc:
        supported = ", ".join(item.value for item in WeightTransferMethod)
        raise ValueError(
            f"Unsupported ParaS weight transfer method {value!r}; "
            f"expected one of: {supported}"
        ) from exc
