from pathlib import Path

import distributed
import numpy as np
import pytest
import zarr

from squisher_segment.segmentation.distributed import distributed_segmentation as segmentation


@pytest.mark.parametrize("stage", ["foreground", "completion"])
def test_segmentation_uses_owned_client_without_default(tmp_path: Path, stage: str) -> None:
    with distributed.LocalCluster(
        n_workers=1, threads_per_worker=1, processes=False, dashboard_address=None,
    ) as cluster, distributed.Client(cluster, set_as_default=False) as client:
        with pytest.raises(ValueError, match="No clients found"):
            distributed.default_client()
        if stage == "foreground":
            image = zarr.create_array(
                tmp_path / "input.zarr",
                data=np.array([1000, 1001], dtype=np.uint16).reshape(1, 1, 2, 1),
                chunks=(1, 1, 1, 1),
            )
            selected = segmentation._select_input_blocks(
                client=client,
                block_crops=[(slice(0, 1), slice(0, 1), slice(i, i + 1), slice(0, 1)) for i in range(2)],
                input_zarr=image, nonempty_channel_index=0, nonempty_threshold=1000,
                blocksize=(1, 1, 1, 1), run_key="owned-client", path_nonempty=tmp_path / "nonempty.json",
            )
            assert selected == [1]
        else:
            future = client.submit(dict, index=(0, 0, 0, 0), worker="cpu", duration_s=1.0, n_masks=2)
            checkpoint = tmp_path / "checkpoint.jsonl"
            assert segmentation._wait_for_futures_retry_once(
                client=client, futures=[future], future_labels={future: "block"},
                stage="Segmentation", log=segmentation.logger, checkpoint_path=checkpoint,
            ) == []
            assert segmentation.load_checkpoint(checkpoint) == {(0, 0, 0, 0)}
