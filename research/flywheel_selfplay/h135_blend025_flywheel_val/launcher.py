"""A3 operational wrapper; the imported H135/H124/H129 science is unchanged."""

from pathlib import Path
import runpy
from unittest.mock import patch


def worker_environment(base):
    return dict(
        base,
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        NUMEXPR_NUM_THREADS="1",
        NUMEXPR_MAX_THREADS="1",
        RAYON_NUM_THREADS="1",
        JE_ARROW_MALLOC_CONF="background_thread:false",
        TOKENIZERS_PARALLELISM="false",
        CUDA_VISIBLE_DEVICES="",
        PYTHONDONTWRITEBYTECODE="1",
    )


def one_thread(call):
    # Load native libraries before limiting them; environment is set by the parent.
    import sklearn.ensemble  # noqa: F401
    import threadpoolctl

    native_limit = threadpoolctl.threadpool_limits

    def capped_limit(limits=None, user_api=None):
        # The unchanged runner requests OpenMP4 inside fit. Keep that scoped
        # context, but cap it at1 so HistGB binning also selects joblib n_jobs=1.
        return native_limit(limits=1, user_api=user_api)

    with (
        native_limit(limits=1),
        patch.object(threadpoolctl, "threadpool_limits", capped_limit),
    ):
        return call()


if __name__ == "__main__":
    one_thread(
        lambda: runpy.run_path(
            str(Path(__file__).with_name("runner.py")), run_name="__main__"
        )
    )
