from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest
import pytz

import vectorbt as vbt
from vectorbt.utils.config import merge_dicts
from vectorbt.utils.datetime_ import to_timezone

seed = 42


# ############# Global ############# #


def setup_module():
    vbt.settings.numba["check_func_suffix"] = True
    vbt.settings.caching.enabled = False
    vbt.settings.caching.whitelist = []
    vbt.settings.caching.blacklist = []


def teardown_module():
    vbt.settings.reset()


# ############# base.py ############# #


class MyData(vbt.Data):
    @classmethod
    def download_symbol(
        cls,
        symbol,
        shape=(5, 3),
        start_date=datetime(2020, 1, 1),
        columns=None,
        index_mask=None,
        column_mask=None,
        return_arr=False,
        tz_localize=None,
        seed=seed,
    ):
        np.random.seed(seed)
        a = np.random.uniform(size=shape) + symbol
        if return_arr:
            return a
        index = [start_date + timedelta(days=i) for i in range(a.shape[0])]
        if a.ndim == 1:
            sr = pd.Series(a, index=index, name=columns)
            if index_mask is not None:
                sr = sr.loc[index_mask]
            if tz_localize is not None:
                sr = sr.tz_localize(tz_localize)
            return sr
        df = pd.DataFrame(a, index=index, columns=columns)
        if index_mask is not None:
            df = df.loc[index_mask]
        if column_mask is not None:
            df = df.loc[:, column_mask]
        if tz_localize is not None:
            df = df.tz_localize(tz_localize)
        return df

    def update_symbol(self, symbol, n=1, **kwargs):
        download_kwargs = self.select_symbol_kwargs(symbol, self.download_kwargs)
        download_kwargs["start_date"] = self.data[symbol].index[-1]
        shape = download_kwargs.pop("shape", (5, 3))
        new_shape = (n, shape[1]) if len(shape) > 1 else (n,)
        new_seed = download_kwargs.pop("seed", seed) + 1
        kwargs = merge_dicts(download_kwargs, kwargs)
        return self.download_symbol(symbol, shape=new_shape, seed=new_seed, **kwargs)


class TestData:
    def test_config(self, tmp_path):
        data = MyData.download([0, 1], shape=(5, 3), columns=["feat0", "feat1", "feat2"])
        assert MyData.loads(data.dumps()) == data
        data.save(tmp_path / "data")
        assert MyData.load(tmp_path / "data") == data

    def test_download(self):
        pd.testing.assert_series_equal(
            MyData.download(0, shape=(5,), return_arr=True).data[0],
            pd.Series(
                [0.3745401188473625, 0.9507143064099162, 0.7319939418114051, 0.5986584841970366, 0.15601864044243652]
            ),
        )
        pd.testing.assert_frame_equal(
            MyData.download(0, shape=(5, 3), return_arr=True).data[0],
            pd.DataFrame(
                [
                    [0.3745401188473625, 0.9507143064099162, 0.7319939418114051],
                    [0.5986584841970366, 0.15601864044243652, 0.15599452033620265],
                    [0.05808361216819946, 0.8661761457749352, 0.6011150117432088],
                    [0.7080725777960455, 0.020584494295802447, 0.9699098521619943],
                    [0.8324426408004217, 0.21233911067827616, 0.18182496720710062],
                ]
            ),
        )
        index = pd.DatetimeIndex(
            [
                "2020-01-01 00:00:00",
                "2020-01-02 00:00:00",
                "2020-01-03 00:00:00",
                "2020-01-04 00:00:00",
                "2020-01-05 00:00:00",
            ],
            freq="D",
            tz=timezone.utc,
        )
        pd.testing.assert_series_equal(
            MyData.download(0, shape=(5,)).data[0],
            pd.Series(
                [0.3745401188473625, 0.9507143064099162, 0.7319939418114051, 0.5986584841970366, 0.15601864044243652],
                index=index,
            ),
        )
        pd.testing.assert_series_equal(
            MyData.download(0, shape=(5,), columns="feat0").data[0],
            pd.Series(
                [0.3745401188473625, 0.9507143064099162, 0.7319939418114051, 0.5986584841970366, 0.15601864044243652],
                index=index,
                name="feat0",
            ),
        )
        pd.testing.assert_frame_equal(
            MyData.download(0, shape=(5, 3)).data[0],
            pd.DataFrame(
                [
                    [0.3745401188473625, 0.9507143064099162, 0.7319939418114051],
                    [0.5986584841970366, 0.15601864044243652, 0.15599452033620265],
                    [0.05808361216819946, 0.8661761457749352, 0.6011150117432088],
                    [0.7080725777960455, 0.020584494295802447, 0.9699098521619943],
                    [0.8324426408004217, 0.21233911067827616, 0.18182496720710062],
                ],
                index=index,
            ),
        )
        pd.testing.assert_frame_equal(
            MyData.download(0, shape=(5, 3), columns=["feat0", "feat1", "feat2"]).data[0],
            pd.DataFrame(
                [
                    [0.3745401188473625, 0.9507143064099162, 0.7319939418114051],
                    [0.5986584841970366, 0.15601864044243652, 0.15599452033620265],
                    [0.05808361216819946, 0.8661761457749352, 0.6011150117432088],
                    [0.7080725777960455, 0.020584494295802447, 0.9699098521619943],
                    [0.8324426408004217, 0.21233911067827616, 0.18182496720710062],
                ],
                index=index,
                columns=pd.Index(["feat0", "feat1", "feat2"], dtype="object"),
            ),
        )
        pd.testing.assert_series_equal(
            MyData.download([0, 1], shape=(5,)).data[0],
            pd.Series(
                [0.3745401188473625, 0.9507143064099162, 0.7319939418114051, 0.5986584841970366, 0.15601864044243652],
                index=index,
            ),
        )
        pd.testing.assert_series_equal(
            MyData.download([0, 1], shape=(5,)).data[1],
            pd.Series(
                [1.3745401188473625, 1.9507143064099162, 1.7319939418114051, 1.5986584841970366, 1.15601864044243652],
                index=index,
            ),
        )
        pd.testing.assert_frame_equal(
            MyData.download([0, 1], shape=(5, 3)).data[0],
            pd.DataFrame(
                [
                    [0.3745401188473625, 0.9507143064099162, 0.7319939418114051],
                    [0.5986584841970366, 0.15601864044243652, 0.15599452033620265],
                    [0.05808361216819946, 0.8661761457749352, 0.6011150117432088],
                    [0.7080725777960455, 0.020584494295802447, 0.9699098521619943],
                    [0.8324426408004217, 0.21233911067827616, 0.18182496720710062],
                ],
                index=index,
            ),
        )
        pd.testing.assert_frame_equal(
            MyData.download([0, 1], shape=(5, 3)).data[1],
            pd.DataFrame(
                [
                    [1.3745401188473625, 1.9507143064099162, 1.7319939418114051],
                    [1.5986584841970366, 1.15601864044243652, 1.15599452033620265],
                    [1.05808361216819946, 1.8661761457749352, 1.6011150117432088],
                    [1.7080725777960455, 1.020584494295802447, 1.9699098521619943],
                    [1.8324426408004217, 1.21233911067827616, 1.18182496720710062],
                ],
                index=index,
            ),
        )
        index2 = pd.DatetimeIndex(
            [
                "2020-01-01 00:00:00",
                "2020-01-02 00:00:00",
                "2020-01-03 00:00:00",
                "2020-01-04 00:00:00",
                "2020-01-05 00:00:00",
            ],
            freq="D",
            tz=pytz.utc,
        ).tz_convert(to_timezone("Europe/Berlin"))
        pd.testing.assert_series_equal(
            MyData.download(0, shape=(5,), tz_localize="UTC", tz_convert="Europe/Berlin").data[0],
            pd.Series(
                [0.3745401188473625, 0.9507143064099162, 0.7319939418114051, 0.5986584841970366, 0.15601864044243652],
                index=index2,
            ),
        )
        index_mask = vbt.symbol_dict({0: [False, True, True, True, True], 1: [True, True, True, True, False]})
        pd.testing.assert_series_equal(
            MyData.download([0, 1], shape=(5,), index_mask=index_mask, missing_index="nan").data[0],
            pd.Series(
                [np.nan, 0.9507143064099162, 0.7319939418114051, 0.5986584841970366, 0.15601864044243652], index=index
            ),
        )
        pd.testing.assert_series_equal(
            MyData.download([0, 1], shape=(5,), index_mask=index_mask, missing_index="nan").data[1],
            pd.Series(
                [1.3745401188473625, 1.9507143064099162, 1.7319939418114051, 1.5986584841970366, np.nan], index=index
            ),
        )
        pd.testing.assert_series_equal(
            MyData.download([0, 1], shape=(5,), index_mask=index_mask, missing_index="drop").data[0],
            pd.Series([0.9507143064099162, 0.7319939418114051, 0.5986584841970366], index=index[1:4]),
        )
        pd.testing.assert_series_equal(
            MyData.download([0, 1], shape=(5,), index_mask=index_mask, missing_index="drop").data[1],
            pd.Series([1.9507143064099162, 1.7319939418114051, 1.5986584841970366], index=index[1:4]),
        )
        column_mask = vbt.symbol_dict({0: [False, True, True], 1: [True, True, False]})
        pd.testing.assert_frame_equal(
            MyData.download(
                [0, 1],
                shape=(5, 3),
                index_mask=index_mask,
                column_mask=column_mask,
                missing_index="nan",
                missing_columns="nan",
            ).data[0],
            pd.DataFrame(
                [
                    [np.nan, np.nan, np.nan],
                    [np.nan, 0.15601864044243652, 0.15599452033620265],
                    [np.nan, 0.8661761457749352, 0.6011150117432088],
                    [np.nan, 0.020584494295802447, 0.9699098521619943],
                    [np.nan, 0.21233911067827616, 0.18182496720710062],
                ],
                index=index,
            ),
        )
        pd.testing.assert_frame_equal(
            MyData.download(
                [0, 1],
                shape=(5, 3),
                index_mask=index_mask,
                column_mask=column_mask,
                missing_index="nan",
                missing_columns="nan",
            ).data[1],
            pd.DataFrame(
                [
                    [1.3745401188473625, 1.9507143064099162, np.nan],
                    [1.5986584841970366, 1.15601864044243652, np.nan],
                    [1.05808361216819946, 1.8661761457749352, np.nan],
                    [1.7080725777960455, 1.020584494295802447, np.nan],
                    [np.nan, np.nan, np.nan],
                ],
                index=index,
            ),
        )
        pd.testing.assert_frame_equal(
            MyData.download(
                [0, 1],
                shape=(5, 3),
                index_mask=index_mask,
                column_mask=column_mask,
                missing_index="drop",
                missing_columns="drop",
            ).data[0],
            pd.DataFrame(
                [[0.15601864044243652], [0.8661761457749352], [0.020584494295802447]],
                index=index[1:4],
                columns=pd.Index([1], dtype="int64"),
            ),
        )
        pd.testing.assert_frame_equal(
            MyData.download(
                [0, 1],
                shape=(5, 3),
                index_mask=index_mask,
                column_mask=column_mask,
                missing_index="drop",
                missing_columns="drop",
            ).data[1],
            pd.DataFrame(
                [[1.15601864044243652], [1.8661761457749352], [1.020584494295802447]],
                index=index[1:4],
                columns=pd.Index([1], dtype="int64"),
            ),
        )
        with pytest.raises(Exception):
            MyData.download(
                [0, 1],
                shape=(5, 3),
                index_mask=index_mask,
                column_mask=column_mask,
                missing_index="raise",
                missing_columns="nan",
            )
        with pytest.raises(Exception):
            MyData.download(
                [0, 1],
                shape=(5, 3),
                index_mask=index_mask,
                column_mask=column_mask,
                missing_index="nan",
                missing_columns="raise",
            )
        with pytest.raises(Exception):
            MyData.download(
                [0, 1],
                shape=(5, 3),
                index_mask=index_mask,
                column_mask=column_mask,
                missing_index="test",
                missing_columns="nan",
            )
        with pytest.raises(Exception):
            MyData.download(
                [0, 1],
                shape=(5, 3),
                index_mask=index_mask,
                column_mask=column_mask,
                missing_index="nan",
                missing_columns="test",
            )

    def test_update(self):
        pd.testing.assert_series_equal(
            MyData.download(0, shape=(5,), return_arr=True).update().data[0],
            pd.Series(
                [0.3745401188473625, 0.9507143064099162, 0.7319939418114051, 0.5986584841970366, 0.11505456638977896]
            ),
        )
        pd.testing.assert_series_equal(
            MyData.download(0, shape=(5,), return_arr=True).update(n=2).data[0],
            pd.Series(
                [
                    0.3745401188473625,
                    0.9507143064099162,
                    0.7319939418114051,
                    0.5986584841970366,
                    0.11505456638977896,
                    0.6090665392794814,
                ]
            ),
        )
        pd.testing.assert_frame_equal(
            MyData.download(0, shape=(5, 3), return_arr=True).update().data[0],
            pd.DataFrame(
                [
                    [0.3745401188473625, 0.9507143064099162, 0.7319939418114051],
                    [0.5986584841970366, 0.15601864044243652, 0.15599452033620265],
                    [0.05808361216819946, 0.8661761457749352, 0.6011150117432088],
                    [0.7080725777960455, 0.020584494295802447, 0.9699098521619943],
                    [0.11505456638977896, 0.6090665392794814, 0.13339096418598828],
                ]
            ),
        )
        pd.testing.assert_frame_equal(
            MyData.download(0, shape=(5, 3), return_arr=True).update(n=2).data[0],
            pd.DataFrame(
                [
                    [0.3745401188473625, 0.9507143064099162, 0.7319939418114051],
                    [0.5986584841970366, 0.15601864044243652, 0.15599452033620265],
                    [0.05808361216819946, 0.8661761457749352, 0.6011150117432088],
                    [0.7080725777960455, 0.020584494295802447, 0.9699098521619943],
                    [0.11505456638977896, 0.6090665392794814, 0.13339096418598828],
                    [0.24058961996534878, 0.3271390558111398, 0.8591374909485977],
                ]
            ),
        )
        index = pd.DatetimeIndex(
            [
                "2020-01-01 00:00:00",
                "2020-01-02 00:00:00",
                "2020-01-03 00:00:00",
                "2020-01-04 00:00:00",
                "2020-01-05 00:00:00",
            ],
            freq="D",
            tz=timezone.utc,
        )
        pd.testing.assert_series_equal(
            MyData.download(0, shape=(5,)).update().data[0],
            pd.Series(
                [0.3745401188473625, 0.9507143064099162, 0.7319939418114051, 0.5986584841970366, 0.11505456638977896],
                index=index,
            ),
        )
        updated_index = pd.DatetimeIndex(
            [
                "2020-01-01 00:00:00",
                "2020-01-02 00:00:00",
                "2020-01-03 00:00:00",
                "2020-01-04 00:00:00",
                "2020-01-05 00:00:00",
                "2020-01-06 00:00:00",
            ],
            freq="D",
            tz=timezone.utc,
        )
        pd.testing.assert_series_equal(
            MyData.download(0, shape=(5,)).update(n=2).data[0],
            pd.Series(
                [
                    0.3745401188473625,
                    0.9507143064099162,
                    0.7319939418114051,
                    0.5986584841970366,
                    0.11505456638977896,
                    0.6090665392794814,
                ],
                index=updated_index,
            ),
        )
        index2 = pd.DatetimeIndex(
            [
                "2020-01-01 00:00:00",
                "2020-01-02 00:00:00",
                "2020-01-03 00:00:00",
                "2020-01-04 00:00:00",
                "2020-01-05 00:00:00",
            ],
            freq="D",
            tz=pytz.utc,
        ).tz_convert(to_timezone("Europe/Berlin"))
        pd.testing.assert_series_equal(
            MyData.download(0, shape=(5,), tz_localize="UTC", tz_convert="Europe/Berlin")
            .update(tz_localize=None)
            .data[0],
            pd.Series(
                [0.3745401188473625, 0.9507143064099162, 0.7319939418114051, 0.5986584841970366, 0.11505456638977896],
                index=index2,
            ),
        )
        index_mask = vbt.symbol_dict({0: [False, True, True, True, True], 1: [True, True, True, True, False]})
        update_index_mask = vbt.symbol_dict({0: [True], 1: [False]})
        pd.testing.assert_series_equal(
            MyData.download([0, 1], shape=(5,), index_mask=index_mask, missing_index="nan")
            .update(index_mask=update_index_mask)
            .data[0],
            pd.Series(
                [np.nan, 0.9507143064099162, 0.7319939418114051, 0.5986584841970366, 0.11505456638977896], index=index
            ),
        )
        pd.testing.assert_series_equal(
            MyData.download([0, 1], shape=(5,), index_mask=index_mask, missing_index="nan")
            .update(index_mask=update_index_mask)
            .data[1],
            pd.Series(
                [1.3745401188473625, 1.9507143064099162, 1.7319939418114051, 1.5986584841970366, np.nan], index=index
            ),
        )
        update_index_mask2 = vbt.symbol_dict({0: [True, False], 1: [False, True]})
        pd.testing.assert_series_equal(
            MyData.download([0, 1], shape=(5,), index_mask=index_mask, missing_index="nan")
            .update(n=2, index_mask=update_index_mask2)
            .data[0],
            pd.Series(
                [np.nan, 0.9507143064099162, 0.7319939418114051, 0.5986584841970366, 0.11505456638977896, np.nan],
                index=updated_index,
            ),
        )
        pd.testing.assert_series_equal(
            MyData.download([0, 1], shape=(5,), index_mask=index_mask, missing_index="nan")
            .update(n=2, index_mask=update_index_mask2)
            .data[1],
            pd.Series(
                [
                    1.3745401188473625,
                    1.9507143064099162,
                    1.7319939418114051,
                    1.5986584841970366,
                    np.nan,
                    1.6090665392794814,
                ],
                index=updated_index,
            ),
        )
        pd.testing.assert_series_equal(
            MyData.download([0, 1], shape=(5,), index_mask=index_mask, missing_index="drop")
            .update(index_mask=update_index_mask)
            .data[0],
            pd.Series([0.9507143064099162, 0.7319939418114051, 0.5986584841970366], index=index[1:4]),
        )
        pd.testing.assert_series_equal(
            MyData.download([0, 1], shape=(5,), index_mask=index_mask, missing_index="drop")
            .update(index_mask=update_index_mask)
            .data[1],
            pd.Series([1.9507143064099162, 1.7319939418114051, 1.5986584841970366], index=index[1:4]),
        )
        pd.testing.assert_series_equal(
            MyData.download([0, 1], shape=(5,), index_mask=index_mask, missing_index="drop")
            .update(n=2, index_mask=update_index_mask2)
            .data[0],
            pd.Series([0.9507143064099162, 0.7319939418114051, 0.5986584841970366], index=index[1:4]),
        )
        pd.testing.assert_series_equal(
            MyData.download([0, 1], shape=(5,), index_mask=index_mask, missing_index="drop")
            .update(n=2, index_mask=update_index_mask2)
            .data[1],
            pd.Series([1.9507143064099162, 1.7319939418114051, 1.5986584841970366], index=index[1:4]),
        )
        column_mask = vbt.symbol_dict({0: [False, True, True], 1: [True, True, False]})
        pd.testing.assert_frame_equal(
            MyData.download(
                [0, 1],
                shape=(5, 3),
                index_mask=index_mask,
                column_mask=column_mask,
                missing_index="nan",
                missing_columns="nan",
            )
            .update(index_mask=update_index_mask)
            .data[0],
            pd.DataFrame(
                [
                    [np.nan, np.nan, np.nan],
                    [np.nan, 0.15601864044243652, 0.15599452033620265],
                    [np.nan, 0.8661761457749352, 0.6011150117432088],
                    [np.nan, 0.020584494295802447, 0.9699098521619943],
                    [np.nan, 0.6090665392794814, 0.13339096418598828],
                ],
                index=index,
            ),
        )
        pd.testing.assert_frame_equal(
            MyData.download(
                [0, 1],
                shape=(5, 3),
                index_mask=index_mask,
                column_mask=column_mask,
                missing_index="nan",
                missing_columns="nan",
            )
            .update(index_mask=update_index_mask)
            .data[1],
            pd.DataFrame(
                [
                    [1.3745401188473625, 1.9507143064099162, np.nan],
                    [1.5986584841970366, 1.15601864044243652, np.nan],
                    [1.05808361216819946, 1.8661761457749352, np.nan],
                    [1.7080725777960455, 1.020584494295802447, np.nan],
                    [np.nan, np.nan, np.nan],
                ],
                index=index,
            ),
        )
        pd.testing.assert_frame_equal(
            MyData.download(
                [0, 1],
                shape=(5, 3),
                index_mask=index_mask,
                column_mask=column_mask,
                missing_index="nan",
                missing_columns="nan",
            )
            .update(n=2, index_mask=update_index_mask2)
            .data[0],
            pd.DataFrame(
                [
                    [np.nan, np.nan, np.nan],
                    [np.nan, 0.15601864044243652, 0.15599452033620265],
                    [np.nan, 0.8661761457749352, 0.6011150117432088],
                    [np.nan, 0.020584494295802447, 0.9699098521619943],
                    [np.nan, 0.6090665392794814, 0.13339096418598828],
                    [np.nan, np.nan, np.nan],
                ],
                index=updated_index,
            ),
        )
        pd.testing.assert_frame_equal(
            MyData.download(
                [0, 1],
                shape=(5, 3),
                index_mask=index_mask,
                column_mask=column_mask,
                missing_index="nan",
                missing_columns="nan",
            )
            .update(n=2, index_mask=update_index_mask2)
            .data[1],
            pd.DataFrame(
                [
                    [1.3745401188473625, 1.9507143064099162, np.nan],
                    [1.5986584841970366, 1.15601864044243652, np.nan],
                    [1.05808361216819946, 1.8661761457749352, np.nan],
                    [1.7080725777960455, 1.020584494295802447, np.nan],
                    [np.nan, np.nan, np.nan],
                    [1.2405896199653488, 1.3271390558111398, np.nan],
                ],
                index=updated_index,
            ),
        )
        pd.testing.assert_frame_equal(
            MyData.download(
                [0, 1],
                shape=(5, 3),
                index_mask=index_mask,
                column_mask=column_mask,
                missing_index="drop",
                missing_columns="drop",
            )
            .update(index_mask=update_index_mask)
            .data[0],
            pd.DataFrame(
                [[0.15601864044243652], [0.8661761457749352], [0.020584494295802447]],
                index=index[1:4],
                columns=pd.Index([1], dtype="int64"),
            ),
        )
        pd.testing.assert_frame_equal(
            MyData.download(
                [0, 1],
                shape=(5, 3),
                index_mask=index_mask,
                column_mask=column_mask,
                missing_index="drop",
                missing_columns="drop",
            )
            .update(index_mask=update_index_mask)
            .data[1],
            pd.DataFrame(
                [[1.15601864044243652], [1.8661761457749352], [1.020584494295802447]],
                index=index[1:4],
                columns=pd.Index([1], dtype="int64"),
            ),
        )
        pd.testing.assert_frame_equal(
            MyData.download(
                [0, 1],
                shape=(5, 3),
                index_mask=index_mask,
                column_mask=column_mask,
                missing_index="drop",
                missing_columns="drop",
            )
            .update(n=2, index_mask=update_index_mask2)
            .data[0],
            pd.DataFrame(
                [[0.15601864044243652], [0.8661761457749352], [0.020584494295802447]],
                index=index[1:4],
                columns=pd.Index([1], dtype="int64"),
            ),
        )
        pd.testing.assert_frame_equal(
            MyData.download(
                [0, 1],
                shape=(5, 3),
                index_mask=index_mask,
                column_mask=column_mask,
                missing_index="drop",
                missing_columns="drop",
            )
            .update(n=2, index_mask=update_index_mask2)
            .data[1],
            pd.DataFrame(
                [[1.15601864044243652], [1.8661761457749352], [1.020584494295802447]],
                index=index[1:4],
                columns=pd.Index([1], dtype="int64"),
            ),
        )

    def test_concat(self):
        index = pd.DatetimeIndex(
            [
                "2020-01-01 00:00:00",
                "2020-01-02 00:00:00",
                "2020-01-03 00:00:00",
                "2020-01-04 00:00:00",
                "2020-01-05 00:00:00",
            ],
            freq="D",
            tz=timezone.utc,
        )
        pd.testing.assert_series_equal(
            MyData.download(0, shape=(5,), columns="feat0").concat()["feat0"],
            pd.Series(
                [0.3745401188473625, 0.9507143064099162, 0.7319939418114051, 0.5986584841970366, 0.15601864044243652],
                index=index,
                name=0,
            ),
        )
        pd.testing.assert_frame_equal(
            MyData.download([0, 1], shape=(5,), columns="feat0").concat()["feat0"],
            pd.DataFrame(
                [
                    [0.3745401188473625, 1.3745401188473625],
                    [0.9507143064099162, 1.9507143064099162],
                    [0.7319939418114051, 1.7319939418114051],
                    [0.5986584841970366, 1.5986584841970366],
                    [0.15601864044243652, 1.15601864044243652],
                ],
                index=index,
                columns=pd.Index([0, 1], dtype="int64", name="symbol"),
            ),
        )
        pd.testing.assert_series_equal(
            MyData.download(0, shape=(5, 3), columns=["feat0", "feat1", "feat2"]).concat()["feat0"],
            pd.Series(
                [0.3745401188473625, 0.5986584841970366, 0.05808361216819946, 0.7080725777960455, 0.8324426408004217],
                index=index,
                name=0,
            ),
        )
        pd.testing.assert_series_equal(
            MyData.download(0, shape=(5, 3), columns=["feat0", "feat1", "feat2"]).concat()["feat1"],
            pd.Series(
                [
                    0.9507143064099162,
                    0.15601864044243652,
                    0.8661761457749352,
                    0.020584494295802447,
                    0.21233911067827616,
                ],
                index=index,
                name=0,
            ),
        )
        pd.testing.assert_series_equal(
            MyData.download(0, shape=(5, 3), columns=["feat0", "feat1", "feat2"]).concat()["feat2"],
            pd.Series(
                [0.7319939418114051, 0.15599452033620265, 0.6011150117432088, 0.9699098521619943, 0.18182496720710062],
                index=index,
                name=0,
            ),
        )
        pd.testing.assert_frame_equal(
            MyData.download([0, 1], shape=(5, 3), columns=["feat0", "feat1", "feat2"]).concat()["feat0"],
            pd.DataFrame(
                [
                    [0.3745401188473625, 1.3745401188473625],
                    [0.5986584841970366, 1.5986584841970366],
                    [0.05808361216819946, 1.05808361216819946],
                    [0.7080725777960455, 1.7080725777960455],
                    [0.8324426408004217, 1.8324426408004217],
                ],
                index=index,
                columns=pd.Index([0, 1], dtype="int64", name="symbol"),
            ),
        )
        pd.testing.assert_frame_equal(
            MyData.download([0, 1], shape=(5, 3), columns=["feat0", "feat1", "feat2"]).concat()["feat1"],
            pd.DataFrame(
                [
                    [0.9507143064099162, 1.9507143064099162],
                    [0.15601864044243652, 1.15601864044243652],
                    [0.8661761457749352, 1.8661761457749352],
                    [0.020584494295802447, 1.020584494295802447],
                    [0.21233911067827616, 1.21233911067827616],
                ],
                index=index,
                columns=pd.Index([0, 1], dtype="int64", name="symbol"),
            ),
        )
        pd.testing.assert_frame_equal(
            MyData.download([0, 1], shape=(5, 3), columns=["feat0", "feat1", "feat2"]).concat()["feat2"],
            pd.DataFrame(
                [
                    [0.7319939418114051, 1.7319939418114051],
                    [0.15599452033620265, 1.15599452033620265],
                    [0.6011150117432088, 1.6011150117432088],
                    [0.9699098521619943, 1.9699098521619943],
                    [0.18182496720710062, 1.18182496720710062],
                ],
                index=index,
                columns=pd.Index([0, 1], dtype="int64", name="symbol"),
            ),
        )

    def test_get(self):
        index = pd.DatetimeIndex(
            [
                "2020-01-01 00:00:00",
                "2020-01-02 00:00:00",
                "2020-01-03 00:00:00",
                "2020-01-04 00:00:00",
                "2020-01-05 00:00:00",
            ],
            freq="D",
            tz=timezone.utc,
        )
        pd.testing.assert_series_equal(
            MyData.download(0, shape=(5,), columns="feat0").get(),
            pd.Series(
                [0.3745401188473625, 0.9507143064099162, 0.7319939418114051, 0.5986584841970366, 0.15601864044243652],
                index=index,
                name="feat0",
            ),
        )
        pd.testing.assert_frame_equal(
            MyData.download(0, shape=(5, 3), columns=["feat0", "feat1", "feat2"]).get(),
            pd.DataFrame(
                [
                    [0.3745401188473625, 0.9507143064099162, 0.7319939418114051],
                    [0.5986584841970366, 0.15601864044243652, 0.15599452033620265],
                    [0.05808361216819946, 0.8661761457749352, 0.6011150117432088],
                    [0.7080725777960455, 0.020584494295802447, 0.9699098521619943],
                    [0.8324426408004217, 0.21233911067827616, 0.18182496720710062],
                ],
                index=index,
                columns=pd.Index(["feat0", "feat1", "feat2"], dtype="object"),
            ),
        )
        pd.testing.assert_series_equal(
            MyData.download(0, shape=(5, 3), columns=["feat0", "feat1", "feat2"]).get("feat0"),
            pd.Series(
                [0.3745401188473625, 0.5986584841970366, 0.05808361216819946, 0.7080725777960455, 0.8324426408004217],
                index=index,
                name="feat0",
            ),
        )
        pd.testing.assert_frame_equal(
            MyData.download([0, 1], shape=(5,), columns="feat0").get(),
            pd.DataFrame(
                [
                    [0.3745401188473625, 1.3745401188473625],
                    [0.9507143064099162, 1.9507143064099162],
                    [0.7319939418114051, 1.7319939418114051],
                    [0.5986584841970366, 1.5986584841970366],
                    [0.15601864044243652, 1.15601864044243652],
                ],
                index=index,
                columns=pd.Index([0, 1], dtype="int64", name="symbol"),
            ),
        )

        class UnnamedSeriesData(vbt.Data):
            @classmethod
            def download_symbol(cls, symbol):
                return pd.Series(
                    [1.0, 2.0, 3.0],
                    index=pd.date_range("2021-01-01", "2021-01-03"),
                )

        pd.testing.assert_series_equal(
            UnnamedSeriesData.download(["RANDNX1", "RANDNX2"]).get(column="RANDNX1"),
            pd.Series(
                [1.0, 2.0, 3.0],
                index=pd.DatetimeIndex(
                    ["2021-01-01 00:00:00", "2021-01-02 00:00:00", "2021-01-03 00:00:00"],
                    freq="D",
                    tz=timezone.utc,
                ),
                name="RANDNX1",
            ),
        )
        pd.testing.assert_frame_equal(
            MyData.download([0, 1], shape=(5, 3), columns=["feat0", "feat1", "feat2"]).get("feat0"),
            pd.DataFrame(
                [
                    [0.3745401188473625, 1.3745401188473625],
                    [0.5986584841970366, 1.5986584841970366],
                    [0.05808361216819946, 1.05808361216819946],
                    [0.7080725777960455, 1.7080725777960455],
                    [0.8324426408004217, 1.8324426408004217],
                ],
                index=index,
                columns=pd.Index([0, 1], dtype="int64", name="symbol"),
            ),
        )
        pd.testing.assert_frame_equal(
            MyData.download([0, 1], shape=(5, 3), columns=["feat0", "feat1", "feat2"]).get(["feat0", "feat1"])[0],
            pd.DataFrame(
                [
                    [0.3745401188473625, 1.3745401188473625],
                    [0.5986584841970366, 1.5986584841970366],
                    [0.05808361216819946, 1.05808361216819946],
                    [0.7080725777960455, 1.7080725777960455],
                    [0.8324426408004217, 1.8324426408004217],
                ],
                index=index,
                columns=pd.Index([0, 1], dtype="int64", name="symbol"),
            ),
        )
        pd.testing.assert_frame_equal(
            MyData.download([0, 1], shape=(5, 3), columns=["feat0", "feat1", "feat2"]).get()[0],
            pd.DataFrame(
                [
                    [0.3745401188473625, 1.3745401188473625],
                    [0.5986584841970366, 1.5986584841970366],
                    [0.05808361216819946, 1.05808361216819946],
                    [0.7080725777960455, 1.7080725777960455],
                    [0.8324426408004217, 1.8324426408004217],
                ],
                index=index,
                columns=pd.Index([0, 1], dtype="int64", name="symbol"),
            ),
        )

    def test_indexing(self):
        assert (
            MyData.download([0, 1], shape=(5,), columns="feat0").iloc[:3].wrapper
            == MyData.download([0, 1], shape=(3,), columns="feat0").wrapper
        )
        assert (
            MyData.download([0, 1], shape=(5, 3), columns=["feat0", "feat1", "feat2"]).iloc[:3].wrapper
            == MyData.download([0, 1], shape=(3, 3), columns=["feat0", "feat1", "feat2"]).wrapper
        )
        assert (
            MyData.download([0, 1], shape=(5, 3), columns=["feat0", "feat1", "feat2"])["feat0"].wrapper
            == MyData.download([0, 1], shape=(5,), columns="feat0").wrapper
        )
        assert (
            MyData.download([0, 1], shape=(5, 3), columns=["feat0", "feat1", "feat2"])[["feat0"]].wrapper
            == MyData.download([0, 1], shape=(5, 1), columns=["feat0"]).wrapper
        )

    def test_stats(self):
        index_mask = vbt.symbol_dict({0: [False, True, True, True, True], 1: [True, True, True, True, False]})
        column_mask = vbt.symbol_dict({0: [False, True, True], 1: [True, True, False]})
        data = MyData.download(
            [0, 1],
            shape=(5, 3),
            index_mask=index_mask,
            column_mask=column_mask,
            missing_index="nan",
            missing_columns="nan",
            columns=["feat0", "feat1", "feat2"],
        )

        stats_index = pd.Index(
            ["Start", "End", "Period", "Total Symbols", "Null Counts: 0", "Null Counts: 1"], dtype="object"
        )
        pd.testing.assert_series_equal(
            data.stats(),
            pd.Series(
                [
                    pd.Timestamp("2020-01-01 00:00:00+0000", tz="UTC"),
                    pd.Timestamp("2020-01-05 00:00:00+0000", tz="UTC"),
                    pd.Timedelta("5 days 00:00:00"),
                    2,
                    2.3333333333333335,
                    2.3333333333333335,
                ],
                index=stats_index,
                name="agg_func_mean",
            ),
        )
        pd.testing.assert_series_equal(
            data.stats(column="feat0"),
            pd.Series(
                [
                    pd.Timestamp("2020-01-01 00:00:00+0000", tz="UTC"),
                    pd.Timestamp("2020-01-05 00:00:00+0000", tz="UTC"),
                    pd.Timedelta("5 days 00:00:00"),
                    2,
                    5,
                    1,
                ],
                index=stats_index,
                name="feat0",
            ),
        )
        pd.testing.assert_series_equal(
            data.stats(group_by=True),
            pd.Series(
                [
                    pd.Timestamp("2020-01-01 00:00:00+0000", tz="UTC"),
                    pd.Timestamp("2020-01-05 00:00:00+0000", tz="UTC"),
                    pd.Timedelta("5 days 00:00:00"),
                    2,
                    7,
                    7,
                ],
                index=stats_index,
                name="group",
            ),
        )
        pd.testing.assert_series_equal(data["feat0"].stats(), data.stats(column="feat0"))
        pd.testing.assert_series_equal(
            data.replace(wrapper=data.wrapper.replace(group_by=True)).stats(), data.stats(group_by=True)
        )
        stats_df = data.stats(agg_func=None)
        assert stats_df.shape == (3, 6)
        pd.testing.assert_index_equal(stats_df.index, data.wrapper.columns)
        pd.testing.assert_index_equal(stats_df.columns, stats_index)


# ############# updater.py ############# #


class TestDataUpdater:
    def test_update(self):
        data = MyData.download(0, shape=(5,), return_arr=True)
        updater = vbt.DataUpdater(data)
        updater.update()
        assert updater.data == data.update()
        assert updater.config["data"] == data.update()

    def test_update_every(self):
        data = MyData.download(0, shape=(5,), return_arr=True)
        kwargs = dict(call_count=0)

        class DataUpdater(vbt.DataUpdater):
            def update(self, kwargs):
                super().update()
                kwargs["call_count"] += 1
                if kwargs["call_count"] == 5:
                    raise vbt.CancelledError

        updater = DataUpdater(data)
        updater.update_every(kwargs=kwargs)
        for i in range(5):
            data = data.update()
        assert updater.data == data
        assert updater.config["data"] == data


# ############# custom.py ############# #


class FakeAperiodic:
    """Stand-in for the `aperiodic` package: deterministic bars, metrics and stream messages."""

    def __init__(self, messages=()):
        self.calls = []
        self.messages = list(messages)
        self.stream_args = None
        self.closed = False

    @staticmethod
    def bars(symbol, start, end, freq="1h"):
        index = pd.date_range(start, end, freq=freq, inclusive="left")
        base = 100.0 if "BTC" in symbol else 10.0
        close = base + np.arange(len(index), dtype=float)
        return pd.DataFrame(
            {
                "time": index.tz_localize(None),
                "open": close - 0.5,
                "high": close + 1.0,
                "low": close - 1.0,
                "close": close,
                "volume": np.full(len(index), 2.0, dtype=np.float32),
                "volume_notional": (close * 2).astype(np.float32),
            }
        )

    def _bars(self, kwargs):
        start = pd.Timestamp(kwargs["start_date"])
        end = pd.Timestamp(kwargs["end_date"]) + pd.Timedelta(days=1)
        return self.bars(kwargs["symbol"], start, end, kwargs["interval"])

    def get_ohlcv(self, **kwargs):
        self.calls.append(("ohlcv", kwargs))
        return self._bars(kwargs)

    def get_derivative_metrics(self, **kwargs):
        self.calls.append((kwargs["metric"], kwargs))
        df = self._bars(kwargs)[["time"]]
        df["funding_rate"] = np.float32(0.0001)
        return df

    def get_metrics(self, **kwargs):
        self.calls.append((kwargs["metric"], kwargs))
        df = self._bars(kwargs)[["time"]]
        df["volume_delta"] = 1.5
        return df

    def stream(self, api_key, **kwargs):
        from types import SimpleNamespace

        self.stream_args = dict(api_key=api_key, **kwargs)
        fake = self

        class Stream:
            def __enter__(self):
                return self

            def __exit__(self, *exc_info):
                fake.closed = True

            def __iter__(self):
                for channel, data, snapshot in fake.messages:
                    yield SimpleNamespace(channel=channel, data=data, snapshot=snapshot)

        return Stream()


def aperiodic_row(symbol, time, close):
    return dict(
        symbol=symbol,
        interval="1h",
        time=int(pd.Timestamp(time, tz="UTC").value // 1000),
        open=close - 0.5,
        high=close + 1.0,
        low=close - 1.0,
        close=close,
        volume=2.0,
        volume_notional=close * 2,
    )


@pytest.fixture
def fake_aperiodic(monkeypatch):
    import sys

    fake = FakeAperiodic()
    monkeypatch.setitem(sys.modules, "aperiodic", fake)
    monkeypatch.delenv("APERIODIC_API_KEY", raising=False)
    return fake


BTC = "perpetual-BTC-USDT:USDT"
ETH = "perpetual-ETH-USDT:USDT"
CHANNEL = "ohlcv.binance-futures.1h"


class TestAperiodicData:
    def test_download(self, fake_aperiodic):
        data = vbt.AperiodicData.download(
            BTC, start="2026-09-01 06:00 UTC", end="2026-09-02 00:00 UTC", interval="1h", api_key="key"
        )
        df = data.data[BTC]
        assert list(df.columns) == ["Open", "High", "Low", "Close", "Volume", "Quote volume"]
        assert (df.dtypes == np.float64).all()
        assert df.index.name == "Open time"
        assert str(df.index.tz) == "UTC"
        assert df.index[0] == pd.Timestamp("2026-09-01 06:00", tz="UTC")
        assert df.index[-1] == pd.Timestamp("2026-09-01 23:00", tz="UTC")
        _, kwargs = fake_aperiodic.calls[0]
        assert kwargs["start_date"] == kwargs["end_date"] == pd.Timestamp("2026-09-01").date()
        assert kwargs["api_key"] == "key"
        assert kwargs["output"] == "pandas"

    def test_download_empty_range(self, fake_aperiodic):
        df = vbt.AperiodicData.download_symbol(BTC, start="2026-09-02 UTC", end="2026-09-01 UTC")
        assert df.empty
        assert list(df.columns) == ["Open", "High", "Low", "Close", "Volume", "Quote volume"]
        assert fake_aperiodic.calls == []

    def test_download_metrics(self, fake_aperiodic):
        data = vbt.AperiodicData.download(
            [BTC, ETH],
            start="2026-09-01 UTC",
            end="2026-09-02 UTC",
            metrics=["funding", "flow"],
        )
        assert [c[0] for c in fake_aperiodic.calls] == ["ohlcv", "funding", "flow"] * 2
        funding = data.get("funding_rate")
        assert list(funding.columns) == [BTC, ETH]
        assert funding.dtypes.eq(np.float64).all()
        np.testing.assert_allclose(funding.values, np.float32(0.0001))
        np.testing.assert_array_equal(data.get("volume_delta").values, 1.5)

    def test_api_key_resolution(self, fake_aperiodic, monkeypatch):
        assert vbt.AperiodicData.resolve_api_key("arg") == "arg"
        monkeypatch.setenv("APERIODIC_API_KEY", "env")
        assert vbt.AperiodicData.resolve_api_key() == "env"
        vbt.settings.data["aperiodic"]["api_key"] = "settings"
        try:
            assert vbt.AperiodicData.resolve_api_key() == "settings"
        finally:
            vbt.settings.data["aperiodic"]["api_key"] = None

    def test_update(self, fake_aperiodic):
        data = vbt.AperiodicData.download(BTC, start="2026-09-01 UTC", end="2026-09-01 12:00 UTC")
        new_data = data.update(end="2026-09-01 18:00 UTC")
        _, kwargs = fake_aperiodic.calls[-1]
        assert kwargs["show_progress"] is False
        assert len(new_data.wrapper.index) == 18
        assert new_data.wrapper.index[-1] == pd.Timestamp("2026-09-01 17:00", tz="UTC")
        pd.testing.assert_frame_equal(new_data.data[BTC].iloc[:12], data.data[BTC], check_freq=False)

    def test_stream(self, fake_aperiodic):
        data = vbt.AperiodicData.download(
            [BTC, ETH], start="2026-09-01 UTC", end="2026-09-01 03:00 UTC", metrics="funding", api_key="key"
        )
        fake_aperiodic.messages = [
            (CHANNEL, aperiodic_row(BTC, "2026-09-01 02:00", 102.0), True),  # snapshot of a known bar
            (CHANNEL, aperiodic_row(BTC, "2026-09-01 03:00", 200.0), False),
            (CHANNEL, aperiodic_row(BTC, "2026-09-01 04:00", 201.0), False),  # ETH lags behind
            (CHANNEL, aperiodic_row(ETH, "2026-09-01 03:00", 20.0), False),
            (CHANNEL, aperiodic_row(ETH, "2026-09-01 04:00", 21.0), False),
            ("ohlcv.okx-perps.1h", aperiodic_row(BTC, "2026-09-01 05:00", 0.0), False),  # not subscribed
        ]
        updates = list(data.stream())
        assert fake_aperiodic.closed
        assert fake_aperiodic.stream_args["api_key"] == "key"
        assert fake_aperiodic.stream_args["channels"] == [
            dict(dataset="ohlcv", exchange="binance-futures", interval="1h", symbols=[BTC, ETH])
        ]
        assert [u.wrapper.index[-1].hour for u in updates] == [3, 4]
        close = updates[-1].get("Close")
        assert close.loc["2026-09-01 03:00", BTC] == 200.0
        assert close.loc["2026-09-01 04:00", ETH] == 21.0
        assert len(close) == 5
        assert not close.isna().any().any()
        funding = updates[-1].get("funding_rate")
        assert funding.iloc[:3].notna().all().all()
        assert funding.iloc[3:].isna().all().all()

    def test_stream_backfill(self, fake_aperiodic):
        data = vbt.AperiodicData.download(BTC, start="2026-09-01 UTC", end="2026-09-01 03:00 UTC")
        fake_aperiodic.messages = [(CHANNEL, aperiodic_row(BTC, "2026-09-01 06:00", 300.0), False)]
        (update,) = list(data.stream())
        close = update.get("Close")
        assert list(close.index.hour) == [0, 1, 2, 3, 4, 5, 6]
        assert list(close.iloc[3:6]) == [103.0, 104.0, 105.0]  # from REST
        assert close.iloc[-1] == 300.0
        _, kwargs = fake_aperiodic.calls[-1]
        assert kwargs["start_date"] == kwargs["end_date"] == pd.Timestamp("2026-09-01").date()

        (update,) = list(data.stream(backfill=False))
        assert list(update.get("Close").index.hour) == [0, 1, 2, 6]

    def test_stream_backfill_keeps_other_symbols(self, fake_aperiodic):
        data = vbt.AperiodicData.download([BTC, ETH], start="2026-09-01 UTC", end="2026-09-01 03:00 UTC")
        fake_aperiodic.messages = [
            (CHANNEL, aperiodic_row(ETH, "2026-09-01 04:00", 24.0), True),
            (CHANNEL, aperiodic_row(BTC, "2026-09-01 05:00", 205.0), False),
            (CHANNEL, aperiodic_row(ETH, "2026-09-01 05:00", 25.0), False),
        ]
        updates = list(data.stream())
        close = updates[-1].get("Close")
        assert list(close.index.hour) == [0, 1, 2, 3, 4, 5]
        assert list(close[ETH]) == [10.0, 11.0, 12.0, 13.0, 24.0, 25.0]
        assert list(close[BTC]) == [100.0, 101.0, 102.0, 103.0, 104.0, 205.0]
