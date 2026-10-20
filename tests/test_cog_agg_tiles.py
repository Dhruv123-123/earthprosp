from earthprosp import cog_agg


def test_tile_overlaps_names():
    s2 = "https://x/swir/2020/N36/ESA_WorldCover_10m_2020_v100_N36W117_SWIR.tif"
    dem = "https://x/Copernicus_DSM_COG_30_S13_00_W072_00_DEM/Copernicus_DSM_COG_30_S13_00_W072_00_DEM.tif"
    assert cog_agg.tile_overlaps(s2, (-121, 31, -108, 43))
    assert not cog_agg.tile_overlaps(s2, (0, 0, 10, 10))
    assert cog_agg.tile_overlaps(dem, (-76, -33, -65, -10))  # lower-left corner (-72, -13)
    assert not cog_agg.tile_overlaps(dem, (-76, -33, -65, -13))  # touches only
    assert cog_agg.tile_overlaps("https://x/unnamed.tif", (0, 0, 1, 1))  # unknown names are kept
