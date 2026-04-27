from dataclasses import dataclass


@dataclass(frozen=True)
class GPMRegion:
    key: str
    long_name: str

    lat_south: float
    lat_north: float
    lon_west: float
    lon_east: float

    original_lat_south: float | None = None
    original_lat_north: float | None = None
    original_lon_west: float | None = None
    original_lon_east: float | None = None

    @property
    def lat_slice(self) -> slice:
        return slice(self.lat_south, self.lat_north)

    @property
    def lon_slice(self) -> slice:
        return slice(self.lon_west, self.lon_east)

    @property
    def has_original_bounds(self) -> bool:
        return (
            self.original_lat_south is not None
            and self.original_lat_north is not None
            and self.original_lon_west is not None
            and self.original_lon_east is not None
        )

    @property
    def original_lat_slice(self) -> slice:
        if not self.has_original_bounds:
            raise ValueError(f"Region {self.key!r} does not have original bounds.")
        return slice(self.original_lat_south, self.original_lat_north)

    @property
    def original_lon_slice(self) -> slice:
        if not self.has_original_bounds:
            raise ValueError(f"Region {self.key!r} does not have original bounds.")
        return slice(self.original_lon_west, self.original_lon_east)


_GPM_REGIONS: dict[str, GPMRegion] = {
    "AFC": GPMRegion(
        key="AFC",
        long_name="Africa",
        lat_south=-40,
        lat_north=35,
        lon_west=-25,
        lon_east=55,
        original_lat_south=-40,
        original_lat_north=40,
        original_lon_west=-30,
        original_lon_east=60,
    ),
    "AKA": GPMRegion(
        key="AKA",
        long_name="Alaska",
        lat_south=35,
        lat_north=60,
        lon_west=-178,
        lon_east=-130,
        original_lat_south=35,
        original_lat_north=67,
        original_lon_west=-178,
        original_lon_east=-115,
    ),
    "CIO": GPMRegion(
        key="CIO",
        long_name="Central Indian Ocean",
        lat_south=-40,
        lat_north=5,
        lon_west=55,
        lon_east=110,
        original_lat_south=-40,
        original_lat_north=10,
        original_lon_west=55,
        original_lon_east=110,
    ),
    "EPO": GPMRegion(
        key="EPO",
        long_name="Eastern Pacific Ocean",
        lat_south=-60,
        lat_north=35,
        lon_west=-178,
        lon_east=-130,
        original_lat_south=-67,
        original_lat_north=45,
        original_lon_west=-178,
        original_lon_east=-130,
    ),
    "EUR": GPMRegion(
        key="EUR",
        long_name="Europe",
        lat_south=35,
        lat_north=60,
        lon_west=-20,
        lon_east=45,
        original_lat_south=35,
        original_lat_north=67,
        original_lon_west=-20,
        original_lon_east=45,
    ),
    "H01": GPMRegion(
        key="H01",
        long_name="Pacific West of South America (Hole1)",
        lat_south=-60,
        lat_north=20,
        lon_west=-130,
        lon_east=-95,
        original_lat_south=-67,
        original_lat_north=25,
        original_lon_west=-140,
        original_lon_east=-85,
    ),
    "H02": GPMRegion(
        key="H02",
        long_name="North Atlantic (Hole2)",
        lat_south=20,
        lat_north=60,
        lon_west=-65,
        lon_east=-25,
        original_lat_south=15,
        original_lat_north=67,
        original_lon_west=-65,
        original_lon_east=-10,
    ),
    "H02p5": GPMRegion(
        key="H02p5",
        long_name="Gap between North Atlantic and Europe (bookkeeping region)",
        lat_south=35,
        lat_north=60,
        lon_west=-25,
        lon_east=-20,
        original_lat_south=None,
        original_lat_north=None,
        original_lon_west=None,
        original_lon_east=None,
    ),
    "H03": GPMRegion(
        key="H03",
        long_name="South of Africa (Hole3)",
        lat_south=-60,
        lat_north=-40,
        lon_west=-25,
        lon_east=75,
        original_lat_south=-67,
        original_lat_north=-35,
        original_lon_west=-30,
        original_lon_east=75,
    ),
    "H04": GPMRegion(
        key="H04",
        long_name="South Indian Ocean (Hole4)",
        lat_south=-60,
        lat_north=-40,
        lon_west=75,
        lon_east=178,
        original_lat_south=-67,
        original_lat_north=-35,
        original_lon_west=70,
        original_lon_east=178,
    ),
    "H05": GPMRegion(
        key="H05",
        long_name="Western Pacific (Hole5)",
        lat_south=5,
        lat_north=35,
        lon_west=130,
        lon_east=178,
        original_lat_south=5,
        original_lat_north=40,
        original_lon_west=125,
        original_lon_east=178,
    ),
    "NAM": GPMRegion(
        key="NAM",
        long_name="North America",
        lat_south=20,
        lat_north=60,
        lon_west=-130,
        lon_east=-65,
        original_lat_south=15,
        original_lat_north=67,
        original_lon_west=-140,
        original_lon_east=-55,
    ),
    "NAS": GPMRegion(
        key="NAS",
        long_name="North Asia",
        lat_south=35,
        lat_north=60,
        lon_west=45,
        lon_east=178,
        original_lat_south=35,
        original_lat_north=67,
        original_lon_west=40,
        original_lon_east=178,
    ),
    "SAM": GPMRegion(
        key="SAM",
        long_name="South America",
        lat_south=-60,
        lat_north=20,
        lon_west=-95,
        lon_east=-25,
        original_lat_south=-67,
        original_lat_north=20,
        original_lon_west=-95,
        original_lon_east=-25,
    ),
    "SAS": GPMRegion(
        key="SAS",
        long_name="South Asia",
        lat_south=5,
        lat_north=35,
        lon_west=55,
        lon_east=130,
        original_lat_south=5,
        original_lat_north=40,
        original_lon_west=55,
        original_lon_east=130,
    ),
    "WMP": GPMRegion(
        key="WMP",
        long_name="Warm Pool",
        lat_south=-40,
        lat_north=5,
        lon_west=110,
        lon_east=178,
        original_lat_south=-40,
        original_lat_north=10,
        original_lon_west=105,
        original_lon_east=178,
    ),
}


def get_region(region_key: str) -> GPMRegion:
    try:
        return _GPM_REGIONS[region_key]
    except KeyError as e:
        valid = ", ".join(_GPM_REGIONS)
        raise KeyError(f"Unknown GPM region {region_key!r}. Valid keys: {valid}") from e


def get_region_keys() -> list[str]:
    return list(_GPM_REGIONS)