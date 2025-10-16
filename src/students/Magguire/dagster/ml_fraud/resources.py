import dagster as dg


class FraudDataConfig(dg.ConfigurableResource):
    data_source: str = "https://raw.githubusercontent.com/aduuna/Kaggle-Data-Credit-Card-Fraud-Detection/master/samplecreditcard.csv"


# # configuration for the raw_xarray_dataset asset
# class FraudDataConfig(dg.Config):
#     # Time,V1,V2,V3,V4,V5,V6,V7,V8,V9,V10,V11,V12,V13,V14,V15,V16,V17,V18,V19,V20,
# V21,V22,V23,V24,V25,V26,V27,V28,Amount,Class
#     Time: int = pyd.Field(
#         default="reanalysis",
#         description="The product type to request"
#     )
#     variable: str = pyd.Field(
#         default="2m_temperature",
#         description="The meteorological variable to retrieve"
#     )
#     year: str = pyd.Field(
#         default="2023",
#         description="The year for which to retrieve data"
#     )
#     month: str = pyd.Field(
#         default="01",
#         description="The month for which to retrieve data"
#     )
#     day: list[str] = pyd.Field(
#         default=[f"{i:02d}" for i in range(1, 16)],
#         description="A list of days to retrieve"
#     )
#     time: list[str] = pyd.Field(
#         default=["00:00", "06:00", "12:00", "18:00"],
#         description="Times of day to retrieve data"
#     )
#     area: list[float] = pyd.Field(
#         default=[50.0, -5.0, 45.0, 5.0],
#         description="Area: [North, West, South, East]"
#     )
#     format: str = pyd.Field(
#         default="netcdf",
#         description="Format to download (e.g., netcdf)"
#     )
