import cdsapi
import dagster as dg


class FraudDataConfig(dg.ConfigurableResource):
    host_url: str = "https://cds.climate.copernicus.eu/api"
    api_key: str = dg.EnvVar("CDS_API_KEY")

    @property
    def client(self) -> cdsapi.Client:
        return cdsapi.Client(url=self.host_url, key=self.api_key)
