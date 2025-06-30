echo -e "url: https://cds.climate.copernicus.eu/api\nkey: $CDS_API_KEY" > ~/.cdsapirc

export DAGSTER_HOME="/workspaces/AIMS_course/.dagster_home/$GITHUB_USER"
mkdir -p "$DAGSTER_HOME"
dagster dev