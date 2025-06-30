echo -e "url: https://cds.climate.copernicus.eu/api\nkey: $CDS_API_KEY" > ~/.cdsapirc

export DAGSTER_HOME="/workspaces/AIMS_course/.dagster_home/$GITHUB_USER"
mkdir -p "$DAGSTER_HOME"

if [ ! -f "$DAGSTER_HOME/dagster.yaml" ]; then
    # Copy the root config
    cp /workspaces/AIMS_course/.dagster_home/dagster.yaml "$DAGSTER_HOME/dagster.yaml"
    # Replace the module line with the user's module
    sed -i "s|module:.*|module: src.students.$GITHUB_USER.mlflow.era5_temperature_project.era5_pipeline.assets|" "$DAGSTER_HOME/dagster.yaml"
fi

dagster dev