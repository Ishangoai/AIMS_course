
import requests


class ClientDownloader:
    def __init__(self, url=None):
        self.url = url

    def download_and_save(self, output_filename):
        if not isinstance(self.url, str) or not self.url.strip():
            raise ValueError("URL is not set or invalid.")

        if not isinstance(output_filename, str) or not output_filename.strip():
            raise ValueError("Invalid output filename.")

        try:
            response = requests.get(self.url, stream=True)
            response.raise_for_status()

        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Request failed: {e}")

        try:
            with open(output_filename, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)

        except IOError as e:
            raise RuntimeError(f"Failed to write file: {e}")
