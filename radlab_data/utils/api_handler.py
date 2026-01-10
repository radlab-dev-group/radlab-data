import requests

from urllib.parse import urljoin


class APIHandler:
    def __init__(self, base_api_url, ping_check=False, debug: bool = False) -> None:
        self._debug = debug
        self.base_api_url = base_api_url
        if ping_check:
            self.call(endpoint="/ping")

    def call(
        self,
        endpoint: str,
        headers: dict | None = None,
        params: dict | None = None,
        method: str = "GET",
        data: dict | None = None,
    ):
        """
        Function to call an API with the given URL, headers, parameters, and method.

        Parameters:
            endpoint (str): The specific endpoint to call.
            headers (dict, optional): Headers to be included in the request.
            params (dict, optional): Parameters to be passed in the request.
            method (str, optional): HTTP method to be used ('GET', 'POST', 'PUT', 'DELETE', etc.).
            data (dict, optional): Data to be sent in the request body (for methods like POST or PUT).

        Returns:
            dict: JSON response from the API if successful, otherwise returns None.
        """
        ep_url = urljoin(self.base_api_url, endpoint)
        if self._debug:
            print("calling", ep_url)
        try:
            method = method.upper()
            # Make the request based on the specified method
            if method == "GET":
                response = requests.get(ep_url, headers=headers, params=params)
            elif method == "POST":
                response = requests.post(
                    ep_url, headers=headers, params=params, json=data
                )
            elif method == "PUT":
                response = requests.put(
                    ep_url, headers=headers, params=params, json=data
                )
            elif method == "DELETE":
                response = requests.delete(ep_url, headers=headers, params=params)
            else:
                raise ValueError("Unsupported HTTP method:", method)

            # Check if the request was successful
            response.raise_for_status()

            # Return the JSON response
            if self._debug:
                print("got response from", ep_url)
            return response.json()

        except requests.RequestException as e:
            if self._debug:
                print("Error calling API:", e)
            return None
