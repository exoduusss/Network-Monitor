# Network Monitor

This is an advanced network monitoring tool written in Python. It provides the following features:

-   **Multi-threaded checking:** Monitors multiple targets concurrently for efficiency.
-   **Real-time visualization:** Displays interactive plots of response times and a world map showing server locations.
-   **Machine learning anomaly detection:** Uses Isolation Forest to identify unusual response time patterns.
-   **Geolocation tracking:** Determines and visualizes the geographical location of the monitored servers.
-   **Performance optimization:** Includes DNS caching to reduce lookup times.
-   **REST API:** Offers an API to access the current status, historical data, and detected anomalies.
-   **SQLite database:** Stores monitoring data with proper datetime handling.
-   **Dark mode visualization:** Features a visually appealing dark theme for plots.

## Usage

1.  Save the Python code as `network_monitor.py`.
2.  Install the required libraries:
    ```bash
    pip install requests matplotlib pandas scikit-learn flask cartopy
    ```
3.  Run the script from your terminal:
    ```bash
    python network_monitor.py
    ```
4.  Follow the prompts to enter the URLs you want to monitor, the monitoring interval, duration, alert threshold, and API port.

## API Endpoints

The monitor provides a REST API accessible at `http://localhost:<your_api_port>`. The following endpoints are available:

-   `/api/status`: Returns the current status of all monitored targets, along with aggregated statistics.
-   `/api/history/<target>`: Returns the historical monitoring data for a specific target.
-   `/api/anomalies`: Returns the targets with detected response time anomalies.


## License

[MIT License](LICENSE) (You might want to add a LICENSE file later)
