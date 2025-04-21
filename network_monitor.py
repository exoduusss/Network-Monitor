import requests
import time
import matplotlib.pyplot as plt
from datetime import datetime
import concurrent.futures
import pandas as pd
from urllib.parse import urlparse
import warnings
import socket
from sklearn.ensemble import IsolationForest
from flask import Flask, jsonify
import numpy as np
from contextlib import contextmanager
import sqlite3
from threading import Thread
import cartopy.crs as ccrs
import cartopy.feature as cfeature

warnings.filterwarnings("ignore", category=UserWarning)

class AdvancedNetworkMonitor:
    def __init__(self, targets, interval=60, alert_threshold=3, api_port=5000):
        """
        Enhanced network monitor with advanced features:
        - Multi-threaded checking
        - Real-time visualization
        - Machine learning anomaly detection
        - Geolocation tracking with map visualization
        - Performance optimization
        - REST API for external access
        - Proper SQLite datetime handling
        - Dark mode visualization
        """
        self.targets = targets
        self.interval = interval
        self.alert_threshold = alert_threshold
        self.api_port = api_port
        self.failure_counts = {target: 0 for target in targets}
        self.history = {target: {'timestamps': [], 'response_times': [], 'status': [], 
                              'status_codes': [], 'locations': []} for target in targets}
        self._dns_cache = {}
        self.anomaly_detectors = {target: IsolationForest(contamination=0.05) for target in targets}

        self.setup_database()
        self.setup_plots()
        self.start_api()

    def setup_database(self):
        """Initialize SQLite database with proper datetime handling"""
        def adapt_datetime(dt):
            return dt.isoformat()
        
        def convert_datetime(text):
            return datetime.fromisoformat(text.decode())

        sqlite3.register_adapter(datetime, adapt_datetime)
        sqlite3.register_converter("datetime", convert_datetime)
        
        self.conn = sqlite3.connect(
            'network_monitor.db',
            detect_types=sqlite3.PARSE_DECLTYPES
        )
        self.cursor = self.conn.cursor()
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS status_checks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME,
                target TEXT,
                status TEXT,
                response_time REAL,
                status_code INTEGER,
                country TEXT,
                city TEXT
            )
        ''')
        self.conn.commit()

    @contextmanager
    def db_session(self):
        """Provide a transactional scope around database operations"""
        try:
            yield
            self.conn.commit()
        except Exception as e:
            self.conn.rollback()
            print(f"Database error: {e}")
            raise

    def log_to_database(self, target, status, response_time, code, location):
        """Safe database logging with proper datetime handling"""
        with self.db_session():
            self.cursor.execute('''
                INSERT INTO status_checks 
                (timestamp, target, status, response_time, status_code, country, city)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                datetime.now(),
                target,
                status,
                response_time,
                code,
                location.get('country') if location else None,
                location.get('city') if location else None
            ))

    def get_history_from_db(self, target, hours=24):
        """Retrieve history from database with proper datetime conversion"""
        with self.db_session():
            self.cursor.execute('''
                SELECT timestamp, status, response_time 
                FROM status_checks 
                WHERE target = ? 
                AND timestamp >= datetime('now', ?)
                ORDER BY timestamp
            ''', (target, f'-{hours} hours'))
            return self.cursor.fetchall()

    def check_status(self, url):
        """Enhanced status check with DNS caching and geolocation"""
        try:
            parsed = urlparse(url)
            if not parsed.scheme:
                url = 'http://' + url
                parsed = urlparse(url)
            
            if parsed.netloc not in self._dns_cache:
                self._dns_cache[parsed.netloc] = socket.gethostbyname(parsed.netloc)
            
            location = self.get_geolocation(url)
            
            start_time = time.time()
            response = requests.get(
                url,
                timeout=5,
                allow_redirects=True,
                headers={'User-Agent': 'NetworkMonitor/3.0'}
            )
            response_time = (time.time() - start_time) * 1000
            
            status = 'UP' if response.status_code < 400 else 'DEGRADED'
            return status, response_time, response.status_code, location
            
        except requests.exceptions.Timeout:
            return 'TIMEOUT', 0, 408, None
        except requests.exceptions.SSLError:
            return 'SSL_ERROR', 0, 495, None
        except requests.exceptions.ConnectionError:
            return 'CONN_ERROR', 0, 503, None
        except Exception as e:
            return 'ERROR', 0, str(e), None

    def get_geolocation(self, url):
        """Get server location information"""
        try:
            domain = urlparse(url).netloc
            ip = self._dns_cache.get(domain, socket.gethostbyname(domain))
            response = requests.get(f'http://ip-api.com/json/{ip}?fields=country,city,lat,lon,isp').json()
            return {
                'country': response.get('country'),
                'city': response.get('city'),
                'lat': response.get('lat'),
                'lon': response.get('lon'),
                'isp': response.get('isp')
            }
        except:
            return None

    def detect_anomalies(self, target):
        """Machine learning anomaly detection on response times"""
        if len(self.history[target]['response_times']) < 10:
            return []
        
        X = np.array(self.history[target]['response_times']).reshape(-1, 1)
        self.anomaly_detectors[target].fit(X)
        anomalies = self.anomaly_detectors[target].predict(X)
        return [i for i, x in enumerate(anomalies) if x == -1]

    def setup_plots(self):
        """Initialize matplotlib figures with proper spacing"""
        plt.ion()
        plt.style.use('dark_background')
        
        # Create figure with adjusted spacing and size
        self.fig = plt.figure(figsize=(14, 8 + len(self.targets)*1.5), facecolor='#121212')
        
        # Use GridSpec for better control of subplot sizes
        gs = plt.GridSpec(len(self.targets) + 2, 1, height_ratios=[1.5]*len(self.targets) + [0.5, 3])

        self.axes = []
        for i in range(len(self.targets)):
            ax = self.fig.add_subplot(gs[i, 0])
            ax.set_facecolor('#1e1e1e')
            ax.grid(color='#404040', linestyle=':', alpha=0.5)
            self.axes.append(ax)
        
        self.map_ax = self.fig.add_subplot(gs[-2:, 0], projection=ccrs.PlateCarree())
        self.map_ax.set_facecolor('#1e1e1e')

        plt.tight_layout(pad=3.0, h_pad=4.0)

    def update_plots(self):
        """Enhanced visualization with location names in titles"""
        try:
            for ax in self.axes:
                ax.clear()
                ax.set_facecolor('#1e1e1e')

            for i, target in enumerate(self.targets):
                timestamps = self.history[target]['timestamps']
                response_times = self.history[target]['response_times']
                
                if not timestamps:
                    continue

                self.axes[i].plot(timestamps, response_times, 'c-', linewidth=1.5, alpha=0.8)

                anomalies = self.detect_anomalies(target)
                for idx in anomalies:
                    if idx < len(timestamps) and idx < len(response_times):
                        self.axes[i].plot(timestamps[idx], response_times[idx], 
                                        'yo', markersize=8, alpha=0.7, zorder=3)

                for j, status in enumerate(self.history[target]['status']):
                    if status != 'UP' and j < len(timestamps):
                        self.axes[i].axvline(x=j, color='r', alpha=0.2, linewidth=0.5)

                location_info = ""
                if self.history[target]['locations'] and self.history[target]['locations'][-1]:
                    loc = self.history[target]['locations'][-1]
                    if loc.get('city') and loc.get('country'):
                        location_info = f" ({loc['city']}, {loc['country']})"
                    elif loc.get('country'):
                        location_info = f" ({loc['country']})"
                
                # Styling with non-overlapping elements
                title = target[:20] + '...' if len(target) > 20 else target
                self.axes[i].set_title(f"{title}{location_info} - Response Times", 
                                     color='white', pad=10, fontsize=9)
                self.axes[i].set_ylabel("ms", color='white', fontsize=8)
                self.axes[i].tick_params(axis='both', colors='white', labelsize=8)
                self.axes[i].grid(color='#404040', linestyle=':', alpha=0.5)

                if len(timestamps) > 5:
                    self.axes[i].set_xticks(np.linspace(0, len(timestamps)-1, min(8, len(timestamps))))
                    plt.setp(self.axes[i].get_xticklabels(), rotation=30, ha='right')

            self.update_map_visualization()

            plt.tight_layout(pad=3.0, h_pad=4.0)
            plt.draw()
            plt.pause(0.01)
            
        except Exception as e:
            print(f"Plotting error: {str(e)}")

    def update_map_visualization(self):
        """Modern map visualization with anti-overlapping"""
        self.map_ax.clear()
        self.map_ax.set_facecolor('#1e1e1e')

        self.map_ax.add_feature(cfeature.LAND.with_scale('50m'), facecolor='#2d2d2d', zorder=0)
        self.map_ax.add_feature(cfeature.OCEAN.with_scale('50m'), facecolor='#1a1a2e', zorder=0)
        self.map_ax.add_feature(cfeature.COASTLINE.with_scale('50m'), edgecolor='#555555', linewidth=0.5, zorder=1)

        locations = []
        for target in self.targets:
            if self.history[target]['locations'] and self.history[target]['locations'][-1]:
                loc = self.history[target]['locations'][-1]
                if loc and 'lat' in loc and 'lon' in loc:
                    locations.append((loc['lon'], loc['lat'], target))
        
        if not locations:
            return

        plotted_positions = set()
        marker_size = 80
        
        for lon, lat, target in locations:
            domain = target.split('//')[-1].split('/')[0]

            adjusted_lon, adjusted_lat = lon, lat
            while (round(adjusted_lon,1), round(adjusted_lat,1)) in plotted_positions:
                adjusted_lon += 0.5
                adjusted_lat += 0.5
            
            plotted_positions.add((round(adjusted_lon,1), round(adjusted_lat,1)))

            self.map_ax.scatter(
                adjusted_lon, adjusted_lat,
                s=marker_size,
                c='#00ffff',
                edgecolors='white',
                linewidths=0.8,
                alpha=0.9,
                zorder=3,
                marker='o',
                transform=ccrs.PlateCarree()
            )

            va = 'bottom' if lat < 0 else 'top'
            label_offset = 1.0 if va == 'top' else -1.0
            self.map_ax.text(
                adjusted_lon, adjusted_lat + label_offset,
                domain[:15] + '...' if len(domain) > 15 else domain,
                color='white',
                fontsize=8,
                ha='center',
                va=va,
                bbox=dict(facecolor='#121212', alpha=0.7, edgecolor='none', pad=1),
                zorder=4,
                transform=ccrs.PlateCarree()
            )

        lons, lats, _ = zip(*locations)
        lon_span = max(lons) - min(lons)
        lat_span = max(lats) - min(lats)
        padding = max(5, 0.2 * max(lon_span, lat_span))
        
        self.map_ax.set_extent([
            min(lons) - padding,
            max(lons) + padding,
            min(lats) - padding,
            max(lats) + padding
        ], crs=ccrs.PlateCarree())

        self.map_ax.gridlines(
            color='#404040',
            alpha=0.3,
            linestyle='--',
            linewidth=0.5,
            draw_labels=True,
            zorder=2
        )

        self.map_ax.set_title("Server Locations", color='white', pad=10, fontsize=10, fontweight='bold')

    def start_api(self):
        """Start Flask API in a separate thread"""
        app = Flask(__name__)
        
        @app.route('/api/status')
        def get_status():
            return jsonify({
                'targets': self.targets,
                'current_status': {t: self.history[t]['status'][-1] if self.history[t]['status'] else None 
                                 for t in self.targets},
                'stats': self.get_stats()
            })
        
        @app.route('/api/history/<target>')
        def get_history(target):
            if target not in self.history:
                return jsonify({'error': 'Target not found'}), 404
            return jsonify(self.history[target])
        
        @app.route('/api/anomalies')
        def get_anomalies():
            return jsonify({t: self.detect_anomalies(t) for t in self.targets})
        
        Thread(target=lambda: app.run(port=self.api_port), daemon=True).start()

    def run_monitor(self, duration=3600):
        """Enhanced monitoring loop with all features"""
        end_time = time.time() + duration
        print(f"Monitoring started. Will run for {duration//3600}h {duration%3600//60}m")
        print(f"API available at http://localhost:{self.api_port}/api/status")
        
        try:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                while time.time() < end_time:
                    futures = {executor.submit(self.check_status, target): target 
                             for target in self.targets}
                    
                    for future in concurrent.futures.as_completed(futures):
                        target = futures[future]
                        status, response_time, code, location = future.result()
                        
                        timestamp = datetime.now().strftime("%H:%M:%S")
                        self.history[target]['timestamps'].append(timestamp)
                        self.history[target]['response_times'].append(response_time)
                        self.history[target]['status'].append(status)
                        self.history[target]['status_codes'].append(code)
                        self.history[target]['locations'].append(location)
                        
                        self.log_to_database(target, status, response_time, code, location)
                        
                        if status != 'UP':
                            self.failure_counts[target] += 1
                            if self.failure_counts[target] >= self.alert_threshold:
                                print(f"ALERT: {target} has failed {self.failure_counts[target]} times consecutively!")
                        else:
                            self.failure_counts[target] = 0
                        
                        print(f"{timestamp} - {target:40} {status:10} {str(code):8} {response_time:6.2f}ms")
                    
                    self.update_plots()
                    time.sleep(self.interval)
        
        except KeyboardInterrupt:
            print("\nMonitoring stopped by user")
        finally:
            self.generate_report()
            plt.ioff()
            plt.show()
            self.conn.close()

    def get_stats(self):
        """Calculate advanced statistics"""
        stats = {}
        for target in self.targets:
            data = self.history[target]
            if not data['response_times']:
                continue
                
            response_times = data['response_times']
            stats[target] = {
                'mean_response': np.mean(response_times),
                'max_response': max(response_times),
                'min_response': min(response_times),
                'availability': (sum(1 for s in data['status'] if s == 'UP') / len(data['status'])) * 100,
                'last_location': data['locations'][-1] if data['locations'] else None,
                'anomalies': len(self.detect_anomalies(target))
            }
        return stats

    def generate_report(self):
        """Generate enhanced report with all data"""
        report = []
        stats = self.get_stats()
        
        for target in self.targets:
            data = self.history[target]
            if not data['timestamps']:
                continue
                
            report.append({
                'Target': target,
                'Availability': f"{stats[target]['availability']:.2f}%",
                'Avg Response': f"{stats[target]['mean_response']:.2f}ms",
                'Location': f"{stats[target]['last_location']['city']}, {stats[target]['last_location']['country']}" 
                           if stats[target]['last_location'] else 'Unknown',
                'Anomalies': stats[target]['anomalies'],
                'Last Status': data['status'][-1]
            })
        
        print("\n=== Advanced Monitoring Report ===")
        print(pd.DataFrame(report).to_string(index=False))
        print(f"\nAccess detailed data via the API at http://localhost:{self.api_port}/api/status")

if __name__ == "__main__":
    print("=== Network Monitor ===")
    targets = input("Enter URLs to monitor (comma separated): ").split(',')
    targets = [url.strip() for url in targets if url.strip()]
    
    if not targets:
        print("No valid targets provided. Using defaults.")
        targets = ['https://google.com', 'https://github.com', 'https://example.com']
    
    interval = int(input("Monitoring interval in seconds (default 60): ") or 60)
    duration = int(input("Monitoring duration in seconds (default 3600): ") or 3600)
    threshold = int(input("Alert threshold (consecutive failures, default 3): ") or 3)
    api_port = int(input("API port (default 5000): ") or 5000)
    
    monitor = AdvancedNetworkMonitor(
        targets=targets,
        interval=interval,
        alert_threshold=threshold,
        api_port=api_port
    )
    monitor.run_monitor(duration)