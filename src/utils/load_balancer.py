#!/usr/bin/env python3
import http.server
import socketserver
import urllib.request
import sys
from itertools import cycle

PORT = 9000

class LoadBalancingHandler(http.server.BaseHTTPRequestHandler):
    backends = []
    backend_cycle = None
    
    def do_GET(self):
        self._proxy_request()
    
    def do_POST(self):
        self._proxy_request()
    
    def _proxy_request(self):
        backend = next(self.backend_cycle)
        
        # Read request body if present
        content_length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(content_length) if content_length > 0 else None
        
        # Forward request
        url = f"{backend}{self.path}"
        headers = {k: v for k, v in self.headers.items() 
                  if k.lower() not in ['host', 'connection']}
        
        try:
            req = urllib.request.Request(url, data=body, headers=headers, method=self.command)
            with urllib.request.urlopen(req, timeout=600) as response:
                self.send_response(response.status)
                for key, value in response.headers.items():
                    if key.lower() not in ['connection', 'transfer-encoding']:
                        self.send_header(key, value)
                self.end_headers()
                self.wfile.write(response.read())
        except Exception as e:
            self.send_error(502, f"Bad Gateway: {str(e)}")
    
    def log_message(self, format, *args):
        sys.stdout.write(f"{self.address_string()} - {format % args}\n")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 load_balancer.py <port1> <port2> ...")
        sys.exit(1)
    
    ports = sys.argv[1:]
    LoadBalancingHandler.backends = [f"http://localhost:{port}" for port in ports]
    LoadBalancingHandler.backend_cycle = cycle(LoadBalancingHandler.backends)
    
    with socketserver.ThreadingTCPServer(("", PORT), LoadBalancingHandler) as httpd:
        print(f"Load balancer running on port {PORT}")
        print(f"Backends: {', '.join(LoadBalancingHandler.backends)}")
        httpd.serve_forever()