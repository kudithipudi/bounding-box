bind = "unix:/var/www/bounding-box/bounding-box.sock"
workers = 2
worker_class = "uvicorn.workers.UvicornWorker"
chdir = "/var/www/bounding-box"
accesslog = "-"
errorlog = "-"

# nginx proxies to this unix socket (see /etc/nginx/sites-enabled/...),
# so the peer connection has no IP at all — uvicorn's default trusted-proxy
# check (forwarded_allow_ips="127.0.0.1") never matches a unix-socket peer.
# Safe to always-trust here since the socket is only reachable by local nginx.
forwarded_allow_ips = "*"