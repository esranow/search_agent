import redis

r = redis.Redis.from_url("redis://localhost:6379")
print(r.ping())  # Should print: True
