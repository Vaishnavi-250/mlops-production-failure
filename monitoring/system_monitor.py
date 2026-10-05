import time
import psutil
import requests


API_URL = "http://127.0.0.1:5000/health"

LATENCY_THRESHOLD = 1.0


print("==========================================")
print("SYSTEM MONITORING")
print("==========================================")


# CPU usage
cpu_usage = psutil.cpu_percent(
    interval=1
)

# Memory usage
memory_usage = psutil.virtual_memory().percent


# API health
try:

    start_time = time.time()

    response = requests.get(
        API_URL,
        timeout=5
    )

    latency = time.time() - start_time

    if response.status_code == 200:

        api_status = "HEALTHY"

    else:

        api_status = "UNHEALTHY"

except Exception as error:

    api_status = "DOWN"
    latency = 0


print(
    "CPU Usage       :",
    f"{cpu_usage:.2f}%"
)

print(
    "Memory Usage    :",
    f"{memory_usage:.2f}%"
)

print(
    "API Status      :",
    api_status
)

print(
    "API Latency     :",
    f"{latency:.4f} seconds"
)


# Alert conditions

print("\n==========================================")
print("MONITORING STATUS")
print("==========================================")


if api_status != "HEALTHY":

    print("ALERT: API IS NOT HEALTHY")

elif latency > LATENCY_THRESHOLD:

    print(
        "ALERT: HIGH API LATENCY"
    )

elif cpu_usage > 80:

    print(
        "ALERT: HIGH CPU USAGE"
    )

elif memory_usage > 80:

    print(
        "ALERT: HIGH MEMORY USAGE"
    )

else:

    print(
        "System Status: NORMAL"
    )


print("==========================================")