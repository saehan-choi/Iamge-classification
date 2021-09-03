import time
from adafruit_servokit import ServoKit

original_time = time.time()
kit = ServoKit(channels=16)

start_time = time.time()
kit.servo[0].angle = 180
time.sleep(1)
kit.servo[0].angle = 0

print(f"original_time = {time.time()-original_time}")
print(f"total_time = {time.time()-start_time}")