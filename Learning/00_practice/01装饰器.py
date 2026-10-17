
import time
from functools import wraps

#步骤1：定义外层函数，接收func
def time_logger(func):
    

    @wraps(func)
    def wrapper(*args,**kwargs):
        print(f"开始执行函数：{func.__name__}")
        start_time = time.time()
        
        result = func(*args,**kwargs)

        end_time = time.time()
        print(f"函数执行时间：{end_time-start_time}")

        return result
    
    return wrapper
    
@time_logger
def slow_task(seconds:int):
    time.sleep(seconds)
    return "任务完成"

result = slow_task(2)
print(result)

print("-" * 30)

@time_logger
def fast_task():
    time.sleep(0.5)
    return "快速任务完成"

result2 = fast_task()
print(result2)
