"""
0.1 Python 进阶重点：async/await
LangChain/LangGraph/FastAPI 等现代框架广泛依赖异步编程来提升 I/O 密集型任务（如大模型 API 调用）的性能。
"""
import asyncio
import time

async def fetch_data(task_id: int, delay: float) -> str:
    """模拟一个异步 I/O 操作，例如网络请求或数据库查询"""
    print(f"任务 {task_id} 开始，预计耗时 {delay} 秒...")
    await asyncio.sleep(delay)  # 异步等待，交出控制权
    print(f"任务 {task_id} 完成！")
    return f"任务 {task_id} 的结果"

async def main() -> None:
    start_time = time.time()
    
    # 并发执行多个异步任务
    print("--- 开始并发执行任务 ---")
    tasks = [
        fetch_data(1, 1.5),
        fetch_data(2, 1.0),
        fetch_data(3, 0.5)
    ]
    
    # asyncio.gather 收集所有任务并等待它们完成
    results = await asyncio.gather(*tasks)
    
    end_time = time.time()
    print(f"\n所有任务执行完毕，总耗时: {end_time - start_time:.4f} 秒")
    print(f"结果: {results}")

if __name__ == "__main__":
    # 运行异步事件循环
    asyncio.run(main())
